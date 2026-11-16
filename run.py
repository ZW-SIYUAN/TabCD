"""Fine-tune one GReaT model and compare native and constrained synthesis."""

import argparse
import gc
import logging
import math
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def generate(great, prompts, masker, device, batch_size, max_length, *, measure=True):
    """Generate one record per prompt, restarting the shared sampling seed."""
    import pandas as pd
    import torch
    from be_great.great_utils import (
        _convert_text_to_tabular_data,
        _convert_tokens_to_text,
    )
    from transformers import LogitsProcessorList, set_seed

    from cdtab.transport import BatchProcessor

    set_seed(3307)
    frames = []
    tokenizer = great.tokenizer
    tokenizer.padding_side = "left"
    great.model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter() if measure else None
    with torch.inference_mode():
        for offset in range(0, len(prompts), batch_size):
            batch = prompts[offset : offset + batch_size]
            inputs = tokenizer(
                batch, padding=True, return_tensors="pt", add_special_tokens=False
            ).to(device)
            processor = (
                None
                if masker is None
                else BatchProcessor(
                    masker,
                    tokenizer,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    tokenizer.pad_token_id,
                )
            )
            output = great.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=1.0,
                num_beams=1,
                num_return_sequences=1,
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([] if processor is None else [processor]),
            )
            controls = None if processor is None else processor.finish(output.tolist())
            frame = _convert_text_to_tabular_data(
                _convert_tokens_to_text(output, tokenizer), great.columns
            )
            if len(frame) != len(batch):
                raise RuntimeError("Native parser changed the requested record count")
            if controls is not None:
                for actual, control in zip(frame.to_dict("records"), controls):
                    for name, expected in control["row_typed"].items():
                        if str(actual[name]) != str(expected):
                            raise RuntimeError("Native parsing changed a constrained value")
            frames.append(frame)
    if device == "cuda":
        torch.cuda.synchronize()
    frame = pd.concat(frames, ignore_index=True)
    seconds = time.perf_counter() - started if started is not None else None
    return frame, seconds


def execute(args, directory):
    import torch
    from be_great import GReaT
    from datasets.utils.logging import disable_progress_bar
    from huggingface_hub import snapshot_download
    from transformers import GPT2Tokenizer, set_seed
    from transformers.utils import logging as hf_logging

    from cdtab import GPT2ByteBPEAdapter, make_masker
    from cdtab.demo import compile_demo, cvr, domains, load_config
    from cdtab.great_bridge import verify_tokenizer_bridge
    from data import computer_hardware, syn_affine

    logging.disable(logging.CRITICAL)
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    disable_progress_bar()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device
    set_seed(2207)
    model = snapshot_download(
        "distilgpt2",
        revision="2290a62682d06624634c1f46a6ad5be0f47f38aa",
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        ],
    )
    prepare = syn_affine.prepare if args.dataset == "syn-affine" else computer_hardware.prepare
    frame = prepare(directory.parent / "data")
    config = load_config(
        Path(__file__).resolve().parent / "configs/rules" / (args.dataset + ".json")
    )
    declared = domains(frame, config)
    if cvr(frame, declared, config["rules"]) != 0:
        raise RuntimeError("Training data violate the declared rules")
    started = time.perf_counter()
    program = compile_demo(frame, config)
    compile_seconds = time.perf_counter() - started
    adapter = GPT2ByteBPEAdapter(GPT2Tokenizer.from_pretrained(model))
    started = time.perf_counter()
    masker = make_masker(program, adapter)
    compile_seconds += time.perf_counter() - started
    batch_size = 8 if device == "cuda" else 2
    great = GReaT(
        model,
        experiment_dir=str(directory / "training"),
        epochs=args.epochs,
        batch_size=batch_size,
        report_to=[],
        disable_tqdm=True,
        logging_strategy="no",
        save_strategy="no",
        learning_rate=0.0001,
        seed=2207,
        data_seed=2207,
        use_cpu=device == "cpu",
        fp16=False,
        bf16=False,
        tf32=False,
        dataloader_num_workers=0,
    )
    verify_tokenizer_bridge(great.tokenizer, adapter)
    trainer = great.fit(frame, conditional_col=frame.columns[0], random_conditional_col=False)
    del trainer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    great.model.to(device)
    set_seed(3307)
    starts = great._get_start_sampler(None, None).get_start_tokens(args.n_rows)
    prompts = [
        great.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for ids in starts
    ]
    max_length = 96 if args.dataset == "syn-affine" else 256
    generate(great, prompts[:batch_size], None, device, batch_size, max_length, measure=False)
    native, native_seconds = generate(great, prompts, None, device, batch_size, max_length)
    controlled, controlled_seconds = generate(
        great, prompts, masker, device, batch_size, max_length
    )
    if len(native) != args.n_rows or len(controlled) != args.n_rows:
        raise RuntimeError("Synthesis did not return the requested row count")
    native_cvr = cvr(native, declared, config["rules"])
    controlled_cvr = cvr(controlled, declared, config["rules"])
    if controlled_cvr != 0:
        raise RuntimeError("Constrained synthesis failed independent validation")
    native.to_csv(directory / "native.csv", index=False)
    controlled.to_csv(directory / "tabcd.csv", index=False)
    return native_cvr, native_seconds, controlled_cvr, controlled_seconds, compile_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["syn-affine", "computer-hardware"])
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--n_rows", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not math.isfinite(args.epochs) or args.epochs <= 0 or args.n_rows <= 0:
        parser.error("epochs and n_rows must be positive")
    directory = Path("outputs") / args.dataset
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            native_cvr, native_seconds, controlled_cvr, controlled_seconds, compile_seconds = (
                execute(args, directory)
            )
    print(f"{'Method':<10} {'Rows':>6} {'CVR (%)':>10} {'Generation (s)':>16}")
    print(f"{'Native':<10} {args.n_rows:>6} {native_cvr * 100:>10.2f} {native_seconds:>16.3f}")
    print(
        f"{'+TabCD':<10} {args.n_rows:>6} {controlled_cvr * 100:>10.2f} {controlled_seconds:>16.3f}"
    )
    print(f"TabCD compile (s): {compile_seconds:.3f}")


if __name__ == "__main__":
    main()
