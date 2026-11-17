# From Rules to Rows: Exact Constrained Decoding for LLM-Based Tabular Synthesis

[Project website and interactive demo](https://zw-siyuan.github.io/TabCD/) · [Paper](https://zw-siyuan.github.io/TabCD/ICDE2027.pdf)

`src/cdtab/` implements the paper's TabCD compiler, exact projections, token masks, and reuse. Use Python 3.12. Internet access downloads pinned distilgpt2 weights from [Hugging Face](https://huggingface.co/distilgpt2) and the Computer Hardware dataset from UCI.

```sh
pip install -e .
python run.py --dataset syn-affine
python run.py --dataset computer-hardware
python -m unittest discover -s tests -v
```

For exact dependency versions, install with `pip install -r requirements-lock.txt -e .`. CUDA is selected when available; otherwise the runner uses CPU.
Use `--epochs 0.05 --n_rows 8` for a small run. Generated CSV files are saved in `outputs/`. Generation time excludes the shared native warmup; TabCD compilation is reported separately.

| Directory | Contents |
| --- | --- |
| `src/cdtab/` | Rule compiler, exact projections, token masks, reuse, GReaT bridge |
| `configs/rules/` | Syn-Affine and Computer Hardware constraints |
| `data/` | Synthetic generator and [UCI Computer Hardware](https://archive.ics.uci.edu/dataset/29/computer+hardware) downloader |

Example: Syn-Affine with `--epochs 0.05 --n_rows 8` on CPU:
```text
Method       Rows    CVR (%)   Generation (s)
Native          8     100.00            4.302
+TabCD          8       0.00            0.555
TabCD compile (s): 0.307
```

Tested with Python 3.12.16 on an AMD Ryzen 7 9800X3D CPU using four threads: the small Syn-Affine and Computer Hardware runs took about 14 and 29 seconds with cached downloads. MIT License: see [LICENSE](LICENSE).
