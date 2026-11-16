"""Admit byte-concatenative tokenizers with singleton byte coverage."""

from types import MappingProxyType

from cdtab.errors import TokenizerContractError
from cdtab.tokenizer_base import (
    GPT2ByteBPEAdapter as _BaseGPT2Adapter,
)
from cdtab.tokenizer_base import (
    StaticTokenAdapter as _BaseStaticAdapter,
)


class StaticTokenAdapter(_BaseStaticAdapter):
    def __init__(self, payloads, eos_token_id: int):
        if type(eos_token_id) is not int or not 0 <= eos_token_id < 2**63:
            raise TokenizerContractError("EOS must be a nonnegative int64 ID")
        if any(
            type(key) is not int or not 0 <= key < 2**63 or (not isinstance(value, bytes))
            for key, value in payloads.items()
        ):
            raise TokenizerContractError("static payloads require nonnegative int64 IDs and bytes")
        super().__init__(payloads, eos_token_id)
        self._payloads = MappingProxyType(self._payloads)


class GPT2ByteBPEAdapter(_BaseGPT2Adapter):
    def __init__(self, tokenizer):
        try:
            from transformers import GPT2Tokenizer
        except ImportError as error:
            raise TokenizerContractError("transformers GPT2Tokenizer is unavailable") from error
        if type(tokenizer) is not GPT2Tokenizer:
            raise TokenizerContractError("Expected slow GPT2Tokenizer")
        special = set(tokenizer.all_special_ids)
        if any(token_id not in special for token_id in tokenizer.get_added_vocab().values()):
            raise TokenizerContractError("Added ordinary tokens are not supported")
        super().__init__(tokenizer)
        self._payloads = MappingProxyType(self._payloads)
        self._tokenizer = tokenizer

    def encode_prefix(self, prefix: bytes) -> tuple[int, ...]:
        try:
            text = prefix.decode("ascii")
        except (AttributeError, UnicodeDecodeError) as error:
            raise TokenizerContractError("row prefix must be ASCII bytes") from error
        ids = tuple(self._tokenizer.encode(text, add_special_tokens=False))
        payloads = tuple(self.payload(token_id) for token_id in ids)
        if any(payload is None for payload in payloads) or b"".join(payloads) != prefix:
            raise TokenizerContractError("encoded row prefix does not preserve its exact bytes")
        if (
            self._tokenizer.decode(
                list(ids), clean_up_tokenization_spaces=False, skip_special_tokens=False
            )
            != text
        ):
            raise TokenizerContractError("prefix detokenization disagrees with byte concatenation")
        return ids

    def check_singleton_decoding(self, witnesses: dict[int, int]) -> None:
        alphabet = sorted(witnesses)
        text = bytes(alphabet).decode("ascii")
        decoded = self._tokenizer.decode(
            [witnesses[byte] for byte in alphabet],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        if decoded != text:
            raise TokenizerContractError(
                "required-byte decoding does not preserve payload concatenation"
            )


def admit_adapter(adapter, required_bytes: frozenset[int], budget) -> dict[int, bytes | None]:
    if type(adapter) not in {StaticTokenAdapter, GPT2ByteBPEAdapter}:
        raise TokenizerContractError("Unsupported tokenizer adapter")
    ids = adapter.token_ids
    if len(set(ids)) != len(ids) or adapter.eos_token_id not in ids:
        raise TokenizerContractError("token IDs must be unique and include EOS")
    payloads = {}
    witnesses = {}
    for token_id in ids:
        budget.check()
        if type(token_id) is not int or token_id < 0:
            raise TokenizerContractError("token IDs must be nonnegative integers")
        payload = adapter.payload(token_id)
        if token_id == adapter.eos_token_id:
            if payload is not None:
                raise TokenizerContractError("EOS cannot also have an ordinary byte payload")
        elif payload is not None:
            if not isinstance(payload, bytes) or not payload:
                raise TokenizerContractError("ordinary token payload must be nonempty bytes")
            if len(payload) == 1:
                witnesses.setdefault(payload[0], token_id)
        payloads[token_id] = payload
    missing = required_bytes - witnesses.keys()
    if missing:
        raise TokenizerContractError(f"Missing single-byte tokens: {sorted(missing)}")
    if type(adapter) is GPT2ByteBPEAdapter:
        adapter.check_singleton_decoding({byte: witnesses[byte] for byte in required_bytes})
    return payloads
