from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol
from cdtab.errors import TokenizerContractError

class TokenPayloadAdapter(Protocol):
    eos_token_id: int

    @property
    def token_ids(self) -> tuple[int, ...]:
        ...

    @property
    def fingerprint(self) -> str:
        ...

    def payload(self, token_id: int) -> bytes | None:
        ...

class StaticTokenAdapter:

    def __init__(self, payloads: Mapping[int, bytes], eos_token_id: int):
        if eos_token_id in payloads:
            raise TokenizerContractError('EOS must not also have a byte payload')
        if not payloads:
            raise TokenizerContractError('at least one non-EOS token is required')
        self._payloads = {int(key): bytes(value) for key, value in payloads.items()}
        if any((not payload for payload in self._payloads.values())):
            raise TokenizerContractError('non-EOS tokens cannot have empty byte payloads')
        self.eos_token_id = int(eos_token_id)

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(sorted((*self._payloads.keys(), self.eos_token_id)))

    def payload(self, token_id: int) -> bytes | None:
        return self._payloads.get(int(token_id))

    @property
    def fingerprint(self) -> str:
        payload = {'eos': self.eos_token_id, 'payloads': [(key, self._payloads[key].hex()) for key in sorted(self._payloads)]}
        return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode('ascii')).hexdigest()

class GPT2ByteBPEAdapter:

    def __init__(self, tokenizer):
        try:
            from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        except ImportError as error:
            raise TokenizerContractError('transformers GPT-2 tokenizer support is unavailable') from error
        if tokenizer.eos_token_id is None:
            raise TokenizerContractError('tokenizer has no EOS token')
        byte_encoder = bytes_to_unicode()
        inverse = {character: byte for byte, character in byte_encoder.items()}
        special_ids = set(tokenizer.all_special_ids)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self._payloads: dict[int, bytes | None] = {}
        for token_id in range(len(tokenizer)):
            if token_id in special_ids:
                self._payloads[token_id] = None
                continue
            token = tokenizer.convert_ids_to_tokens(token_id)
            try:
                payload = bytes((inverse[character] for character in token))
            except (KeyError, TypeError) as error:
                raise TokenizerContractError(f'token {token_id} is not valid GPT-2 byte-BPE') from error
            if not payload:
                raise TokenizerContractError(f'token {token_id} has an empty payload')
            self._payloads[token_id] = payload

    @classmethod
    def from_pretrained(cls, path: str | Path):
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise TokenizerContractError('transformers is unavailable') from error
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=False)
        return cls(tokenizer)

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(range(len(self._payloads)))

    def payload(self, token_id: int) -> bytes | None:
        if not 0 <= int(token_id) < len(self._payloads):
            return None
        return self._payloads[int(token_id)]

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.eos_token_id).encode('ascii'))
        for token_id in self.token_ids:
            payload = self._payloads[token_id]
            digest.update(token_id.to_bytes(8, 'little'))
            digest.update(b'\xff' if payload is None else payload)
            digest.update(b'\x00')
        return digest.hexdigest()
