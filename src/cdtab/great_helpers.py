import json
import re
from decimal import Decimal, localcontext
from cdtab.errors import InvalidTransitionError, TokenizerContractError
PROMPT_CONTRACT = 'great-one-field-exact-anchor-preserve-bytes-v1'

def verify_tokenizer_bridge(native, adapter):
    from transformers import GPT2Tokenizer, GPT2TokenizerFast
    if type(native) not in (GPT2Tokenizer, GPT2TokenizerFast):
        raise TokenizerContractError('unregistered native tokenizer class')
    slow = adapter._tokenizer
    if native.get_vocab() != slow.get_vocab() or set(native.all_special_ids) != set(slow.all_special_ids):
        raise TokenizerContractError('native and registered token vocabularies differ')
    if native.eos_token_id != adapter.eos_token_id:
        raise TokenizerContractError('EOS IDs differ')
    decoder = None
    if native.is_fast:
        decoder = json.loads(native.backend_tokenizer.decoder.__getstate__())
        if decoder.get('type') != 'ByteLevel':
            raise TokenizerContractError('native decoder is not ByteLevel')
    return {'contract': PROMPT_CONTRACT, 'native_class': type(native).__name__, 'native_decoder': decoder, 'adapter_fingerprint': adapter.fingerprint}

def state_for_native_prompt(engine, prompt):
    if type(prompt) is not bytes or len(prompt) > 512:
        raise InvalidTransitionError('native prompt must be at most 512 bytes')
    try:
        text = prompt.decode('ascii')
    except UnicodeDecodeError as error:
        raise InvalidTransitionError('native prompt is not ASCII') from error
    if not text.endswith(','):
        return engine.state_for_prefix(prompt)
    matches = [(n, text[len(n) + 4:-1]) for n in engine.program.spec.schema.names if text.startswith(n + ' is ')]
    if len(matches) != 1:
        raise InvalidTransitionError('native anchor must contain exactly one known field')
    field, number = matches[0]
    if len(number) > 160 or re.fullmatch('-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?', number) is None:
        raise InvalidTransitionError('unsupported native numeric anchor spelling')
    scale = engine.program.spec.schema.field(field).field_type.scale
    with localcontext() as context:
        context.prec = 192
        scaled = Decimal(number) * 10 ** scale
        if scaled != scaled.to_integral_value():
            raise InvalidTransitionError('native anchor is not on the declared precision grid')
        value = int(scaled)
    return engine.state_for_anchor(field, value)
