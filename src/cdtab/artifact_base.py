import json
from cdtab.spec import _canonical
FORMAT = 'cdtab-factorized-declarative-v1'

def encode_artifact(program, budget):
    from cdtab.compiler import COMPILER_CONTRACT, SERIALIZER_CONTRACT
    payload = {'format': FORMAT, 'compiler': COMPILER_CONTRACT, 'serializer': SERIALIZER_CONTRACT, 'spec': _canonical(program.spec), 'policy': program.policy.to_payload(), 'program_fingerprint': program.program_fingerprint}
    data = bytearray()
    for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=True).iterencode(payload):
        budget.check()
        part = chunk.encode('ascii')
        if len(data) + len(part) > program._limits.max_artifact_bytes:
            budget.fail('declarative artifact exceeds max_artifact_bytes')
        data.extend(part)
    return bytes(data)
