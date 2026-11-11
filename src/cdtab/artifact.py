import json
from cdtab.spec import _canonical
from cdtab.order_errors import ArtifactIntegrityError
FORMAT = 'cdtab-typed-alternatives-v07.0'

def encode_artifact(program):
    payload = {'format': FORMAT, 'spec': _canonical(program.extended_spec), 'policy': program.policy.to_payload(), 'program_fingerprint': program.program_fingerprint}
    data = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    if len(data) > program._limits.max_artifact_bytes:
        raise ArtifactIntegrityError('extended artifact exceeds budget')
    return data
