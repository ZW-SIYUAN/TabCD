from cdtab.primitives import ConstraintSpec
from cdtab.order_errors import SerializerContractError
SERIALIZER_CONTRACT = 'ascii-prefix-free-numeric-v1'

def validate_field_headers(spec: ConstraintSpec) -> None:
    headers = sorted(((f'{name} is '.encode('ascii'), name) for name in spec.schema.names))
    for (short, short_name), (long, long_name) in zip(headers, headers[1:]):
        if long.startswith(short):
            raise SerializerContractError(f'ambiguous field headers: {short_name!r} prefixes {long_name!r}; rename the fields before compilation')
