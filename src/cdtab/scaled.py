def encode_scaled_integer(value: int, scale: int) -> str:
    if scale < 0:
        raise ValueError('scale must be non-negative')
    value = int(value)
    if scale == 0:
        return str(value)
    sign = '-' if value < 0 else ''
    digits = str(abs(value)).rjust(scale + 1, '0')
    return f'{sign}{digits[:-scale]}.{digits[-scale:]}'

def decode_scaled_integer(text: str, scale: int) -> int:
    if not text or not text.isascii():
        raise ValueError('numeric lexeme must be non-empty ASCII')
    sign = -1 if text.startswith('-') else 1
    unsigned = text[1:] if sign < 0 else text
    if scale == 0:
        if not unsigned.isdigit():
            raise ValueError('invalid integer lexeme')
        value = sign * int(unsigned)
    else:
        pieces = unsigned.split('.')
        if len(pieces) != 2 or not pieces[0].isdigit() or (not pieces[1].isdigit()):
            raise ValueError('invalid fixed-point lexeme')
        if len(pieces[1]) != scale:
            raise ValueError('wrong fixed-point scale')
        value = sign * int(pieces[0] + pieces[1])
    if encode_scaled_integer(value, scale) != text:
        raise ValueError('non-canonical numeric lexeme')
    return value
