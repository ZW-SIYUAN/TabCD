from __future__ import annotations

class CDTabError(Exception):
    code = 'CDTAB_ERROR'

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class ConstraintTypeError(CDTabError):
    code = 'CONSTRAINT_TYPE_ERROR'

class UnsupportedConstraintError(CDTabError):
    code = 'UNSUPPORTED_OPERATOR'

class CompileLimitError(CDTabError):
    code = 'COMPILE_LIMIT'

class EmptyRelationError(CDTabError):
    code = 'EMPTY_RELATION'

class InfeasibleAnchorError(CDTabError):
    code = 'INFEASIBLE_ANCHOR'

class InvalidTransitionError(CDTabError):
    code = 'INVALID_TRANSITION'

class TokenizerContractError(CDTabError):
    code = 'UNSUPPORTED_TOKENIZER'

class InvalidModelLogitsError(CDTabError):
    code = 'INVALID_MODEL_LOGITS'

class ZeroLegalModelMassError(CDTabError):
    code = 'ZERO_LEGAL_MODEL_MASS'
