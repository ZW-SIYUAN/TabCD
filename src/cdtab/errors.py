"""Errors raised by constraint admission and native-token decoding."""

from __future__ import annotations


class CDTabError(Exception):
    code = "CDTAB_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConstraintTypeError(CDTabError):
    code = "CONSTRAINT_TYPE_ERROR"


class UnsupportedConstraintError(CDTabError):
    code = "UNSUPPORTED_OPERATOR"


class EmptyRelationError(CDTabError):
    code = "EMPTY_RELATION"


class InvalidTransitionError(CDTabError):
    code = "INVALID_TRANSITION"


class TokenizerContractError(CDTabError):
    code = "UNSUPPORTED_TOKENIZER"


class InvalidModelLogitsError(CDTabError):
    code = "INVALID_MODEL_LOGITS"


class ZeroLegalModelMassError(CDTabError):
    code = "ZERO_LEGAL_MODEL_MASS"


class OrderPolicyError(CDTabError):
    code = "ORDER_POLICY_ERROR"


class OrderSelectionError(CDTabError):
    code = "ORDER_SELECTION_ERROR"


class RelationMaterializationLimitError(CDTabError):
    code = "RELATION_MATERIALIZATION_LIMIT"


class ArtifactIntegrityError(CDTabError):
    code = "ARTIFACT_INTEGRITY"


class SerializerContractError(CDTabError):
    code = "UNSUPPORTED_SERIALIZER"


class ResourceMonitorError(CDTabError):
    code = "RESOURCE_MONITOR_UNAVAILABLE"
