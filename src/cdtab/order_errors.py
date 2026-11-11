from __future__ import annotations
from cdtab.errors import CDTabError

class OrderPolicyError(CDTabError):
    code = 'ORDER_POLICY_ERROR'

class OrderSelectionError(CDTabError):
    code = 'ORDER_SELECTION_ERROR'

class RelationMaterializationLimitError(CDTabError):
    code = 'RELATION_MATERIALIZATION_LIMIT'

class ArtifactIntegrityError(CDTabError):
    code = 'ARTIFACT_INTEGRITY'

class SerializerContractError(CDTabError):
    code = 'UNSUPPORTED_SERIALIZER'

class ResourceMonitorError(CDTabError):
    code = 'RESOURCE_MONITOR_UNAVAILABLE'
