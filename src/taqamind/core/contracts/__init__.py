"""Public interface for shared industrial data contracts."""

from taqamind.core.contracts.dataset import (
    DatasetContract,
    PredictionContext,
)
from taqamind.core.contracts.domain import (
    DomainContract,
    IndustrialTaskContract,
    ProcessContract,
)
from taqamind.core.contracts.enums import (
    ConfirmationStatus,
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureOrigin,
    FeatureRole,
    IssueImpact,
    MissingValuePolicy,
    PipelineStage,
    PlatformModule,
    ReadinessStatus,
    TargetKind,
    TaskType,
    ValidationSeverity,
)
from taqamind.core.contracts.feature import (
    FeatureConstraints,
    FeatureContract,
)
from taqamind.core.contracts.target import (
    RateDefinition,
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
)
from taqamind.core.contracts.validation import (
    ValidationIssue,
    ValidationReport,
    ValidationSummary,
)

__all__ = [
    "ConfirmationStatus",
    "DatasetContract",
    "DatasetPurpose",
    "DataType",
    "DomainContract",
    "FeatureAvailability",
    "FeatureConstraints",
    "FeatureContract",
    "FeatureOrigin",
    "FeatureRole",
    "IndustrialTaskContract",
    "IssueImpact",
    "MissingValuePolicy",
    "PipelineStage",
    "PlatformModule",
    "PredictionContext",
    "ProcessContract",
    "RateDefinition",
    "ReadinessStatus",
    "TargetConfirmation",
    "TargetContract",
    "TargetDefinition",
    "TargetKind",
    "TaskType",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "ValidationSummary",
]