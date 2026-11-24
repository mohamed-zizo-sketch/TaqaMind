"""Public interface for shared industrial data contracts."""

from taqamind.core.contracts.dataset import (
    DatasetContract,
    PredictionContext,
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
    "FeatureAvailability",
    "FeatureConstraints",
    "FeatureContract",
    "FeatureOrigin",
    "FeatureRole",
    "IssueImpact",
    "MissingValuePolicy",
    "PipelineStage",
    "PredictionContext",
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