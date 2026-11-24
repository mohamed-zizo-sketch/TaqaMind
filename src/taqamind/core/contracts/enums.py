"""Shared enumerations for industrial data contracts."""

from enum import StrEnum


class DataType(StrEnum):
    """Supported logical data types."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    CATEGORY = "category"
    DATE = "date"
    DATETIME = "datetime"


class FeatureRole(StrEnum):
    """The purpose of a column within a dataset."""

    PREDICTOR = "predictor"
    TARGET = "target"
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    METADATA = "metadata"
    CONSTRAINT = "constraint"
    KPI = "kpi"


class FeatureAvailability(StrEnum):
    """When a feature becomes available."""

    STATIC = "static"
    PRE_PRODUCTION = "pre_production"
    IN_PRODUCTION = "in_production"
    POST_PRODUCTION = "post_production"
    UNKNOWN = "unknown"


class TaskType(StrEnum):
    """Supported analytical and machine learning task types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    OPTIMIZATION = "optimization"
    DESCRIPTIVE_ANALYTICS = "descriptive_analytics"


class ValidationSeverity(StrEnum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DatasetPurpose(StrEnum):
    """The intended usage of a dataset."""

    TRAINING = "training"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"
    RESEARCH = "research"


class ReadinessStatus(StrEnum):
    """Dataset readiness for the requested operation."""

    READY = "ready"
    LIMITED = "limited"
    BLOCKED = "blocked"

class MissingValuePolicy(StrEnum):
    """Allowed strategies for missing feature values."""

    REJECT = "reject"
    ALLOW = "allow"
    IMPUTE = "impute"


class FeatureOrigin(StrEnum):
    """How a feature value is obtained."""

    MEASURED = "measured"
    RECORDED = "recorded"
    DERIVED = "derived"
    CONFIGURED = "configured"
    UNKNOWN = "unknown"

class TargetKind(StrEnum):
    """The measurement semantics of a prediction target."""

    QUANTITY = "quantity"
    RATE = "rate"
    CONTINUOUS = "continuous"
    CATEGORY = "category"
    COUNT = "count"


class ConfirmationStatus(StrEnum):
    """Business semantics confirmation status."""

    UNCONFIRMED = "unconfirmed"
    CONFIRMED = "confirmed"

class IssueImpact(StrEnum):
    """Operational impact of a validation issue."""

    NONE = "none"
    LIMITED = "limited"
    BLOCKING = "blocking"


class PipelineStage(StrEnum):
    """Stages that can produce validation reports."""

    CONTRACT_VALIDATION = "contract_validation"
    INGESTION = "ingestion"
    PROFILING = "profiling"
    COLUMN_MAPPING = "column_mapping"
    SEMANTIC_VALIDATION = "semantic_validation"
    LINEAGE = "lineage"
    CLEANING = "cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    READINESS = "readiness"
    SPLITTING = "splitting"
    TRAINING = "training"
    EVALUATION = "evaluation"
    REGISTRY = "registry"
    PREDICTION = "prediction"