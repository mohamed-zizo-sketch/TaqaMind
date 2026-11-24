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