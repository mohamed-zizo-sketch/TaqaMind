"""Public interfaces for industrial pipeline execution."""

from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    SHA256Digest,
    calculate_bytes_sha256,
    calculate_file_sha256,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import (
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMetadata,
    DatasetState,
    TransformationRecord,
)
from taqamind.core.interfaces.processing import (
    ColumnMapper,
    DataCleaner,
    DataIngestion,
    DataProfiler,
    FeatureEngineer,
    SemanticValidator,
    validate_approved_mapping,
    validate_processing_input,
)
from taqamind.core.interfaces.processing_models import (
    ApprovedMapping,
    ColumnMappingEntry,
    ColumnProfile,
    DatasetProfile,
    MappingMethod,
    MappingProposal,
    ProfilingResult,
)
from taqamind.core.interfaces.readiness import (
    TrainingReadinessResult,
)
from taqamind.core.interfaces.results import (
    ExecutionError,
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.splitting import (
    SplitConfiguration,
    SplitManifest,
    SplitPartition,
    SplitRole,
    SplitStrategy,
)

__all__ = [
    "ApprovedMapping",
    "ArtifactFormat",
    "ArtifactReference",
    "ColumnMapper",
    "ColumnMappingEntry",
    "ColumnProfile",
    "DataCleaner",
    "DataIngestion",
    "DataProfiler",
    "DatasetArtifact",
    "DatasetColumnSchema",
    "DatasetMetadata",
    "DatasetProfile",
    "DatasetState",
    "ExecutionError",
    "ExecutionStatus",
    "FeatureEngineer",
    "MappingMethod",
    "MappingProposal",
    "PipelineContext",
    "ProfilingResult",
    "SHA256Digest",
    "SemanticValidator",
    "SplitConfiguration",
    "SplitManifest",
    "SplitPartition",
    "SplitRole",
    "SplitStrategy",
    "StageResult",
    "StageTiming",
    "TrainingReadinessResult",
    "TransformationRecord",
    "calculate_bytes_sha256",
    "calculate_file_sha256",
    "validate_approved_mapping",
    "validate_processing_input",
]