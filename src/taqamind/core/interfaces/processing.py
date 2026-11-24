"""Shared protocols for industrial data processing stages."""

from typing import Protocol, runtime_checkable

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import DatasetPurpose
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import (
    DatasetArtifact,
    DatasetState,
)
from taqamind.core.interfaces.processing_models import (
    ApprovedMapping,
    MappingProposal,
    ProfilingResult,
)
from taqamind.core.interfaces.results import StageResult


def validate_processing_input(
    *,
    context: PipelineContext,
    artifact: DatasetArtifact,
    contract: DatasetContract,
    expected_state: DatasetState,
) -> None:
    """Validate common input requirements before processing."""

    context.validate_dataset_contract(contract)

    artifact.metadata.validate_scope(context)

    if artifact.metadata.state != expected_state:
        raise ValueError(
            f"Expected dataset state '{expected_state.value}', "
            f"received '{artifact.metadata.state.value}'."
        )

    if expected_state == DatasetState.RAW:
        artifact.metadata.validate_input_context(context)


def validate_approved_mapping(
    *,
    context: PipelineContext,
    artifact: DatasetArtifact,
    contract: DatasetContract,
    mapping: ApprovedMapping,
) -> None:
    """Validate an approved mapping against its source dataset."""

    validate_processing_input(
        context=context,
        artifact=artifact,
        contract=contract,
        expected_state=DatasetState.INGESTED,
    )

    proposal = mapping.proposal

    if proposal.factory_id != context.factory_id:
        raise ValueError(
            "Mapping factory mismatch."
        )

    if proposal.execution_id != context.execution_id:
        raise ValueError(
            "Mapping execution mismatch."
        )

    if (
        proposal.dataset_revision_id
        != artifact.metadata.revision_id
    ):
        raise ValueError(
            "Mapping dataset revision mismatch."
        )

    source_names = {
        column.name
        for column in artifact.metadata.columns
    }

    mapped_sources = {
        entry.source_column
        for entry in proposal.entries
    }

    unmapped_sources = set(proposal.unmapped_columns)

    if mapped_sources | unmapped_sources != source_names:
        raise ValueError(
            "Mapping must account for every source column."
        )

    canonical_columns = contract.column_map

    mapped_targets = set()

    for entry in proposal.entries:
        if entry.canonical_column not in canonical_columns:
            raise ValueError(
                f"Unknown canonical column: {entry.canonical_column}"
            )

        feature = canonical_columns[entry.canonical_column]

        if entry.canonical_unit != feature.unit:
            raise ValueError(
                f"Canonical unit mismatch for '{feature.name}'."
            )

        if (
            feature.unit is not None
            and entry.source_unit is None
        ):
            raise ValueError(
                f"Unknown source unit for '{feature.name}'."
            )

        if (
            context.purpose == DatasetPurpose.PREDICTION
            and feature.name == contract.target.feature.name
        ):
            raise ValueError(
                "Prediction mapping cannot include the target."
            )

        mapped_targets.add(entry.canonical_column)

    if context.purpose != DatasetPurpose.RESEARCH:
        required = set(
            contract.required_columns(context.purpose)
        )

        missing = required - mapped_targets

        if missing:
            raise ValueError(
                f"Required canonical columns are missing: "
                f"{sorted(missing)}"
            )


@runtime_checkable
class DataIngestion(Protocol):
    """Parse and register an original dataset artifact."""

    def ingest(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        """Transform a RAW artifact into an INGESTED revision."""
        ...


@runtime_checkable
class DataProfiler(Protocol):
    """Inspect dataset structure and observed statistics."""

    def profile(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> ProfilingResult:
        """Profile an INGESTED artifact without modifying it."""
        ...


@runtime_checkable
class ColumnMapper(Protocol):
    """Propose and apply canonical industrial column mappings."""

    def propose_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> MappingProposal:
        """Create a proposal without changing the dataset."""
        ...

    def apply_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        mapping: ApprovedMapping,
    ) -> StageResult:
        """Transform an INGESTED artifact into a MAPPED revision."""
        ...


@runtime_checkable
class SemanticValidator(Protocol):
    """Validate industrial meaning, units and value constraints."""

    def validate(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        """Inspect a MAPPED artifact without modifying it."""
        ...


@runtime_checkable
class DataCleaner(Protocol):
    """Apply deterministic and approved data-cleaning rules."""

    def clean(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        """Transform a MAPPED artifact into a CLEANED revision."""
        ...


@runtime_checkable
class FeatureEngineer(Protocol):
    """Generate deterministic and leakage-safe industrial features."""

    def engineer(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        """Transform a CLEANED artifact into an ENGINEERED revision."""
        ...