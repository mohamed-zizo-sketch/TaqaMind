"""Deterministic canonical column mapping engine."""

import json
from datetime import UTC, datetime
from io import BytesIO

import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    IssueImpact,
    PipelineStage,
    ValidationSeverity,
)
from taqamind.core.contracts.validation import ValidationIssue, ValidationReport
from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import (
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetState,
)
from taqamind.core.interfaces.processing import (
    ColumnMapper,
    validate_approved_mapping,
    validate_processing_input,
)
from taqamind.core.interfaces.processing_models import (
    ApprovedMapping,
    ColumnMappingEntry,
    MappingMethod,
    MappingProposal,
)
from taqamind.core.interfaces.results import (
    ExecutionError,
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage


class DeterministicColumnMapper(ColumnMapper):
    """Propose and apply canonical column mappings using contract names and aliases."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        implementation_version: str = "0.1.0",
    ) -> None:
        self._storage = storage
        self._implementation_version = implementation_version

    def propose_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> MappingProposal:
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.INGESTED,
        )

        target_name = contract.target.feature.name
        candidate_features = [
            feature
            for feature in contract.all_columns
            if not (
                context.purpose == DatasetPurpose.PREDICTION
                and feature.name == target_name
            )
        ]

        exact_lookup = {feature.name: feature for feature in candidate_features}
        casefold_lookup = {
            feature.name.casefold(): feature for feature in candidate_features
        }
        alias_lookup = {
            alias.strip().casefold(): feature
            for feature in candidate_features
            for alias in feature.aliases
        }

        source_columns = [column.name for column in artifact.metadata.columns]
        matched_by_source: dict[str, ColumnMappingEntry] = {}
        claimed_canonicals: set[str] = set()

        # Pass 1: exact canonical name matches (auto-confirmed, highest priority).
        for source_col in source_columns:
            feature = exact_lookup.get(source_col)
            if feature is None or feature.name in claimed_canonicals:
                continue
            claimed_canonicals.add(feature.name)
            matched_by_source[source_col] = ColumnMappingEntry(
                source_column=source_col,
                canonical_column=feature.name,
                method=MappingMethod.EXACT,
                source_unit=feature.unit,
                canonical_unit=feature.unit,
                requires_confirmation=False,
                reason="Exact canonical column name match.",
            )

        # Pass 2: declared alias matches (require confirmation).
        for source_col in source_columns:
            if source_col in matched_by_source:
                continue
            normalized = source_col.strip().casefold()
            feature = alias_lookup.get(normalized)
            if feature is None or feature.name in claimed_canonicals:
                continue
            claimed_canonicals.add(feature.name)
            matched_by_source[source_col] = ColumnMappingEntry(
                source_column=source_col,
                canonical_column=feature.name,
                method=MappingMethod.ALIAS,
                source_unit=feature.unit,
                canonical_unit=feature.unit,
                requires_confirmation=True,
                reason=f"Matched contract alias for '{feature.name}'.",
            )

        # Pass 3: case-insensitive / whitespace-trimmed canonical name matches.
        for source_col in source_columns:
            if source_col in matched_by_source:
                continue
            normalized = source_col.strip().casefold()
            feature = casefold_lookup.get(normalized)
            if feature is None or feature.name in claimed_canonicals:
                continue
            claimed_canonicals.add(feature.name)
            matched_by_source[source_col] = ColumnMappingEntry(
                source_column=source_col,
                canonical_column=feature.name,
                method=MappingMethod.RULE_BASED,
                source_unit=feature.unit,
                canonical_unit=feature.unit,
                requires_confirmation=True,
                reason=f"Case-insensitive match for canonical column '{feature.name}'.",
            )

        entries = tuple(
            matched_by_source[source_col]
            for source_col in source_columns
            if source_col in matched_by_source
        )
        unmapped = tuple(
            source_col
            for source_col in source_columns
            if source_col not in matched_by_source
        )

        return MappingProposal(
            factory_id=context.factory_id,
            execution_id=context.execution_id,
            dataset_revision_id=artifact.metadata.revision_id,
            entries=entries,
            unmapped_columns=unmapped,
        )

    def apply_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        mapping: ApprovedMapping,
    ) -> StageResult:
        started_at = datetime.now(UTC)
        validate_approved_mapping(
            context=context,
            artifact=artifact,
            contract=contract,
            mapping=mapping,
        )

        entries = mapping.proposal.entries
        if not entries:
            return StageResult(
                context=context,
                stage=PipelineStage.COLUMN_MAPPING,
                status=ExecutionStatus.FAILED,
                input_artifact=artifact,
                report=None,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
                error=ExecutionError(
                    code="MAPPING_EMPTY_OUTPUT",
                    message="Approved mapping produced no mapped canonical columns.",
                ),
            )

        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True)

        rename_map = {
            entry.source_column: entry.canonical_column for entry in entries
        }
        canonical_order = [
            feature.name
            for feature in contract.all_columns
            if feature.name in set(rename_map.values())
        ]
        mapped_df = df[list(rename_map.keys())].rename(columns=rename_map)
        mapped_df = mapped_df[canonical_order]

        mapped_bytes = mapped_df.to_csv(index=False).encode("utf-8")
        mapped_ref = ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=calculate_bytes_sha256(mapped_bytes),
            size_bytes=len(mapped_bytes),
        )
        self._storage.put(
            factory_id=context.factory_id,
            reference=mapped_ref,
            stream=BytesIO(mapped_bytes),
        )

        config_payload = json.dumps(
            mapping.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        config_sha256 = calculate_bytes_sha256(config_payload)

        columns = tuple(
            DatasetColumnSchema(
                name=str(col),
                physical_dtype=str(mapped_df[col].dtype),
            )
            for col in mapped_df.columns
        )
        mapped_metadata = artifact.metadata.derive(
            context=context,
            state=DatasetState.MAPPED,
            row_count=len(mapped_df),
            columns=columns,
            operation="apply_mapping",
            implementation_version=self._implementation_version,
            configuration_sha256=config_sha256,
            description="Applied approved canonical column mapping.",
        )
        output_artifact = DatasetArtifact(
            metadata=mapped_metadata,
            reference=mapped_ref,
        )

        issues: list[ValidationIssue] = []
        if mapping.proposal.unmapped_columns:
            issues.append(
                ValidationIssue(
                    code="UNMAPPED_SOURCE_COLUMNS_DROPPED",
                    stage=PipelineStage.COLUMN_MAPPING,
                    severity=ValidationSeverity.INFO,
                    impact=IssueImpact.NONE,
                    message=(
                        "Unmapped source columns were excluded from the mapped revision."
                    ),
                    columns=mapping.proposal.unmapped_columns,
                )
            )

        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.COLUMN_MAPPING,),
            issues=tuple(issues),
        )
        return StageResult(
            context=context,
            stage=PipelineStage.COLUMN_MAPPING,
            status=ExecutionStatus.SUCCEEDED,
            input_artifact=artifact,
            output_artifact=output_artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )
