"""Unit-aware ColumnMapper adapter mitigating Core AUD-001 until fixed in Core."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

import pandas as pd

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    IssueImpact,
    PipelineStage,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from taqamind.core.interfaces import (
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    ColumnMapper,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetState,
    ExecutionError,
    ExecutionStatus,
    MappingProposal,
    PipelineContext,
    StageResult,
    StageTiming,
    calculate_bytes_sha256,
    validate_approved_mapping,
)
from taqamind.core.mapping import DeterministicColumnMapper
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.domains.textile.common.units import (
    UnsupportedUnitConversionError,
    get_unit_conversion_factor,
)


class UnitAwareColumnMapperAdapter(ColumnMapper):
    """Wrap Core DeterministicColumnMapper to enforce verified unit conversions (AUD-001).

    Migration path: Remove once Core `DeterministicColumnMapper.apply_mapping` converts
    or blocks mismatched `source_unit` -> `canonical_unit`.
    """

    def __init__(
        self,
        storage: LocalDatasetArtifactStorage,
        *,
        inner_mapper: ColumnMapper | None = None,
        implementation_version: str = "0.1.0",
    ) -> None:
        self._storage = storage
        self._inner = inner_mapper or DeterministicColumnMapper(storage)
        self._implementation_version = implementation_version

    def propose_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> MappingProposal:
        return self._inner.propose_mapping(
            context=context, artifact=artifact, contract=contract
        )

    def apply_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        mapping: ApprovedMapping,
    ) -> StageResult:
        entries = mapping.proposal.entries
        needs_conversion = any(
            entry.source_unit != entry.canonical_unit
            for entry in entries
            if entry.canonical_unit is not None
        )
        if not needs_conversion:
            return self._inner.apply_mapping(
                context=context,
                artifact=artifact,
                contract=contract,
                mapping=mapping,
            )

        started_at = datetime.now(UTC)
        validate_approved_mapping(
            context=context,
            artifact=artifact,
            contract=contract,
            mapping=mapping,
        )

        factors: dict[str, float] = {}
        for entry in entries:
            if entry.canonical_unit is None:
                continue
            if entry.source_unit is None:
                return self._failed(
                    context, artifact, started_at, "UNIT_SOURCE_UNKNOWN",
                    f"Missing source_unit for '{entry.canonical_column}'.",
                )
            try:
                factors[entry.canonical_column] = get_unit_conversion_factor(
                    entry.source_unit, entry.canonical_unit
                )
            except UnsupportedUnitConversionError as exc:
                return self._failed(
                    context, artifact, started_at, "UNIT_CONVERSION_UNSUPPORTED", str(exc)
                )

        # Preserve string/category dtype to avoid leading-zero loss (AUD-004).
        dtype_hints = self._string_dtype_hints(mapping, contract)
        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True, dtype=dtype_hints)

        rename_map = {entry.source_column: entry.canonical_column for entry in entries}
        canonical_order = [
            f.name for f in contract.all_columns if f.name in set(rename_map.values())
        ]
        mapped_df = df[list(rename_map.keys())].rename(columns=rename_map)[canonical_order].copy()

        for col_name, factor in factors.items():
            if factor != 1.0:
                try:
                    numeric = pd.to_numeric(mapped_df[col_name], errors="raise")
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Column '{col_name}' contains non-numeric values during unit conversion."
                    ) from exc
                mapped_df[col_name] = numeric * factor

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

        columns = tuple(
            DatasetColumnSchema(name=str(col), physical_dtype=str(mapped_df[col].dtype))
            for col in mapped_df.columns
        )
        mapped_metadata = artifact.metadata.derive(
            context=context,
            state=DatasetState.MAPPED,
            row_count=len(mapped_df),
            columns=columns,
            operation="apply_mapping_with_unit_conversion",
            implementation_version=self._implementation_version,
            description="Applied approved mapping with verified SI unit conversions.",
        )
        issues: list[ValidationIssue] = []
        if mapping.proposal.unmapped_columns and context.purpose != DatasetPurpose.RESEARCH:
            issues.append(
                ValidationIssue(
                    code="UNMAPPED_SOURCE_COLUMNS_DROPPED",
                    stage=PipelineStage.COLUMN_MAPPING,
                    severity=ValidationSeverity.INFO,
                    impact=IssueImpact.NONE,
                    message="Unmapped source columns were excluded from the mapped revision.",
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
            output_artifact=DatasetArtifact(metadata=mapped_metadata, reference=mapped_ref),
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )

    @staticmethod
    def _string_dtype_hints(
        mapping: ApprovedMapping, contract: DatasetContract
    ) -> dict[str, str]:
        hints: dict[str, str] = {}
        col_map = contract.column_map
        for entry in mapping.proposal.entries:
            feature = col_map.get(entry.canonical_column)
            if feature is not None and feature.data_type.value in {"string", "category"}:
                hints[entry.source_column] = "string"
        return hints

    @staticmethod
    def _failed(
        context: PipelineContext,
        artifact: DatasetArtifact,
        started_at: datetime,
        code: str,
        message: str,
    ) -> StageResult:
        return StageResult(
            context=context,
            stage=PipelineStage.COLUMN_MAPPING,
            status=ExecutionStatus.FAILED,
            input_artifact=artifact,
            report=None,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
            error=ExecutionError(code=code, message=message),
        )
