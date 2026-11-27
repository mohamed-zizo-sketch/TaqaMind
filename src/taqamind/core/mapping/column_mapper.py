"""Deterministic canonical column mapping engine."""

from datetime import UTC, datetime
from io import BytesIO

import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    DataType,
    IssueImpact,
    PipelineStage,
    ValidationSeverity,
)
from taqamind.core.contracts.validation import ValidationIssue, ValidationReport
from taqamind.core.ingestion.artifact_csv import csv_configuration_sha256, read_verified_csv
from taqamind.core.ingestion.resource_policy import (
    CsvResourcePolicy,
    resolve_internal_csv_policy,
)
from taqamind.core.ingestion.safe_csv import SafeCsvError, SafeCsvReader
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
from taqamind.core.mapping.unit_conversion import (
    UnitConversionError,
    UnitConverter,
)


class DeterministicColumnMapper(ColumnMapper):
    """Propose and apply canonical column mappings using contract names and aliases."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        implementation_version: str = "0.4.0",
        resource_policy: CsvResourcePolicy | None = None,
    ) -> None:
        self._storage = storage
        self._implementation_version = implementation_version
        self._unit_converter = UnitConverter()
        self._resource_policy = resolve_internal_csv_policy(resource_policy)
        self._reader = SafeCsvReader(resource_policy=self._resource_policy)

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
            if not (context.purpose == DatasetPurpose.PREDICTION and feature.name == target_name)
        ]

        exact_lookup = {feature.name: feature for feature in candidate_features}
        casefold_lookup = {feature.name.casefold(): feature for feature in candidate_features}
        alias_lookup = {
            alias.strip().casefold(): feature
            for feature in candidate_features
            for alias in feature.aliases
        }

        source_columns = [column.name for column in artifact.metadata.columns]
        matched_by_source: dict[str, ColumnMappingEntry] = {}
        claimed_canonicals: set[str] = set()

        # Pass 1: exact canonical name matches (highest priority).
        for source_col in source_columns:
            feature = exact_lookup.get(source_col)
            if feature is None or feature.name in claimed_canonicals:
                continue
            claimed_canonicals.add(feature.name)
            matched_by_source[source_col] = ColumnMappingEntry(
                source_column=source_col,
                canonical_column=feature.name,
                method=MappingMethod.EXACT,
                source_unit=None,
                canonical_unit=feature.unit,
                requires_confirmation=feature.unit is not None,
                reason=(
                    "Exact canonical column name match. Source unit requires explicit confirmation."
                    if feature.unit is not None
                    else "Exact canonical column name match."
                ),
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
                source_unit=None,
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
                source_unit=None,
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
            source_col for source_col in source_columns if source_col not in matched_by_source
        )

        return MappingProposal(
            factory_id=context.factory_id,
            execution_id=context.execution_id,
            dataset_revision_id=artifact.metadata.revision_id,
            entries=entries,
            unmapped_columns=unmapped,
        )

    def _apply_unit_conversions(
        self,
        *,
        df: pd.DataFrame,
        contract: DatasetContract,
        mapping: ApprovedMapping,
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Apply explicitly confirmed and compatible unit conversions."""

        converted_df = df.copy()
        issues: list[ValidationIssue] = []
        confirmed_columns = set(mapping.confirmed_source_columns)

        for entry in mapping.proposal.entries:
            feature = contract.column_map[entry.canonical_column]
            source_unit = entry.source_unit
            canonical_unit = entry.canonical_unit

            if source_unit is None and canonical_unit is None:
                continue

            if source_unit is None or canonical_unit is None:
                raise UnitConversionError(f"Incomplete unit definition for '{feature.name}'.")

            if entry.source_column not in confirmed_columns:
                raise UnitConversionError(
                    f"Source unit for '{entry.source_column}' has not been explicitly confirmed."
                )

            if source_unit == canonical_unit:
                # Confirmed identical units, including domain-specific compound units.
                continue

            if feature.data_type not in {DataType.INTEGER, DataType.FLOAT}:
                raise UnitConversionError(
                    f"Unit conversion requires a numeric feature: '{feature.name}'."
                )

            convert = (
                self._unit_converter.convert_integer_series
                if feature.data_type == DataType.INTEGER
                else self._unit_converter.convert_series
            )
            converted_df[entry.source_column] = convert(
                converted_df[entry.source_column],
                source_unit=source_unit,
                canonical_unit=canonical_unit,
            )

            issues.append(
                ValidationIssue(
                    code="UNIT_CONVERSION_APPLIED",
                    stage=PipelineStage.COLUMN_MAPPING,
                    severity=ValidationSeverity.INFO,
                    impact=IssueImpact.NONE,
                    message=(
                        f"Converted '{entry.source_column}' "
                        f"from '{source_unit}' to '{canonical_unit}' "
                        f"using UnitConverter v{self._unit_converter.VERSION}."
                    ),
                    columns=(entry.canonical_column,),
                )
            )

        return converted_df, issues

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

        try:
            df = read_verified_csv(
                storage=self._storage,
                factory_id=context.factory_id,
                reference=artifact.reference,
                reader=self._reader,
                policy=self._resource_policy,
            )

            expected_columns = tuple(col.name for col in artifact.metadata.columns)
            if tuple(df.columns) != expected_columns or len(df) != artifact.metadata.row_count:
                raise SafeCsvError(
                    "MAPPING_SCHEMA_MISMATCH",
                    "CSV columns or row count differ from the input artifact metadata.",
                )
        except SafeCsvError as exc:
            return StageResult(
                context=context,
                stage=PipelineStage.COLUMN_MAPPING,
                status=ExecutionStatus.FAILED,
                input_artifact=artifact,
                report=None,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
                error=ExecutionError(code=exc.code, message=str(exc), retryable=False),
            )

        try:
            df, conversion_issues = self._apply_unit_conversions(
                df=df,
                contract=contract,
                mapping=mapping,
            )
        except UnitConversionError as exc:
            return StageResult(
                context=context,
                stage=PipelineStage.COLUMN_MAPPING,
                status=ExecutionStatus.FAILED,
                input_artifact=artifact,
                report=None,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
                error=ExecutionError(
                    code="UNIT_CONVERSION_FAILED",
                    message=str(exc),
                    retryable=False,
                ),
            )

        rename_map = {entry.source_column: entry.canonical_column for entry in entries}
        canonical_order = [
            feature.name
            for feature in contract.all_columns
            if feature.name in set(rename_map.values())
        ]
        mapped_df = df[list(rename_map.keys())].rename(columns=rename_map)
        mapped_df = mapped_df[canonical_order]

        mapped_bytes = mapped_df.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
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

        config_sha256 = csv_configuration_sha256(
            self._resource_policy,
            extra=mapping.model_dump(mode="json"),
        )

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
            description=(
                "Applied approved canonical column mapping with lexical value preservation."
            ),
        )
        output_artifact = DatasetArtifact(
            metadata=mapped_metadata,
            reference=mapped_ref,
        )

        issues: list[ValidationIssue] = list(conversion_issues)
        if mapping.proposal.unmapped_columns:
            issues.append(
                ValidationIssue(
                    code="UNMAPPED_SOURCE_COLUMNS_DROPPED",
                    stage=PipelineStage.COLUMN_MAPPING,
                    severity=ValidationSeverity.INFO,
                    impact=IssueImpact.NONE,
                    message=("Unmapped source columns were excluded from the mapped revision."),
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
