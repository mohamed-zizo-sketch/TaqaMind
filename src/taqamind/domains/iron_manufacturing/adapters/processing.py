"""Protocol-compatible data processing adapters (Stages 1-6).

Implements Shared Core protocols:
    - `DataIngestion` -> `LocalCsvDataIngestion`
    - `DataProfiler` -> `LocalDataProfiler`
    - `ColumnMapper` -> `LocalColumnMapper`
    - `SemanticValidator` -> `LocalSemanticValidator`
    - `DataCleaner` -> `LocalDataCleaner`
    - `FeatureEngineer` -> `LocalFeatureEngineer`

Migration note:
    Generic CSV ingestion, profiling, alias/unit mapping, contract-driven
    semantic validation, and deterministic cleaning are separated from process-
    specific industrial rules (injected via callables) so these adapters can be
    migrated cleanly into `taqamind.core.{ingestion,profiling,mapping,validation,
    preprocessing,feature_engineering}`.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    DataType,
    FeatureOrigin,
    FeatureRole,
    IssueImpact,
    MissingValuePolicy,
    PipelineStage,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
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
from taqamind.core.interfaces.dataset_registry import DatasetRegistry
from taqamind.core.interfaces.processing import (
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
from taqamind.core.interfaces.results import (
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.domains.iron_manufacturing.common.hashing import calculate_config_sha256
from taqamind.domains.iron_manufacturing.common.normalization import (
    normalize_header,
    safe_read_csv_bytes,
)
from taqamind.domains.iron_manufacturing.common.units import (
    UnitConversionError,
    convert_series_unit,
    normalize_unit_name,
)
from taqamind.domains.iron_manufacturing.config import (
    AMBIGUOUS_RAW_HEADERS,
    IMPLEMENTATION_VERSION,
)


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _schema_from_df(df: pd.DataFrame) -> tuple[DatasetColumnSchema, ...]:
    return tuple(
        DatasetColumnSchema(name=str(col), physical_dtype=str(df[col].dtype))
        for col in df.columns
    )


def _read_artifact_df(
    storage: DatasetArtifactStorage,
    factory_id,
    artifact: DatasetArtifact,
) -> pd.DataFrame:
    with storage.open_verified(factory_id=factory_id, reference=artifact.reference) as stream:
        raw_bytes = stream.read()
    df, _, _, _ = safe_read_csv_bytes(raw_bytes)
    return df


def _store_derived_artifact(
    *,
    storage: DatasetArtifactStorage,
    registry: DatasetRegistry,
    context: PipelineContext,
    parent: DatasetArtifact,
    df: pd.DataFrame,
    next_state: DatasetState,
    operation: str,
    description: str,
    config_payload: dict | None = None,
) -> DatasetArtifact:
    csv_bytes = _df_to_csv_bytes(df)
    content_sha = calculate_bytes_sha256(csv_bytes)
    reference = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=content_sha,
        size_bytes=len(csv_bytes),
    )
    storage.put(
        factory_id=context.factory_id,
        reference=reference,
        stream=io.BytesIO(csv_bytes),
    )
    new_meta = parent.metadata.derive(
        context=context,
        state=next_state,
        row_count=int(len(df)),
        columns=_schema_from_df(df),
        operation=operation,
        implementation_version=IMPLEMENTATION_VERSION,
        description=description,
        configuration_sha256=(
            calculate_config_sha256(config_payload) if config_payload is not None else None
        ),
    )
    new_artifact = DatasetArtifact(metadata=new_meta, reference=reference)
    registry.register(factory_id=context.factory_id, artifact=new_artifact)
    return new_artifact


class LocalCsvDataIngestion:
    """Protocol-compatible CSV ingestion adapter (`DataIngestion`)."""

    def __init__(
        self,
        *,
        storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
    ) -> None:
        self._storage = storage
        self._dataset_registry = dataset_registry

    def ingest(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        started_at = datetime.now(UTC)
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.RAW,
        )

        issues: list[ValidationIssue] = []
        try:
            with self._storage.open_verified(
                factory_id=context.factory_id,
                reference=artifact.reference,
            ) as stream:
                raw_bytes = stream.read()
            df, encoding, delimiter, _ = safe_read_csv_bytes(raw_bytes)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    code="MALFORMED_CSV_INPUT",
                    stage=PipelineStage.INGESTION,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=str(exc),
                    recommendation="Provide a valid UTF-8/CP1256 CSV with unique headers.",
                )
            )
            report = ValidationReport(
                dataset_id=context.dataset_id,
                contract_version=context.contract_version,
                dataset_revision_id=artifact.metadata.revision_id,
                domain=context.domain,
                process=context.process,
                purpose=context.purpose,
                completed_stages=(PipelineStage.INGESTION,),
                issues=tuple(issues),
            )
            return StageResult(
                context=context,
                stage=PipelineStage.INGESTION,
                status=ExecutionStatus.BLOCKED,
                input_artifact=artifact,
                report=report,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
            )

        if len(df) == 0:
            issues.append(
                ValidationIssue(
                    code="EMPTY_DATASET",
                    stage=PipelineStage.INGESTION,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message="Uploaded CSV contains zero data rows.",
                    recommendation="Upload a CSV containing at least one production record.",
                )
            )
            report = ValidationReport(
                dataset_id=context.dataset_id,
                contract_version=context.contract_version,
                dataset_revision_id=artifact.metadata.revision_id,
                domain=context.domain,
                process=context.process,
                purpose=context.purpose,
                completed_stages=(PipelineStage.INGESTION,),
                issues=tuple(issues),
            )
            return StageResult(
                context=context,
                stage=PipelineStage.INGESTION,
                status=ExecutionStatus.BLOCKED,
                input_artifact=artifact,
                report=report,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
            )

        output_artifact = _store_derived_artifact(
            storage=self._storage,
            registry=self._dataset_registry,
            context=context,
            parent=artifact,
            df=df,
            next_state=DatasetState.INGESTED,
            operation="csv_ingestion",
            description=f"Parsed raw CSV (encoding={encoding}, delimiter='{delimiter}').",
            config_payload={"encoding": encoding, "delimiter": delimiter},
        )
        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.INGESTION,),
            issues=(),
        )
        return StageResult(
            context=context,
            stage=PipelineStage.INGESTION,
            status=ExecutionStatus.SUCCEEDED,
            input_artifact=artifact,
            output_artifact=output_artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )


class LocalDataProfiler:
    """Protocol-compatible dataset profiler adapter (`DataProfiler`)."""

    def __init__(self, *, storage: DatasetArtifactStorage) -> None:
        self._storage = storage

    def profile(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> ProfilingResult:
        started_at = datetime.now(UTC)
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.INGESTED,
        )

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        col_profiles: list[ColumnProfile] = []
        for col_schema in artifact.metadata.columns:
            series = df[col_schema.name]
            missing_count = int(series.isna().sum())
            unique_count = int(series.dropna().nunique())
            num_min: float | None = None
            num_max: float | None = None
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
                valid = finite.dropna()
                if len(valid) > 0:
                    num_min = float(valid.min())
                    num_max = float(valid.max())
            col_profiles.append(
                ColumnProfile(
                    name=col_schema.name,
                    physical_dtype=col_schema.physical_dtype,
                    missing_count=missing_count,
                    unique_count=unique_count,
                    numeric_minimum=num_min,
                    numeric_maximum=num_max,
                )
            )

        exact_dups = int(df.duplicated().sum()) if len(df) > 1 else 0
        profile = DatasetProfile(
            dataset_revision_id=artifact.metadata.revision_id,
            factory_id=context.factory_id,
            row_count=int(len(df)),
            columns=tuple(col_profiles),
            exact_duplicate_row_count=exact_dups,
        )
        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.PROFILING,),
            issues=(),
        )
        stage_result = StageResult(
            context=context,
            stage=PipelineStage.PROFILING,
            status=ExecutionStatus.SUCCEEDED,
            input_artifact=artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )
        return ProfilingResult(result=stage_result, profile=profile)


class LocalColumnMapper:
    """Protocol-compatible deterministic column mapper (`ColumnMapper`).

    Applies exact name matching first, then verified contract aliases.
    Never infers meaning for bare ambiguous headers (e.g., `Weight`, `Waste`).
    """

    def __init__(
        self,
        *,
        storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        explicit_column_overrides: dict[str, str] | None = None,
        explicit_unit_overrides: dict[str, str] | None = None,
    ) -> None:
        self._storage = storage
        self._dataset_registry = dataset_registry
        self._column_overrides = dict(explicit_column_overrides or {})
        self._unit_overrides = dict(explicit_unit_overrides or {})

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

        canonical_map = contract.column_map
        alias_lookup: dict[str, str] = {}
        for col in contract.all_columns:
            if col.origin == FeatureOrigin.DERIVED:
                continue
            for alias in col.aliases:
                alias_lookup[normalize_header(alias)] = col.name

        entries: list[ColumnMappingEntry] = []
        unmapped: list[str] = []
        used_canonical: set[str] = set()

        for col_schema in artifact.metadata.columns:
            src = col_schema.name
            if src in self._column_overrides:
                canonical = self._column_overrides[src]
                if canonical not in canonical_map or canonical in used_canonical:
                    unmapped.append(src)
                    continue
                feature = canonical_map[canonical]
                src_unit = self._unit_overrides.get(src, feature.unit)
                entries.append(
                    ColumnMappingEntry(
                        source_column=src,
                        canonical_column=canonical,
                        method=MappingMethod.MANUAL,
                        source_unit=src_unit,
                        canonical_unit=feature.unit,
                        requires_confirmation=True,
                        reason="Explicit user column override requires approval.",
                    )
                )
                used_canonical.add(canonical)
                continue

            if src in canonical_map and canonical_map[src].origin != FeatureOrigin.DERIVED:
                if src in used_canonical:
                    unmapped.append(src)
                    continue
                feature = canonical_map[src]
                src_unit = self._unit_overrides.get(src, feature.unit)
                unit_differs = (
                    feature.unit is not None
                    and src_unit is not None
                    and normalize_unit_name(src_unit) != normalize_unit_name(feature.unit)
                )
                entries.append(
                    ColumnMappingEntry(
                        source_column=src,
                        canonical_column=src,
                        method=MappingMethod.EXACT,
                        source_unit=src_unit,
                        canonical_unit=feature.unit,
                        requires_confirmation=unit_differs,
                        reason=(
                            f"Unit conversion requested ({src_unit} -> {feature.unit})."
                            if unit_differs
                            else None
                        ),
                    )
                )
                used_canonical.add(src)
                continue

            norm = normalize_header(src)
            if norm in AMBIGUOUS_RAW_HEADERS:
                unmapped.append(src)
                continue

            matched = alias_lookup.get(norm)
            if matched is not None and matched not in used_canonical:
                feature = canonical_map[matched]
                src_unit = self._unit_overrides.get(src, feature.unit)
                entries.append(
                    ColumnMappingEntry(
                        source_column=src,
                        canonical_column=matched,
                        method=MappingMethod.ALIAS,
                        source_unit=src_unit,
                        canonical_unit=feature.unit,
                        requires_confirmation=True,
                        reason=f"Matched verified alias '{src}' -> '{matched}'.",
                    )
                )
                used_canonical.add(matched)
            else:
                unmapped.append(src)

        return MappingProposal(
            factory_id=context.factory_id,
            execution_id=context.execution_id,
            dataset_revision_id=artifact.metadata.revision_id,
            entries=tuple(entries),
            unmapped_columns=tuple(unmapped),
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

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        issues: list[ValidationIssue] = []

        rename_map: dict[str, str] = {}
        converted_units: dict[str, str] = {}
        keep_cols: list[str] = []

        for entry in mapping.proposal.entries:
            rename_map[entry.source_column] = entry.canonical_column
            keep_cols.append(entry.canonical_column)

        # Keep mapped canonical columns only (drop unmapped extras cleanly after auditing).
        mapped_df = df[[e.source_column for e in mapping.proposal.entries]].rename(
            columns=rename_map
        )

        for entry in mapping.proposal.entries:
            if entry.canonical_unit is not None and entry.source_unit is not None:
                src_norm = normalize_unit_name(entry.source_unit)
                dst_norm = normalize_unit_name(entry.canonical_unit)
                if src_norm != dst_norm:
                    try:
                        mapped_df[entry.canonical_column] = convert_series_unit(
                            mapped_df[entry.canonical_column],
                            entry.source_unit,
                            entry.canonical_unit,
                        )
                        converted_units[entry.canonical_column] = (
                            f"{entry.source_unit} -> {entry.canonical_unit}"
                        )
                    except UnitConversionError as exc:
                        issues.append(
                            ValidationIssue(
                                code="INVALID_UNIT_CONVERSION",
                                stage=PipelineStage.COLUMN_MAPPING,
                                severity=ValidationSeverity.ERROR,
                                impact=IssueImpact.BLOCKING,
                                message=str(exc),
                                columns=(entry.canonical_column,),
                                recommendation="Provide a dimensionally compatible unit.",
                            )
                        )

        if issues:
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
                status=ExecutionStatus.BLOCKED,
                input_artifact=artifact,
                report=report,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
            )

        output_artifact = _store_derived_artifact(
            storage=self._storage,
            registry=self._dataset_registry,
            context=context,
            parent=artifact,
            df=mapped_df[keep_cols],
            next_state=DatasetState.MAPPED,
            operation="canonical_column_mapping",
            description=(
                f"Applied approved mapping ({len(keep_cols)} columns, "
                f"{len(converted_units)} unit conversions)."
            ),
            config_payload={
                "approved_by": mapping.approved_by,
                "renamed": rename_map,
                "unit_conversions": converted_units,
            },
        )
        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.COLUMN_MAPPING,),
            issues=(),
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


class LocalSemanticValidator:
    """Contract-driven semantic validator implementing `SemanticValidator`."""

    def __init__(
        self,
        *,
        storage: DatasetArtifactStorage,
        process_validators: (
            dict[str, Callable[[pd.DataFrame], list[ValidationIssue]]] | None
        ) = None,
        forbidden_predictors_by_process: dict[str, frozenset[str]] | None = None,
    ) -> None:
        self._storage = storage
        self._process_validators = dict(process_validators or {})
        self._forbidden = dict(forbidden_predictors_by_process or {})

    def validate(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        started_at = datetime.now(UTC)
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.MAPPED,
        )

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        issues: list[ValidationIssue] = []

        # 1. Check required raw columns for purpose
        if context.purpose != DatasetPurpose.RESEARCH:
            required_cols = set(contract.required_columns(context.purpose))
            missing_req = sorted(required_cols - set(df.columns))
            if missing_req:
                issues.append(
                    ValidationIssue(
                        code="MISSING_REQUIRED_COLUMNS",
                        stage=PipelineStage.SEMANTIC_VALIDATION,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Required columns missing for {context.purpose.value}: "
                            f"{missing_req}."
                        ),
                        columns=tuple(missing_req),
                        recommendation=(
                            "Include all required columns defined in the DatasetContract."
                        ),
                    )
                )

        # 2. Reject target in prediction datasets and post-production/forbidden predictors
        target_name = contract.target.feature.name
        if context.purpose == DatasetPurpose.PREDICTION and target_name in df.columns:
            issues.append(
                ValidationIssue(
                    code="TARGET_IN_PREDICTION_INPUT",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=f"Prediction input must not contain target column '{target_name}'.",
                    columns=(target_name,),
                    recommendation="Remove target column before submitting prediction CSV.",
                )
            )

        forbidden_set = self._forbidden.get(contract.process, frozenset())
        present_forbidden = sorted(set(df.columns) & forbidden_set)
        if present_forbidden:
            issues.append(
                ValidationIssue(
                    code="POST_PRODUCTION_PREDICTOR_LEAKAGE",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        "Post-production or in-process columns detected in pre-production input: "
                        f"{present_forbidden}."
                    ),
                    columns=tuple(present_forbidden),
                    recommendation=(
                        "Exclude in-process and post-production columns from predictors."
                    ),
                )
            )

        # 3. Column-level data type, missing policy, finite numeric, and constraint checks
        for col_name in df.columns:
            if col_name not in contract.column_map:
                continue
            feature = contract.column_map[col_name]
            series = df[col_name]

            if feature.missing_policy == MissingValuePolicy.REJECT:
                null_mask = series.isna() | (series.astype(str).str.strip() == "")
                if bool(null_mask.any()):
                    rows = tuple(int(i) for i in np.flatnonzero(null_mask)[:20])
                    code = (
                        "MISSING_TARGET_VALUES"
                        if feature.role == FeatureRole.TARGET
                        else "MISSING_REQUIRED_VALUES"
                    )
                    issues.append(
                        ValidationIssue(
                            code=code,
                            stage=PipelineStage.SEMANTIC_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                f"Column '{col_name}' violates non-null policy "
                                f"({int(null_mask.sum())} missing rows)."
                            ),
                            columns=(col_name,),
                            affected_rows=int(null_mask.sum()),
                            row_samples=rows,
                            recommendation=(
                                "Target values cannot be imputed; supply verified measurements."
                                if feature.role == FeatureRole.TARGET
                                else "Fill or correct missing values for this required column."
                            ),
                        )
                    )

            if feature.data_type in {DataType.FLOAT, DataType.INTEGER}:
                non_null = series.dropna()
                coerced = pd.to_numeric(non_null, errors="coerce")
                bad_parse = coerced.isna()
                if bool(bad_parse.any()):
                    bad_idx = non_null.index[bad_parse]
                    issues.append(
                        ValidationIssue(
                            code="INVALID_NUMERIC_TYPE",
                            stage=PipelineStage.SEMANTIC_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=f"Column '{col_name}' contains non-numeric values.",
                            columns=(col_name,),
                            affected_rows=int(bad_parse.sum()),
                            row_samples=tuple(int(i) for i in bad_idx[:20]),
                        )
                    )
                else:
                    non_finite = ~np.isfinite(coerced.to_numpy(dtype=float))
                    if bool(non_finite.any()):
                        bad_idx = non_null.index[non_finite]
                        issues.append(
                            ValidationIssue(
                                code="NON_FINITE_NUMERIC_VALUES",
                                stage=PipelineStage.SEMANTIC_VALIDATION,
                                severity=ValidationSeverity.ERROR,
                                impact=IssueImpact.BLOCKING,
                                message=f"Column '{col_name}' contains NaN/Inf values.",
                                columns=(col_name,),
                                affected_rows=int(non_finite.sum()),
                                row_samples=tuple(int(i) for i in bad_idx[:20]),
                            )
                        )
                    else:
                        c = feature.constraints
                        if c.minimum is not None:
                            viol = (
                                coerced < c.minimum
                                if c.minimum_inclusive
                                else coerced <= c.minimum
                            )
                            if bool(viol.any()):
                                bad_idx = non_null.index[viol]
                                issues.append(
                                    ValidationIssue(
                                        code="CONSTRAINT_MINIMUM_VIOLATION",
                                        stage=PipelineStage.SEMANTIC_VALIDATION,
                                        severity=ValidationSeverity.ERROR,
                                        impact=IssueImpact.BLOCKING,
                                        message=(
                                            f"Column '{col_name}' has {int(viol.sum())} value(s) "
                                            f"below minimum {c.minimum}."
                                        ),
                                        columns=(col_name,),
                                        affected_rows=int(viol.sum()),
                                        row_samples=tuple(int(i) for i in bad_idx[:20]),
                                    )
                                )
                        if c.maximum is not None:
                            viol = (
                                coerced > c.maximum
                                if c.maximum_inclusive
                                else coerced >= c.maximum
                            )
                            if bool(viol.any()):
                                bad_idx = non_null.index[viol]
                                issues.append(
                                    ValidationIssue(
                                        code="CONSTRAINT_MAXIMUM_VIOLATION",
                                        stage=PipelineStage.SEMANTIC_VALIDATION,
                                        severity=ValidationSeverity.ERROR,
                                        impact=IssueImpact.BLOCKING,
                                        message=(
                                            f"Column '{col_name}' has {int(viol.sum())} value(s) "
                                            f"above maximum {c.maximum}."
                                        ),
                                        columns=(col_name,),
                                        affected_rows=int(viol.sum()),
                                        row_samples=tuple(int(i) for i in bad_idx[:20]),
                                    )
                                )

            elif feature.data_type in {DataType.DATE, DataType.DATETIME}:
                non_null = series.dropna()
                parsed = pd.to_datetime(non_null, errors="coerce")
                bad_ts = parsed.isna()
                if bool(bad_ts.any()):
                    bad_idx = non_null.index[bad_ts]
                    issues.append(
                        ValidationIssue(
                            code="INVALID_TIMESTAMP_VALUES",
                            stage=PipelineStage.SEMANTIC_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=f"Timestamp column '{col_name}' has unparseable dates.",
                            columns=(col_name,),
                            affected_rows=int(bad_ts.sum()),
                            row_samples=tuple(int(i) for i in bad_idx[:20]),
                        )
                    )

            elif feature.constraints.allowed_values is not None:
                non_null = series.dropna().astype(str).str.strip()
                allowed = set(feature.constraints.allowed_values)
                bad_cat = ~non_null.isin(allowed)
                if bool(bad_cat.any()):
                    bad_idx = non_null.index[bad_cat]
                    issues.append(
                        ValidationIssue(
                            code="INVALID_CATEGORY_VALUE",
                            stage=PipelineStage.SEMANTIC_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                f"Column '{col_name}' contains values outside allowed set "
                                f"{sorted(allowed)}."
                            ),
                            columns=(col_name,),
                            affected_rows=int(bad_cat.sum()),
                            row_samples=tuple(int(i) for i in bad_idx[:20]),
                        )
                    )

        # 4. Duplicate business keys inside the same upload
        if contract.business_keys and all(k in df.columns for k in contract.business_keys):
            dup_mask = df.duplicated(subset=list(contract.business_keys), keep=False)
            if bool(dup_mask.any()):
                bad_rows = tuple(int(i) for i in np.flatnonzero(dup_mask)[:20])
                issues.append(
                    ValidationIssue(
                        code="DUPLICATE_BUSINESS_KEYS_IN_BATCH",
                        stage=PipelineStage.SEMANTIC_VALIDATION,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Duplicate business key(s) detected within upload for "
                            f"{contract.business_keys}; resolve conflicts explicitly."
                        ),
                        columns=contract.business_keys,
                        affected_rows=int(dup_mask.sum()),
                        row_samples=bad_rows,
                        recommendation="Deduplicate or disambiguate conflicting business keys.",
                    )
                )

        # 5. Domain/process-specific physical validation rules
        proc_validator = self._process_validators.get(contract.process)
        if proc_validator is not None:
            issues.extend(proc_validator(df))

        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.SEMANTIC_VALIDATION,),
            issues=tuple(issues),
        )
        status = (
            ExecutionStatus.BLOCKED if report.has_blocking_issues else ExecutionStatus.SUCCEEDED
        )
        return StageResult(
            context=context,
            stage=PipelineStage.SEMANTIC_VALIDATION,
            status=status,
            input_artifact=artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )


class LocalDataCleaner:
    """Deterministic pre-split data cleaner implementing `DataCleaner`.

    Rules enforced:
        - Preserves valid zero-scrap records.
        - Never imputes target values or replaces missing target with zero.
        - Never fits statistical imputers, encoders, or scalers before splitting.
        - Materializes optional declared raw columns as NaN when absent.
    """

    def __init__(
        self,
        *,
        storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
    ) -> None:
        self._storage = storage
        self._dataset_registry = dataset_registry

    def clean(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        started_at = datetime.now(UTC)
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.MAPPED,
        )

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        cleaned = df.copy()

        # Trim string/category columns and standardize ISO timestamps deterministically
        for col in contract.all_columns:
            if col.origin == FeatureOrigin.DERIVED:
                continue
            if col.name not in cleaned.columns:
                if (
                    col.role == FeatureRole.PREDICTOR
                    and not col.required
                    and col.nullable
                ):
                    cleaned[col.name] = np.nan
                continue

            if col.data_type in {DataType.STRING, DataType.CATEGORY}:
                cleaned[col.name] = cleaned[col.name].map(
                    lambda v: v.strip() if isinstance(v, str) else v
                )
            elif col.data_type in {DataType.FLOAT, DataType.INTEGER}:
                cleaned[col.name] = pd.to_numeric(cleaned[col.name], errors="coerce")
            elif col.data_type in {DataType.DATE, DataType.DATETIME}:
                parsed = pd.to_datetime(cleaned[col.name], errors="coerce")
                cleaned[col.name] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")

        output_artifact = _store_derived_artifact(
            storage=self._storage,
            registry=self._dataset_registry,
            context=context,
            parent=artifact,
            df=cleaned,
            next_state=DatasetState.CLEANED,
            operation="deterministic_cleaning",
            description=(
                "Applied deterministic whitespace normalization, canonical dtype casting, "
                "and optional predictor schema materialization without fitting statistics."
            ),
        )
        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.CLEANING,),
            issues=(),
        )
        return StageResult(
            context=context,
            stage=PipelineStage.CLEANING,
            status=ExecutionStatus.SUCCEEDED,
            input_artifact=artifact,
            output_artifact=output_artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )


class LocalFeatureEngineer:
    """Deterministic pre-split feature engineer implementing `FeatureEngineer`."""

    def __init__(
        self,
        *,
        storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        process_engineers: dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
    ) -> None:
        self._storage = storage
        self._dataset_registry = dataset_registry
        self._process_engineers = dict(process_engineers)

    def engineer(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        started_at = datetime.now(UTC)
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.CLEANED,
        )

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        engineer_fn = self._process_engineers.get(contract.process)
        engineered = engineer_fn(df) if engineer_fn is not None else df.copy()

        # Ensure every declared predictor column exists in deterministic contract order
        for predictor_name in contract.predictor_names:
            if predictor_name not in engineered.columns:
                engineered[predictor_name] = np.nan

        if context.purpose == DatasetPurpose.PREDICTION:
            target_name = contract.target.feature.name
            if target_name in engineered.columns:
                engineered = engineered.drop(columns=[target_name])

        output_artifact = _store_derived_artifact(
            storage=self._storage,
            registry=self._dataset_registry,
            context=context,
            parent=artifact,
            df=engineered,
            next_state=DatasetState.ENGINEERED,
            operation="deterministic_feature_engineering",
            description=(
                f"Computed deterministic pre-production features for process '{contract.process}'."
            ),
            config_payload={"process": contract.process},
        )
        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.FEATURE_ENGINEERING,),
            issues=(),
        )
        return StageResult(
            context=context,
            stage=PipelineStage.FEATURE_ENGINEERING,
            status=ExecutionStatus.SUCCEEDED,
            input_artifact=artifact,
            output_artifact=output_artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )
