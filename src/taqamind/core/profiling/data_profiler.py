"""Dataset profiling engine implementing the DataProfiler protocol."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import DataType, FeatureRole, PipelineStage
from taqamind.core.contracts.validation import ValidationReport
from taqamind.core.ingestion.safe_csv import SafeCsvError, SafeCsvReader
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetState
from taqamind.core.interfaces.processing import (
    DataProfiler,
    validate_processing_input,
)
from taqamind.core.interfaces.processing_models import (
    ColumnProfile,
    DatasetProfile,
    ProfilingResult,
)
from taqamind.core.interfaces.results import (
    ExecutionError,
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage


class PandasDataProfiler(DataProfiler):
    """Profile an INGESTED CSV artifact without mutating it."""

    def __init__(self, storage: DatasetArtifactStorage) -> None:
        self._storage = storage
        self._reader = SafeCsvReader()

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

        try:
            with self._storage.open_verified(
                factory_id=context.factory_id,
                reference=artifact.reference,
            ) as stream:
                df = self._reader.read(stream.read())
        except SafeCsvError as exc:
            return self._failed_result(context, artifact, started_at, exc.code, str(exc))

        observed_schema = tuple((str(col), str(df[col].dtype)) for col in df.columns)
        expected_schema = tuple(
            (col.name, col.physical_dtype) for col in artifact.metadata.columns
        )
        if observed_schema != expected_schema or len(df) != artifact.metadata.row_count:
            return self._failed_result(
                context,
                artifact,
                started_at,
                "PROFILE_SCHEMA_MISMATCH",
                "Artifact schema or row count differs from safe CSV parsing. "
                "Re-ingest the original raw CSV with the current ingestion engine.",
            )

        column_profiles: list[ColumnProfile] = []
        for col_name in df.columns:
            series = df[col_name]
            missing_count = int(series.isna().sum())
            unique_count = int(series.nunique(dropna=True))

            numeric_minimum: float | None = None
            numeric_maximum: float | None = None
            # Numeric summaries are diagnostic only; never change the lexical data.
            # Before mapping, use only exact declared numeric column names.
            feature = contract.column_map.get(str(col_name))
            if (
                feature is not None
                and feature.data_type in {DataType.INTEGER, DataType.FLOAT}
                and feature.role != FeatureRole.IDENTIFIER
                and col_name not in contract.business_keys
            ):
                numeric = pd.to_numeric(series, errors="coerce").to_numpy(
                    dtype=float, na_value=np.nan
                )
                finite_values = numeric[np.isfinite(numeric)]
                if finite_values.size:
                    numeric_minimum = float(finite_values.min())
                    numeric_maximum = float(finite_values.max())

            column_profiles.append(
                ColumnProfile(
                    name=str(col_name),
                    physical_dtype=str(series.dtype),
                    missing_count=missing_count,
                    unique_count=unique_count,
                    numeric_minimum=numeric_minimum,
                    numeric_maximum=numeric_maximum,
                )
            )

        dataset_profile = DatasetProfile(
            dataset_revision_id=artifact.metadata.revision_id,
            factory_id=artifact.metadata.factory_id,
            row_count=len(df),
            columns=tuple(column_profiles),
            exact_duplicate_row_count=int(df.duplicated().sum()),
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
            output_artifact=None,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )
        return ProfilingResult(result=stage_result, profile=dataset_profile)

    @staticmethod
    def _failed_result(
        context: PipelineContext,
        artifact: DatasetArtifact,
        started_at: datetime,
        code: str,
        message: str,
    ) -> ProfilingResult:
        return ProfilingResult(
            result=StageResult(
                context=context,
                stage=PipelineStage.PROFILING,
                status=ExecutionStatus.FAILED,
                input_artifact=artifact,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
                error=ExecutionError(code=code, message=message, retryable=False),
            ),
            profile=None,
        )
