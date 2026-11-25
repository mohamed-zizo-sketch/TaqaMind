"""Dataset profiling engine implementing the DataProfiler protocol."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import PipelineStage
from taqamind.core.contracts.validation import ValidationReport
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
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage


class PandasDataProfiler(DataProfiler):
    """Profile an INGESTED CSV artifact without mutating it."""

    def __init__(self, storage: DatasetArtifactStorage) -> None:
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

        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True)

        column_profiles: list[ColumnProfile] = []
        for col_name in df.columns:
            series = df[col_name]
            missing_count = int(series.isna().sum())
            unique_count = int(series.nunique(dropna=True))

            numeric_minimum: float | None = None
            numeric_maximum: float | None = None
            if pd.api.types.is_numeric_dtype(series):
                finite_values = series[np.isfinite(series)]
                if not finite_values.empty:
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
    