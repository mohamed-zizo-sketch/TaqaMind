"""CSV data ingestion engine implementing the DataIngestion protocol."""

import csv
from datetime import UTC, datetime
from io import BytesIO, StringIO

import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import PipelineStage
from taqamind.core.contracts.validation import ValidationReport
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
    DataIngestion,
    validate_processing_input,
)
from taqamind.core.interfaces.results import (
    ExecutionError,
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage


class CsvDataIngestion(DataIngestion):
    """Ingest a verified RAW CSV artifact into an INGESTED revision."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        implementation_version: str = "0.1.0",
        max_rows: int | None = None,
    ) -> None:
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows must be positive when provided.")
        self._storage = storage
        self._implementation_version = implementation_version
        self._max_rows = max_rows

    def _failed_result(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        started_at: datetime,
        code: str,
        message: str,
    ) -> StageResult:
        return StageResult(
            context=context,
            stage=PipelineStage.INGESTION,
            status=ExecutionStatus.FAILED,
            input_artifact=artifact,
            report=None,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
            error=ExecutionError(code=code, message=message),
        )

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

        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            raw_bytes = stream.read()

        if not raw_bytes.strip():
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="CSV_EMPTY_FILE",
                message="Uploaded CSV file is empty.",
            )

        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="CSV_DECODE_ERROR",
                message="CSV file must be valid UTF-8.",
            )

        reader = csv.reader(StringIO(text))
        try:
            headers = next(reader)
        except StopIteration:
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="CSV_MISSING_HEADER",
                message="CSV file does not contain a header row.",
            )

        cleaned_headers = [header.strip() for header in headers]
        if not cleaned_headers or any(not header for header in cleaned_headers):
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="CSV_BLANK_HEADER",
                message="CSV header contains blank column names.",
            )

        if len(cleaned_headers) != len(set(cleaned_headers)):
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="CSV_DUPLICATE_HEADER",
                message="CSV header contains duplicate column names.",
            )

        try:
            df = pd.read_csv(StringIO(text), keep_default_na=True)
        except Exception as exc:
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="CSV_PARSE_ERROR",
                message=f"Failed to parse CSV content: {exc}",
            )

        df.columns = cleaned_headers

        if self._max_rows is not None and len(df) > self._max_rows:
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="CSV_ROW_LIMIT_EXCEEDED",
                message=f"CSV row count {len(df)} exceeds limit {self._max_rows}.",
            )

        ingested_bytes = df.to_csv(index=False).encode("utf-8")
        ingested_ref = ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=calculate_bytes_sha256(ingested_bytes),
            size_bytes=len(ingested_bytes),
        )
        self._storage.put(
            factory_id=context.factory_id,
            reference=ingested_ref,
            stream=BytesIO(ingested_bytes),
        )

        columns = tuple(
            DatasetColumnSchema(name=str(col), physical_dtype=str(df[col].dtype))
            for col in df.columns
        )
        ingested_metadata = artifact.metadata.derive(
            context=context,
            state=DatasetState.INGESTED,
            row_count=len(df),
            columns=columns,
            operation="ingest_csv",
            implementation_version=self._implementation_version,
            description="Ingested raw CSV into verified immutable revision.",
        )
        output_artifact = DatasetArtifact(
            metadata=ingested_metadata,
            reference=ingested_ref,
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
