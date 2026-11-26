"""CSV data ingestion engine implementing the DataIngestion protocol."""

import json
from datetime import UTC, datetime
from io import BytesIO

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import PipelineStage
from taqamind.core.contracts.validation import ValidationReport
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
        implementation_version: str = "0.2.0",
        max_rows: int | None = None,
        delimiter: str = ",",
    ) -> None:
        self._storage = storage
        self._implementation_version = implementation_version
        self._reader = SafeCsvReader(delimiter=delimiter, max_rows=max_rows)
        configuration = {
            "delimiter": delimiter,
            "max_rows": max_rows,
            "reader_version": self._reader.VERSION,
        }
        self._configuration_sha256 = calculate_bytes_sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )

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

        try:
            df = self._reader.read(raw_bytes)
        except SafeCsvError as exc:
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code=exc.code,
                message=str(exc),
            )

        ingested_bytes = df.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
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
            configuration_sha256=self._configuration_sha256,
            description="Ingested raw CSV without implicit value or type coercion.",
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
