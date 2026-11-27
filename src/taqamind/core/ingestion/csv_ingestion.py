"""CSV data ingestion engine implementing the DataIngestion protocol."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from io import BytesIO

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import PipelineStage
from taqamind.core.contracts.validation import ValidationReport
from taqamind.core.ingestion.resource_policy import (
    DEFAULT_RAW_CSV_POLICY,
    CsvResourcePolicy,
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
    DataIngestion,
    validate_processing_input,
)
from taqamind.core.interfaces.results import (
    ExecutionError,
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import (
    ArtifactSizeLimitError,
    DatasetArtifactStorage,
    validate_artifact_scope,
)


class CsvDataIngestion(DataIngestion):
    """Ingest a verified RAW CSV artifact into an INGESTED revision."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        implementation_version: str = "0.3.0",
        max_rows: int | None = None,
        delimiter: str = ",",
        resource_policy: CsvResourcePolicy | None = None,
    ) -> None:
        self._storage = storage
        self._implementation_version = implementation_version

        if resource_policy is not None and max_rows is not None:
            raise ValueError("Use resource_policy or max_rows, not both.")

        policy = resource_policy if resource_policy is not None else DEFAULT_RAW_CSV_POLICY

        if max_rows is not None:
            policy = replace(policy, max_rows=max_rows)

        self._resource_policy = policy

        self._reader = SafeCsvReader(
            delimiter=delimiter,
            resource_policy=policy,
        )

        configuration = {
            "delimiter": delimiter,
            "resource_policy": asdict(policy),
            "reader_version": self._reader.VERSION,
        }

        self._configuration_sha256 = calculate_bytes_sha256(
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
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

        validate_artifact_scope(
            factory_id=context.factory_id,
            reference=artifact.reference,
        )

        if artifact.reference.size_bytes > self._resource_policy.max_bytes:
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="CSV_BYTE_LIMIT_EXCEEDED",
                message="CSV byte limit exceeded.",
            )

        try:
            with self._storage.open_verified(
                factory_id=context.factory_id,
                reference=artifact.reference,
            ) as stream:
                df = self._reader.read_stream(stream)

        except SafeCsvError as exc:
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code=exc.code,
                message=str(exc),
            )

        except ArtifactSizeLimitError:
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="ARTIFACT_SIZE_LIMIT_EXCEEDED",
                message="Artifact exceeds the configured storage size limit.",
            )

        ingested_bytes = df.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
        ingested_ref = ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=calculate_bytes_sha256(ingested_bytes),
            size_bytes=len(ingested_bytes),
        )
        try:
            self._storage.put(
                factory_id=context.factory_id,
                reference=ingested_ref,
                stream=BytesIO(ingested_bytes),
            )
        except ArtifactSizeLimitError:
            return self._failed_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                code="ARTIFACT_SIZE_LIMIT_EXCEEDED",
                message="Artifact exceeds the configured storage size limit.",
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
