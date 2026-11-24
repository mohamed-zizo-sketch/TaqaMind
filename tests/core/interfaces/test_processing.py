"""Tests for shared processing protocols and input validation."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureContract,
    FeatureRole,
    PipelineStage,
    PredictionContext,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
    ValidationReport,
)
from taqamind.core.interfaces import (
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    ColumnMapper,
    ColumnMappingEntry,
    ColumnProfile,
    DataCleaner,
    DataIngestion,
    DataProfiler,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMetadata,
    DatasetProfile,
    DatasetState,
    ExecutionStatus,
    FeatureEngineer,
    MappingMethod,
    MappingProposal,
    PipelineContext,
    ProfilingResult,
    SemanticValidator,
    StageResult,
    StageTiming,
    validate_approved_mapping,
    validate_processing_input,
)


@pytest.fixture
def contract():
    predictor = FeatureContract(
        name="planned_quantity_kg",
        display_name="Planned Quantity",
        description="Planned production quantity.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        unit="kg",
    )

    target = TargetContract(
        target_id="waste_prediction",
        feature=FeatureContract(
            name="actual_waste_kg",
            display_name="Waste",
            description="Measured production waste.",
            data_type=DataType.FLOAT,
            role=FeatureRole.TARGET,
            availability=FeatureAvailability.POST_PRODUCTION,
            unit="kg",
        ),
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition="Measured production waste.",
            granularity="production_run",
        ),
    )

    return DatasetContract(
        dataset_id="textile_knitting_waste",
        domain="textile",
        process="knitting",
        description="Synthetic dataset.",
        granularity="production_run",
        features=(predictor,),
        target=target,
        prediction_context=PredictionContext(
            description="Predict before production.",
        ),
    )


@pytest.fixture
def context(contract):
    return PipelineContext(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        domain=contract.domain,
        process=contract.process,
        dataset_id=contract.dataset_id,
        task_id="waste_prediction",
        contract_version=contract.version,
        target_version=contract.target.version,
        task_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
    )


@pytest.fixture
def raw_artifact(context):
    metadata = DatasetMetadata.create_raw(
        context=context,
        source_file_sha256="a" * 64,
    )

    reference = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256="a" * 64,
        size_bytes=1024,
    )

    return DatasetArtifact(
        metadata=metadata,
        reference=reference,
    )


@pytest.fixture
def ingested_artifact(context, raw_artifact):
    metadata = raw_artifact.metadata.derive(
        context=context,
        state=DatasetState.INGESTED,
        row_count=100,
        columns=(
            DatasetColumnSchema(
                name="Planned Qty",
                physical_dtype="float64",
            ),
            DatasetColumnSchema(
                name="Waste",
                physical_dtype="float64",
            ),
        ),
        operation="csv_parsing",
        implementation_version="1.0.0",
        description="Synthetic ingestion.",
    )

    return DatasetArtifact(
        metadata=metadata,
        reference=ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256="b" * 64,
            size_bytes=2048,
        ),
    )


@pytest.fixture
def mapping(context, ingested_artifact):
    proposal = MappingProposal(
        factory_id=context.factory_id,
        execution_id=context.execution_id,
        dataset_revision_id=(
            ingested_artifact.metadata.revision_id
        ),
        entries=(
            ColumnMappingEntry(
                source_column="Planned Qty",
                canonical_column="planned_quantity_kg",
                method=MappingMethod.ALIAS,
                source_unit="kg",
                canonical_unit="kg",
                requires_confirmation=True,
            ),
            ColumnMappingEntry(
                source_column="Waste",
                canonical_column="actual_waste_kg",
                method=MappingMethod.ALIAS,
                source_unit="kg",
                canonical_unit="kg",
                requires_confirmation=True,
            ),
        ),
    )

    return ApprovedMapping(
        proposal=proposal,
        approved_by="synthetic_reviewer",
        confirmed_source_columns=(
            "Planned Qty",
            "Waste",
        ),
    )


def test_valid_raw_input(context, raw_artifact, contract):
    validate_processing_input(
        context=context,
        artifact=raw_artifact,
        contract=contract,
        expected_state=DatasetState.RAW,
    )


def test_valid_ingested_input(
    context,
    ingested_artifact,
    contract,
):
    validate_processing_input(
        context=context,
        artifact=ingested_artifact,
        contract=contract,
        expected_state=DatasetState.INGESTED,
    )


def test_wrong_input_state(
    context,
    raw_artifact,
    contract,
):
    with pytest.raises(ValueError, match="Expected dataset state"):
        validate_processing_input(
            context=context,
            artifact=raw_artifact,
            contract=contract,
            expected_state=DatasetState.MAPPED,
        )


def test_incompatible_contract(
    context,
    raw_artifact,
    contract,
):
    payload = contract.model_dump()
    payload["version"] = "2.0.0"

    updated = DatasetContract.model_validate(payload)

    with pytest.raises(ValueError, match="version"):
        validate_processing_input(
            context=context,
            artifact=raw_artifact,
            contract=updated,
            expected_state=DatasetState.RAW,
        )


def test_wrong_factory(
    context,
    raw_artifact,
    contract,
):
    payload = context.model_dump()
    payload["factory_id"] = uuid4()

    other = PipelineContext.model_validate(payload)

    with pytest.raises(ValueError, match="factory_id"):
        validate_processing_input(
            context=other,
            artifact=raw_artifact,
            contract=contract,
            expected_state=DatasetState.RAW,
        )


def test_valid_approved_mapping(
    context,
    ingested_artifact,
    contract,
    mapping,
):
    validate_approved_mapping(
        context=context,
        artifact=ingested_artifact,
        contract=contract,
        mapping=mapping,
    )


def test_mapping_execution_mismatch(
    context,
    ingested_artifact,
    contract,
    mapping,
):
    payload = mapping.model_dump()
    payload["proposal"]["execution_id"] = uuid4()

    updated = ApprovedMapping.model_validate(payload)

    with pytest.raises(ValueError, match="execution mismatch"):
        validate_approved_mapping(
            context=context,
            artifact=ingested_artifact,
            contract=contract,
            mapping=updated,
        )


def test_mapping_requires_all_training_columns(
    context,
    ingested_artifact,
    contract,
    mapping,
):
    payload = mapping.model_dump()

    payload["proposal"]["entries"] = (
        payload["proposal"]["entries"][0],
    )

    payload["proposal"]["unmapped_columns"] = ("Waste",)
    payload["confirmed_source_columns"] = ("Planned Qty",)

    updated = ApprovedMapping.model_validate(payload)

    with pytest.raises(
        ValueError,
        match="Required canonical columns",
    ):
        validate_approved_mapping(
            context=context,
            artifact=ingested_artifact,
            contract=contract,
            mapping=updated,
        )


def test_prediction_mapping_rejects_target(
    context,
    ingested_artifact,
    contract,
    mapping,
):
    payload = context.model_dump()

    payload["purpose"] = DatasetPurpose.PREDICTION
    payload["model_version_id"] = uuid4()

    prediction_context = PipelineContext.model_validate(
        payload
    )

    with pytest.raises(
        ValueError,
        match="cannot include the target",
    ):
        validate_approved_mapping(
            context=prediction_context,
            artifact=ingested_artifact,
            contract=contract,
            mapping=mapping,
        )


def test_mapping_unit_mismatch(
    context,
    ingested_artifact,
    contract,
    mapping,
):
    payload = mapping.model_dump()
    payload["proposal"]["entries"][0]["canonical_unit"] = "m"

    updated = ApprovedMapping.model_validate(payload)

    with pytest.raises(ValueError, match="Canonical unit mismatch"):
        validate_approved_mapping(
            context=context,
            artifact=ingested_artifact,
            contract=contract,
            mapping=updated,
        )


def test_mapping_unknown_source_unit(
    context,
    ingested_artifact,
    contract,
    mapping,
):
    payload = mapping.model_dump()
    payload["proposal"]["entries"][0]["source_unit"] = None

    updated = ApprovedMapping.model_validate(payload)

    with pytest.raises(ValueError, match="Unknown source unit"):
        validate_approved_mapping(
            context=context,
            artifact=ingested_artifact,
            contract=contract,
            mapping=updated,
        )


class SyntheticProcessor:
    """Protocol test double, not a production implementation."""

    def ingest(self, *, context, artifact, contract):
        raise NotImplementedError

    def profile(self, *, context, artifact, contract):
        raise NotImplementedError

    def propose_mapping(self, *, context, artifact, contract):
        raise NotImplementedError

    def apply_mapping(
        self,
        *,
        context,
        artifact,
        contract,
        mapping,
    ):
        raise NotImplementedError

    def validate(self, *, context, artifact, contract):
        raise NotImplementedError

    def clean(self, *, context, artifact, contract):
        raise NotImplementedError

    def engineer(self, *, context, artifact, contract):
        raise NotImplementedError


@pytest.mark.parametrize(
    "protocol",
    [
        DataIngestion,
        DataProfiler,
        ColumnMapper,
        SemanticValidator,
        DataCleaner,
        FeatureEngineer,
    ],
)
def test_structural_protocol_compatibility(protocol):
    processor = SyntheticProcessor()

    assert isinstance(processor, protocol)


@pytest.mark.parametrize(
    "protocol",
    [
        DataIngestion,
        DataProfiler,
        ColumnMapper,
        SemanticValidator,
        DataCleaner,
        FeatureEngineer,
    ],
)
def test_empty_object_is_not_compatible(protocol):
    assert not isinstance(object(), protocol)


def test_incomplete_mapper_is_not_compatible():
    class IncompleteMapper:
        def propose_mapping(self, **kwargs):
            raise NotImplementedError

    assert not isinstance(
        IncompleteMapper(),
        ColumnMapper,
    )


def make_profiling_result(context, artifact):
    started = datetime.now(UTC)

    report = ValidationReport(
        dataset_id=context.dataset_id,
        contract_version=context.contract_version,
        dataset_revision_id=artifact.metadata.revision_id,
        domain=context.domain,
        process=context.process,
        purpose=context.purpose,
        completed_stages=(PipelineStage.PROFILING,),
    )

    result = StageResult(
        context=context,
        stage=PipelineStage.PROFILING,
        status=ExecutionStatus.SUCCEEDED,
        input_artifact=artifact,
        report=report,
        timing=StageTiming(
            started_at=started,
            finished_at=datetime.now(UTC),
        ),
    )

    profile = DatasetProfile(
        factory_id=context.factory_id,
        dataset_revision_id=artifact.metadata.revision_id,
        row_count=artifact.metadata.row_count,
        columns=tuple(
            ColumnProfile(
                name=column.name,
                physical_dtype=column.physical_dtype,
                missing_count=0,
                unique_count=50,
            )
            for column in artifact.metadata.columns
        ),
    )

    return ProfilingResult(
        result=result,
        profile=profile,
    )


def test_valid_profiling_result(
    context,
    ingested_artifact,
):
    result = make_profiling_result(
        context,
        ingested_artifact,
    )

    assert result.result.can_continue
    assert result.profile.row_count == 100
    assert result.profile.column_count == 2


def test_successful_profiling_requires_profile(
    context,
    ingested_artifact,
):
    result = make_profiling_result(
        context,
        ingested_artifact,
    )

    with pytest.raises(ValidationError):
        ProfilingResult(
            result=result.result,
            profile=None,
        )


def test_profiling_revision_mismatch(
    context,
    ingested_artifact,
):
    result = make_profiling_result(
        context,
        ingested_artifact,
    )

    payload = result.profile.model_dump()
    payload["dataset_revision_id"] = uuid4()

    profile = DatasetProfile.model_validate(payload)

    with pytest.raises(
        ValidationError,
        match="revision mismatch",
    ):
        ProfilingResult(
            result=result.result,
            profile=profile,
        )


def test_profiling_schema_mismatch(
    context,
    ingested_artifact,
):
    result = make_profiling_result(
        context,
        ingested_artifact,
    )

    payload = result.profile.model_dump()
    payload["columns"][0]["name"] = "Unknown Column"

    profile = DatasetProfile.model_validate(payload)

    with pytest.raises(
        ValidationError,
        match="schema does not match",
    ):
        ProfilingResult(
            result=result.result,
            profile=profile,
        )