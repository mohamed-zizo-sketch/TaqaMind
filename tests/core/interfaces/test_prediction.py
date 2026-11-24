"""Tests for factory-scoped regression prediction contracts."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    RateDefinition,
)
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMetadata,
    DatasetState,
    DerivedRatePrediction,
    PipelineContext,
    PredictionBatchResult,
    PredictionBatchStatus,
    PredictionModelIdentity,
    PredictionStatus,
    RecordPrediction,
    StageTiming,
)
from tests.core.interfaces.test_readiness import make_contract


def make_prediction_contract():
    contract = make_contract(confirmed=True)

    payload = contract.model_dump()

    payload["target"]["feature"]["constraints"]["minimum"] = 0

    payload["target"]["derived_rate"] = RateDefinition(
        name="waste_rate_pct",
        numerator="actual_waste_kg",
        denominator="planned_quantity_kg",
        granularity="production_run",
    )

    return DatasetContract.model_validate(payload)


def make_prediction_batch(**overrides):
    contract = make_prediction_contract()

    context = PipelineContext(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        domain=contract.domain,
        process=contract.process,
        task_id="waste_prediction",
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_version="1.0.0",
        target_version=contract.target.version,
        purpose=DatasetPurpose.PREDICTION,
        model_version_id=uuid4(),
    )

    metadata = DatasetMetadata.create_raw(
        context=context,
        source_file_sha256="a" * 64,
    )

    columns = (
        DatasetColumnSchema(
            name="planned_quantity_kg",
            physical_dtype="float64",
        ),
    )

    for state in (
        DatasetState.INGESTED,
        DatasetState.MAPPED,
        DatasetState.CLEANED,
        DatasetState.ENGINEERED,
    ):
        metadata = metadata.derive(
            context=context,
            state=state,
            row_count=2,
            columns=columns,
            operation="synthetic_processing",
            implementation_version="1.0.0",
            description="Synthetic prediction dataset.",
        )

    artifact = DatasetArtifact(
        metadata=metadata,
        reference=ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256="b" * 64,
            size_bytes=1024,
        ),
    )

    model = PredictionModelIdentity(
        model_version_id=context.model_version_id,
        factory_id=context.factory_id,
        domain=context.domain,
        process=context.process,
        task_id=context.task_id,
        dataset_id=context.dataset_id,
        contract_version=context.contract_version,
        task_version=context.task_version,
        target_version=context.target_version,
    )

    predictions = (
        RecordPrediction(
            record_id=uuid4(),
            status=PredictionStatus.SUCCEEDED,
            predicted_value=30.0,
            derived_rate=DerivedRatePrediction(
                value=3.0,
                denominator_value=1000.0,
                unit="%",
            ),
        ),
        RecordPrediction(
            record_id=uuid4(),
            status=PredictionStatus.SUCCEEDED,
            predicted_value=45.0,
            derived_rate=DerivedRatePrediction(
                value=3.0,
                denominator_value=1500.0,
                unit="%",
            ),
        ),
    )

    data = {
        "context": context,
        "model": model,
        "contract": contract,
        "input_artifact": artifact,
        "predictions": predictions,
        "timing": StageTiming(
            started_at=context.created_at,
            finished_at=datetime.now(UTC),
        ),
    }

    data.update(overrides)

    return PredictionBatchResult(**data)


def test_successful_prediction_batch():
    batch = make_prediction_batch()

    assert batch.status == PredictionBatchStatus.SUCCEEDED
    assert batch.successful_count == 2
    assert batch.rejected_count == 0
    assert batch.target_unit == "kg"


def test_partial_prediction_batch():
    batch = make_prediction_batch()

    payload = batch.model_dump()

    predictions = list(payload["predictions"])

    predictions[1] = {
        "record_id": batch.predictions[1].record_id,
        "status": PredictionStatus.REJECTED,
        "rejection_reason": "Invalid industrial input.",
    }

    payload["predictions"] = tuple(predictions)

    restored = PredictionBatchResult.model_validate(
        payload
    )

    assert restored.status == PredictionBatchStatus.PARTIAL
    assert restored.successful_count == 1
    assert restored.rejected_count == 1


def test_all_predictions_rejected():
    batch = make_prediction_batch()

    payload = batch.model_dump()

    payload["predictions"] = tuple(
        {
            "record_id": prediction.record_id,
            "status": PredictionStatus.REJECTED,
            "rejection_reason": "Invalid industrial input.",
        }
        for prediction in batch.predictions
    )

    restored = PredictionBatchResult.model_validate(
        payload
    )

    assert restored.status == PredictionBatchStatus.REJECTED


def test_successful_record_requires_prediction():
    with pytest.raises(ValidationError):
        RecordPrediction(
            record_id=uuid4(),
            status=PredictionStatus.SUCCEEDED,
        )


def test_rejected_record_requires_reason():
    with pytest.raises(ValidationError):
        RecordPrediction(
            record_id=uuid4(),
            status=PredictionStatus.REJECTED,
        )


def test_rejected_record_cannot_expose_value():
    with pytest.raises(ValidationError):
        RecordPrediction(
            record_id=uuid4(),
            status=PredictionStatus.REJECTED,
            predicted_value=10.0,
            rejection_reason="Invalid input.",
        )


def test_nonfinite_prediction_rejected():
    with pytest.raises(ValidationError):
        RecordPrediction(
            record_id=uuid4(),
            status=PredictionStatus.SUCCEEDED,
            predicted_value=float("inf"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("factory_id", uuid4()),
        ("model_version_id", uuid4()),
        ("domain", "food_beverage"),
        ("process", "weaving"),
        ("task_id", "energy_prediction"),
        ("dataset_id", "another_dataset"),
        ("contract_version", "2.0.0"),
        ("task_version", "2.0.0"),
        ("target_version", "2.0.0"),
    ],
)
def test_incompatible_model_rejected(field, value):
    batch = make_prediction_batch()

    payload = batch.model_dump()
    payload["model"][field] = value

    with pytest.raises(
        ValidationError,
        match=field,
    ):
        PredictionBatchResult.model_validate(payload)


def test_negative_waste_prediction_rejected():
    batch = make_prediction_batch()

    payload = batch.model_dump()

    payload["predictions"][0]["predicted_value"] = -5.0
    payload["predictions"][0]["derived_rate"] = None

    with pytest.raises(
        ValidationError,
        match="below the target minimum",
    ):
        PredictionBatchResult.model_validate(payload)


def test_invalid_derived_rate_rejected():
    batch = make_prediction_batch()

    payload = batch.model_dump()

    payload["predictions"][0]["derived_rate"]["value"] = 99.0

    with pytest.raises(
        ValidationError,
        match="does not match predicted quantity",
    ):
        PredictionBatchResult.model_validate(payload)


def test_zero_denominator_cannot_produce_defined_rate():
    with pytest.raises(ValidationError):
        DerivedRatePrediction(
            value=5.0,
            denominator_value=0,
            unit="%",
        )


def test_undefined_rate_is_representable():
    rate = DerivedRatePrediction(
        value=None,
        denominator_value=0,
        unit="%",
        undefined_reason="Denominator is zero.",
    )

    assert rate.value is None


def test_duplicate_prediction_records():
    batch = make_prediction_batch()

    payload = batch.model_dump()

    payload["predictions"][1]["record_id"] = (
        batch.predictions[0].record_id
    )

    with pytest.raises(
        ValidationError,
        match="identifiers must be unique",
    ):
        PredictionBatchResult.model_validate(payload)


def test_missing_prediction_result():
    batch = make_prediction_batch()

    payload = batch.model_dump()

    payload["predictions"] = (
        payload["predictions"][0],
    )

    with pytest.raises(
        ValidationError,
        match="does not match input rows",
    ):
        PredictionBatchResult.model_validate(payload)


def test_prediction_record_coverage():
    batch = make_prediction_batch()

    expected = tuple(
        prediction.record_id
        for prediction in batch.predictions
    )

    batch.validate_record_coverage(expected)


def test_prediction_wrong_record_coverage():
    batch = make_prediction_batch()

    with pytest.raises(
        ValueError,
        match="expected records",
    ):
        batch.validate_record_coverage(
            (uuid4(), uuid4())
        )


def test_prediction_artifact_cannot_contain_target():
    batch = make_prediction_batch()

    payload = batch.model_dump()

    payload["input_artifact"]["metadata"]["columns"] += (
        {
            "name": "actual_waste_kg",
            "physical_dtype": "float64",
        },
    )

    with pytest.raises(
        ValidationError,
        match="cannot contain the target",
    ):
        PredictionBatchResult.model_validate(payload)


def test_prediction_serialization():
    batch = make_prediction_batch()

    restored = PredictionBatchResult.model_validate_json(
        batch.model_dump_json()
    )

    assert restored == batch


def test_prediction_batch_is_frozen():
    batch = make_prediction_batch()

    with pytest.raises(ValidationError):
        batch.prediction_id = uuid4()