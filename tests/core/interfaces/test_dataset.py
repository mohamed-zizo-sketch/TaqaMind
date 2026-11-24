"""Tests for dataset metadata and revision lineage."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DatasetPurpose,
    PipelineStage,
)
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMetadata,
    DatasetState,
    PipelineContext,
)

SOURCE_HASH = "a" * 64

COLUMNS = (
    DatasetColumnSchema(
        name="Planned Quantity",
        physical_dtype="float64",
    ),
    DatasetColumnSchema(
        name="Run ID",
        physical_dtype="string",
    ),
)


def make_context(**overrides):
    data = {
        "factory_id": uuid4(),
        "dataset_revision_id": uuid4(),
        "domain": "textile",
        "process": "knitting",
        "task_id": "waste_prediction",
        "dataset_id": "textile_knitting_waste",
        "contract_version": "1.0.0",
        "task_version": "1.0.0",
        "target_version": "1.0.0",
        "purpose": DatasetPurpose.TRAINING,
    }

    data.update(overrides)

    return PipelineContext(**data)


@pytest.fixture
def context():
    return make_context()


@pytest.fixture
def raw(context):
    return DatasetMetadata.create_raw(
        context=context,
        source_file_sha256=SOURCE_HASH,
    )


def derive_dataset(
    metadata,
    context,
    state=DatasetState.INGESTED,
    **overrides,
):
    data = {
        "context": context,
        "state": state,
        "row_count": 1000,
        "columns": COLUMNS,
        "operation": "dataset_transformation",
        "implementation_version": "1.0.0",
        "description": "Synthetic transformation.",
    }

    data.update(overrides)

    return metadata.derive(**data)


@pytest.fixture
def ingested(raw, context):
    return derive_dataset(raw, context)


def test_raw_dataset_identity(raw, context):
    assert raw.revision_id == (
        context.dataset_revision_id
    )

    assert raw.root_revision_id == raw.revision_id
    assert raw.parent_revision_id is None
    assert raw.state == DatasetState.RAW


def test_raw_dataset_schema_is_unknown(raw):
    assert raw.row_count is None
    assert raw.column_count is None
    assert raw.schema_fingerprint is None


def test_input_context_matches(raw, context):
    raw.validate_input_context(context)


def test_input_context_rejects_wrong_revision(
    raw,
    context,
):
    payload = context.model_dump()
    payload["dataset_revision_id"] = uuid4()

    different_context = PipelineContext.model_validate(
        payload
    )

    with pytest.raises(
        ValueError,
        match="input revision mismatch",
    ):
        raw.validate_input_context(different_context)


def test_ingested_dataset_has_new_revision(raw, ingested):
    assert ingested.revision_id != raw.revision_id


def test_ingested_dataset_preserves_root(raw, ingested):
    assert ingested.root_revision_id == raw.root_revision_id


def test_ingested_dataset_references_parent(raw, ingested):
    assert ingested.parent_revision_id == raw.revision_id


def test_source_hash_is_preserved(raw, ingested):
    assert (
        ingested.source_file_sha256
        == raw.source_file_sha256
    )


def test_transformation_metadata(ingested, context):
    transformation = ingested.transformation

    assert transformation is not None
    assert transformation.stage == PipelineStage.INGESTION

    assert (
        transformation.execution_id
        == context.execution_id
    )


def test_ingested_dataset_statistics(ingested):
    assert ingested.row_count == 1000
    assert ingested.column_count == 2


def test_schema_fingerprint_exists(ingested):
    assert len(ingested.schema_fingerprint) == 64


def test_same_schema_has_same_fingerprint(raw, context):
    first = derive_dataset(raw, context)

    second = derive_dataset(
        raw,
        context,
        row_count=2000,
    )

    assert (
        first.schema_fingerprint
        == second.schema_fingerprint
    )


def test_changed_schema_changes_fingerprint(raw, context):
    first = derive_dataset(raw, context)

    second = derive_dataset(
        raw,
        context,
        columns=(
            DatasetColumnSchema(
                name="different_column",
                physical_dtype="float64",
            ),
        ),
    )

    assert (
        first.schema_fingerprint
        != second.schema_fingerprint
    )


def test_changed_dtype_changes_fingerprint(raw, context):
    first = derive_dataset(raw, context)

    second = derive_dataset(
        raw,
        context,
        columns=(
            DatasetColumnSchema(
                name="Planned Quantity",
                physical_dtype="string",
            ),
            COLUMNS[1],
        ),
    )

    assert (
        first.schema_fingerprint
        != second.schema_fingerprint
    )


def test_changed_column_order_changes_fingerprint(
    raw,
    context,
):
    first = derive_dataset(raw, context)

    second = derive_dataset(
        raw,
        context,
        columns=tuple(reversed(COLUMNS)),
    )

    assert (
        first.schema_fingerprint
        != second.schema_fingerprint
    )


def test_full_revision_chain(raw, context):
    ingested = derive_dataset(
        raw,
        context,
        DatasetState.INGESTED,
    )

    mapped = derive_dataset(
        ingested,
        context,
        DatasetState.MAPPED,
    )

    cleaned = derive_dataset(
        mapped,
        context,
        DatasetState.CLEANED,
        row_count=950,
    )

    engineered = derive_dataset(
        cleaned,
        context,
        DatasetState.ENGINEERED,
        row_count=950,
    )

    revisions = (
        raw,
        ingested,
        mapped,
        cleaned,
        engineered,
    )

    assert len({
        revision.revision_id
        for revision in revisions
    }) == 5

    for revision in revisions:
        assert (
            revision.root_revision_id
            == raw.revision_id
        )

    for parent, child in zip(
        revisions[:-1],
        revisions[1:],
        strict=True,
    ):
        assert (
            child.parent_revision_id
            == parent.revision_id
        )


def test_original_metadata_remains_unchanged(raw, context):
    derive_dataset(raw, context)

    assert raw.state == DatasetState.RAW
    assert raw.row_count is None
    assert raw.transformation is None


def test_cannot_skip_states(raw, context):
    with pytest.raises(ValueError, match="cannot skip"):
        derive_dataset(
            raw,
            context,
            DatasetState.ENGINEERED,
        )


def test_cannot_reverse_state(ingested, context):
    with pytest.raises(ValueError, match="must advance"):
        derive_dataset(
            ingested,
            context,
            DatasetState.RAW,
        )


def test_cannot_repeat_same_state(ingested, context):
    with pytest.raises(ValueError, match="must advance"):
        derive_dataset(
            ingested,
            context,
            DatasetState.INGESTED,
        )


def test_duplicate_column_names(raw, context):
    with pytest.raises(ValidationError):
        derive_dataset(
            raw,
            context,
            columns=(COLUMNS[0], COLUMNS[0]),
        )


def test_negative_row_count(raw, context):
    with pytest.raises(ValidationError):
        derive_dataset(
            raw,
            context,
            row_count=-10,
        )


def test_empty_rows_are_representable(raw, context):
    result = derive_dataset(
        raw,
        context,
        row_count=0,
    )

    assert result.row_count == 0


def test_processed_dataset_requires_schema(raw, context):
    with pytest.raises(ValidationError):
        derive_dataset(
            raw,
            context,
            columns=(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("factory_id", uuid4()),
        ("domain", "food_beverage"),
        ("process", "weaving"),
        ("dataset_id", "different_dataset"),
        ("contract_version", "2.0.0"),
        ("task_id", "energy_prediction"),
        ("task_version", "2.0.0"),
        ("target_version", "2.0.0"),
    ],
)
def test_incompatible_context_rejected(
    raw,
    context,
    field,
    value,
):
    payload = context.model_dump()
    payload[field] = value

    other_context = PipelineContext.model_validate(
        payload
    )

    with pytest.raises(ValueError, match=field):
        derive_dataset(raw, other_context)


def test_valid_raw_artifact(raw):
    reference = ArtifactReference(
        factory_id=raw.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=SOURCE_HASH,
        size_bytes=1024,
    )

    artifact = DatasetArtifact(
        metadata=raw,
        reference=reference,
    )

    assert artifact.metadata == raw


def test_artifact_factory_mismatch(raw):
    reference = ArtifactReference(
        factory_id=uuid4(),
        format=ArtifactFormat.CSV,
        content_sha256=SOURCE_HASH,
        size_bytes=1024,
    )

    with pytest.raises(
        ValidationError,
        match="Artifact factory",
    ):
        DatasetArtifact(
            metadata=raw,
            reference=reference,
        )


def test_raw_artifact_hash_mismatch(raw):
    reference = ArtifactReference(
        factory_id=raw.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256="b" * 64,
        size_bytes=1024,
    )

    with pytest.raises(
        ValidationError,
        match="source hash",
    ):
        DatasetArtifact(
            metadata=raw,
            reference=reference,
        )


def test_raw_artifact_must_be_csv(raw):
    reference = ArtifactReference(
        factory_id=raw.factory_id,
        format=ArtifactFormat.PARQUET,
        content_sha256=SOURCE_HASH,
        size_bytes=1024,
    )

    with pytest.raises(ValidationError):
        DatasetArtifact(
            metadata=raw,
            reference=reference,
        )


def test_metadata_is_frozen(raw):
    with pytest.raises(ValidationError):
        raw.state = DatasetState.CLEANED


def test_metadata_json_round_trip(ingested):
    restored = DatasetMetadata.model_validate_json(
        ingested.model_dump_json()
    )

    assert restored == ingested