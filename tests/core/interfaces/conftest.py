"""Shared fixtures for core interface tests."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
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


@pytest.fixture
def context():
    return PipelineContext(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        domain="textile",
        process="knitting",
        task_id="waste_prediction",
        dataset_id="textile_knitting_waste",
        contract_version="1.0.0",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
        created_at=datetime.now(UTC) - timedelta(seconds=10),
    )


@pytest.fixture
def raw_artifact(context):
    metadata = DatasetMetadata.create_raw(
        context=context,
        source_file_sha256=SOURCE_HASH,
    )

    reference = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=SOURCE_HASH,
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
        row_count=1000,
        columns=(
            DatasetColumnSchema(
                name="planned_quantity_kg",
                physical_dtype="float64",
            ),
        ),
        operation="csv_parsing",
        implementation_version="1.0.0",
        description="Synthetic ingestion.",
    )

    reference = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256="b" * 64,
        size_bytes=2048,
    )

    return DatasetArtifact(
        metadata=metadata,
        reference=reference,
    )