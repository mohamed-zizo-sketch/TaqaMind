"""Unit-safety regression tests for AUD-001 and TargetConfirmation defaults.

Covers:
- UnitAwareColumnMapperAdapter.apply_mapping rejects incompatible unit pairs with ValueError.
- TargetConfirmation() defaults to ConfirmationStatus.UNCONFIRMED.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from taqamind.core.contracts import ConfirmationStatus, TargetConfirmation
from taqamind.core.interfaces import (
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    ColumnMappingEntry,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    MappingMethod,
    MappingProposal,
    PipelineContext,
    calculate_bytes_sha256,
)
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.domains.food_beverage.adapters.unit_mapper import UnitAwareColumnMapperAdapter


def _make_storage(tmp_path: Path) -> LocalDatasetArtifactStorage:
    return LocalDatasetArtifactStorage(tmp_path / "blobs")


def _make_context(factory_id=None) -> PipelineContext:
    fid = factory_id or uuid4()
    return PipelineContext(
        execution_id=uuid4(),
        factory_id=fid,
        domain="food_beverage",
        process="bakery",
        dataset_id="bakery_dataset",
        dataset_revision_id=uuid4(),
        contract_version="1.0.0",
        task_id="bakery_task",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose="training",
    )


def _store_csv(storage: LocalDatasetArtifactStorage, factory_id, df: pd.DataFrame):
    csv_bytes = df.to_csv(index=False).encode()
    sha = calculate_bytes_sha256(csv_bytes)
    rev = uuid4()
    ref = ArtifactReference(
        artifact_id=rev,
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha,
        size_bytes=len(csv_bytes),
    )
    storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(csv_bytes))
    meta = DatasetMetadata(
        revision_id=rev,
        root_revision_id=rev,
        factory_id=factory_id,
        domain="food_beverage",
        process="bakery",
        dataset_id="bakery_dataset",
        contract_version="1.0.0",
        task_id="bakery_task",
        task_version="1.0.0",
        target_version="1.0.0",
        source_file_sha256=sha,
        state=DatasetState.RAW,
        row_count=None,
        columns=(),
    )
    return DatasetArtifact(metadata=meta, reference=ref)


def _make_incompatible_mapping(
    context: PipelineContext,
    artifact: DatasetArtifact,
    *,
    source_column: str,
    source_unit: str,
    canonical_unit: str,
) -> ApprovedMapping:
    """Build an ApprovedMapping with an incompatible unit pair."""
    entry = ColumnMappingEntry(
        source_column=source_column,
        canonical_column=source_column,
        method=MappingMethod.EXACT,
        source_unit=source_unit,
        canonical_unit=canonical_unit,
        requires_confirmation=False,
    )
    proposal = MappingProposal(
        factory_id=context.factory_id,
        execution_id=context.execution_id,
        dataset_revision_id=artifact.metadata.revision_id,
        entries=(entry,),
    )
    return ApprovedMapping(
        proposal=proposal,
        approved_by="test_harness",
        confirmed_source_columns=(),
    )


class TestTargetConfirmationDefault:
    def test_default_status_is_unconfirmed(self) -> None:
        """TargetConfirmation() must default to UNCONFIRMED (AUD-008 guard)."""
        tc = TargetConfirmation()
        assert tc.status == ConfirmationStatus.UNCONFIRMED


class TestUnitAwareColumnMapperAdapterRejectsInvalidUnits:
    def test_incompatible_unit_raises_value_error(self, tmp_path: Path) -> None:
        """apply_mapping must raise ValueError for unit pairs not in UNIT_CONVERSION_FACTORS."""
        storage = _make_storage(tmp_path)
        mapper = UnitAwareColumnMapperAdapter(storage)

        factory_id = uuid4()
        df = pd.DataFrame({"batch_count": [10, 20, 30]})
        artifact = _store_csv(storage, factory_id, df)
        context = _make_context(factory_id)
        mapping = _make_incompatible_mapping(
            context,
            artifact,
            source_column="batch_count",
            source_unit="count",
            canonical_unit="kg",
        )

        with pytest.raises(ValueError, match="INCOMPATIBLE_UNIT"):
            mapper.apply_mapping(
                context=context,
                artifact=artifact,
                contract=None,  # type: ignore[arg-type]
                mapping=mapping,
            )

    def test_unknown_source_unit_raises_value_error(self, tmp_path: Path) -> None:
        """apply_mapping must raise ValueError for a completely unknown source unit."""
        storage = _make_storage(tmp_path)
        mapper = UnitAwareColumnMapperAdapter(storage)

        factory_id = uuid4()
        df = pd.DataFrame({"energy": [100, 200, 300]})
        artifact = _store_csv(storage, factory_id, df)
        context = _make_context(factory_id)
        mapping = _make_incompatible_mapping(
            context,
            artifact,
            source_column="energy",
            source_unit="kWh",
            canonical_unit="kg",
        )

        with pytest.raises(ValueError, match="INCOMPATIBLE_UNIT"):
            mapper.apply_mapping(
                context=context,
                artifact=artifact,
                contract=None,  # type: ignore[arg-type]
                mapping=mapping,
            )
