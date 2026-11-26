"""Tests for record identity, versioning, lineage, and data quality auditing."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import PipelineContext, RecordMatchStatus
from taqamind.domains.food_beverage.pipeline import FoodBeveragePipeline
from tests.domains.food_beverage.conftest import (
    FACTORY_A,
    FACTORY_B,
    make_synthetic_bakery_dataframe,
)


def test_lineage_distinguishes_new_unchanged_corrected_and_cross_factory(
    tmp_pipeline: FoodBeveragePipeline,
    tmp_path: Path,
    synthetic_confirmation,
) -> None:
    df1 = make_synthetic_bakery_dataframe(
        factory_id=FACTORY_A, n_rows=10, seed=1, include_target=True
    )
    p1 = tmp_path / "f1_v1.csv"
    df1.to_csv(p1, index=False)

    prep1 = tmp_pipeline.prepare_dataset(
        csv_path=p1,
        factory_id=FACTORY_A,
        process_id="bakery",
        purpose=DatasetPurpose.TRAINING,
        target_confirmation=synthetic_confirmation,
    )
    assert len(prep1.lineage.new_record_ids) == 10
    assert len(prep1.lineage.unchanged_record_ids) == 0

    # Re-upload exact same content -> unchanged
    prep2 = tmp_pipeline.prepare_dataset(
        csv_path=p1,
        factory_id=FACTORY_A,
        process_id="bakery",
        purpose=DatasetPurpose.TRAINING,
        target_confirmation=synthetic_confirmation,
    )
    assert len(prep2.lineage.unchanged_record_ids) == 10
    assert (
        prep2.engineered_artifact.metadata.source_file_sha256
        == prep1.engineered_artifact.metadata.source_file_sha256
    )

    # Modify 1 row -> corrected version with same record_id!
    df2 = df1.copy()
    df2.loc[0, "waste_quantity_kg"] = float(df2.loc[0, "waste_quantity_kg"]) + 5.0
    p2 = tmp_path / "f1_v2.csv"
    df2.to_csv(p2, index=False)

    prep3 = tmp_pipeline.prepare_dataset(
        csv_path=p2,
        factory_id=FACTORY_A,
        process_id="bakery",
        purpose=DatasetPurpose.TRAINING,
        target_confirmation=synthetic_confirmation,
    )
    assert len(prep3.lineage.corrected_record_ids) == 1
    assert len(prep3.lineage.unchanged_record_ids) == 9
    assert prep3.lineage.row_to_record_id[0] == prep1.lineage.row_to_record_id[0]

    # Same data uploaded under FACTORY_B -> completely isolated new records
    df_b = make_synthetic_bakery_dataframe(
        factory_id=FACTORY_B, n_rows=10, seed=1, include_target=True
    )
    pb = tmp_path / "fb.csv"
    df_b.to_csv(pb, index=False)
    prep_b = tmp_pipeline.prepare_dataset(
        csv_path=pb,
        factory_id=FACTORY_B,
        process_id="bakery",
        purpose=DatasetPurpose.TRAINING,
        target_confirmation=synthetic_confirmation,
    )
    assert len(prep_b.lineage.new_record_ids) == 10
    assert set(prep_b.lineage.new_record_ids).isdisjoint(set(prep1.lineage.new_record_ids))


def test_ambiguous_duplicate_business_keys_detected(
    tmp_pipeline: FoodBeveragePipeline,
    tmp_path: Path,
    synthetic_confirmation,
) -> None:
    df = make_synthetic_bakery_dataframe(n_rows=10, include_target=True)
    df.loc[1, "batch_id"] = df.loc[0, "batch_id"]  # duplicate business key
    p = tmp_path / "dup_bkey.csv"
    df.to_csv(p, index=False)

    with pytest.raises(ValueError, match="Duplicate business key"):
        tmp_pipeline.prepare_dataset(
            csv_path=p,
            factory_id=FACTORY_A,
            process_id="bakery",
            purpose=DatasetPurpose.TRAINING,
            target_confirmation=synthetic_confirmation,
        )

    contract = tmp_pipeline.get_contract("bakery", target_confirmation=synthetic_confirmation)
    ctx = PipelineContext(
        execution_id=uuid4(),
        factory_id=FACTORY_A,
        domain="food_beverage",
        process="bakery",
        dataset_id=contract.dataset_id,
        dataset_revision_id=uuid4(),
        contract_version=contract.version,
        task_id="food_beverage_bakery_waste_v1",
        task_version="1.0.0",
        target_version=contract.target.version,
        purpose=DatasetPurpose.TRAINING,
    )
    lineage = tmp_pipeline.membership_builder.register_rows(
        context=ctx,
        contract=contract,
        dataset_revision_id=ctx.dataset_revision_id,
        df=df,
    )
    assert len(lineage.ambiguous_row_indices) == 2
    assert lineage.match_results[0].status == RecordMatchStatus.AMBIGUOUS
