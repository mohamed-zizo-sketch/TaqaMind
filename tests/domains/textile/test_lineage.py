"""Tests for dataset registration, record lineage, correction versioning, and factory isolation."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import RegistryNotFoundError
from taqamind.domains.textile.pipeline import TextilePipelineService
from taqamind.domains.textile.weaving import (
    build_synthetic_weaving_confirmation,
    build_weaving_dataset_contract,
    generate_synthetic_weaving_training_frame,
)


def test_lineage_new_exact_corrected_and_factory_isolation(tmp_path: Path) -> None:
    service = TextilePipelineService(tmp_path / "store")
    contract = build_weaving_dataset_contract(confirmation=build_synthetic_weaving_confirmation())
    factory_a = uuid4()
    factory_b = uuid4()

    df = generate_synthetic_weaving_training_frame(num_orders=5, batches_per_order=2, seed=11)
    csv_1 = tmp_path / "batch1.csv"
    df.to_csv(csv_1, index=False)

    prep1 = service.ingest_and_prepare(
        factory_id=factory_a,
        process_id="weaving",
        csv_path=csv_1,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    assert prep1.lineage_summary is not None
    assert prep1.lineage_summary.new_records == len(df)
    assert prep1.lineage_summary.exact_matches == 0

    # Upload an extended CSV with the same 10 rows + 2 new rows.
    df_extra = generate_synthetic_weaving_training_frame(num_orders=2, batches_per_order=1, seed=99)
    df_extra["batch_record_id"] = ["WB-NEW-01", "WB-NEW-02"]
    df_extra["order_id"] = ["ORD-NEW-01", "ORD-NEW-02"]
    df_combined = pd_concat(df, df_extra)
    csv_2 = tmp_path / "batch2.csv"
    df_combined.to_csv(csv_2, index=False)

    prep2 = service.ingest_and_prepare(
        factory_id=factory_a,
        process_id="weaving",
        csv_path=csv_2,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    assert prep2.lineage_summary is not None
    assert prep2.lineage_summary.exact_matches == len(df)
    assert prep2.lineage_summary.new_records == 2

    # Unconfirmed modification to an existing record must be blocked.
    df_modified = df.copy()
    df_modified.loc[0, "waste_length_yds"] = float(df_modified.loc[0, "waste_length_yds"]) + 5.0
    csv_mod = tmp_path / "batch_mod.csv"
    df_modified.to_csv(csv_mod, index=False)
    with pytest.raises(ValueError, match="without explicit confirmation"):
        service.ingest_and_prepare(
            factory_id=factory_a,
            process_id="weaving",
            csv_path=csv_mod,
            purpose=DatasetPurpose.TRAINING,
            contract=contract,
        )

    # Factory B cannot see Factory A's dataset revision.
    with pytest.raises(RegistryNotFoundError):
        service.stores.dataset_registry.get(
            factory_id=factory_b,
            dataset_revision_id=prep1.raw_artifact.metadata.revision_id,
        )


def pd_concat(left, right):
    import pandas as pd

    return pd.concat([left, right], ignore_index=True)
