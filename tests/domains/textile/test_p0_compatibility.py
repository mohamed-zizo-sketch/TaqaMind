"""Textile P0 Shared Core compatibility regression tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from taqamind.core.contracts import DatasetPurpose, PipelineStage
from taqamind.core.interfaces import (
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    ColumnMappingEntry,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMetadata,
    DatasetState,
    MappingMethod,
    MappingProposal,
    PipelineContext,
    TransformationRecord,
    calculate_bytes_sha256,
)
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.domains.textile.adapters.unit_mapper import UnitAwareColumnMapperAdapter
from taqamind.domains.textile.common.formulas import (
    compute_knitting_tightness_factor,
    compute_weaving_thread_density_sum,
)
from taqamind.domains.textile.weaving import (
    build_synthetic_weaving_confirmation,
    build_weaving_dataset_contract,
    generate_synthetic_weaving_training_frame,
)


def test_unit_mapper_rejects_non_numeric_values_without_silent_coercion(tmp_path: Path) -> None:
    """TI-009: Invalid numeric tokens in unit-converted columns must raise, never become NaN."""
    storage = LocalDatasetArtifactStorage(tmp_path / "blobs")
    mapper = UnitAwareColumnMapperAdapter(storage)
    contract = build_weaving_dataset_contract(confirmation=build_synthetic_weaving_confirmation())
    factory_id = uuid4()

    df = generate_synthetic_weaving_training_frame(num_orders=1, batches_per_order=2).copy()
    df["planned_beam_length_yds"] = df["planned_beam_length_yds"].astype(object)
    df.loc[df.index[1], "planned_beam_length_yds"] = "not_a_number"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    ref = ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=calculate_bytes_sha256(csv_bytes),
        size_bytes=len(csv_bytes),
    )
    storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(csv_bytes))
    root_id = uuid4()
    rev_id = uuid4()
    artifact = DatasetArtifact(
        metadata=DatasetMetadata(
            revision_id=rev_id,
            root_revision_id=root_id,
            factory_id=factory_id,
            domain=contract.domain,
            process=contract.process,
            dataset_id=contract.dataset_id,
            contract_version=contract.version,
            task_id="weaving_waste_regression",
            task_version="1.0.0",
            target_version=contract.target.version,
            source_file_sha256=ref.content_sha256,
            state=DatasetState.INGESTED,
            row_count=len(df),
            columns=tuple(
                DatasetColumnSchema(name=c, physical_dtype=str(df[c].dtype)) for c in df.columns
            ),
            transformation=TransformationRecord(
                execution_id=uuid4(),
                parent_revision_id=root_id,
                stage=PipelineStage.INGESTION,
                operation="csv_ingestion",
                implementation_version="1.0.0",
                description="Ingested",
            ),
        ),
        reference=ref,
    )
    ctx = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=rev_id,
        domain=contract.domain,
        process=contract.process,
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_id="weaving_waste_regression",
        task_version="1.0.0",
        target_version=contract.target.version,
        purpose=DatasetPurpose.TRAINING,
    )
    entries = []
    for col in df.columns:
        feat = contract.column_map.get(col)
        canon_unit = feat.unit if feat is not None else None
        entries.append(
            ColumnMappingEntry(
                source_column=col,
                canonical_column=col,
                method=MappingMethod.EXACT,
                source_unit=("m" if col == "planned_beam_length_yds" else canon_unit),
                canonical_unit=canon_unit,
                requires_confirmation=False,
            )
        )
    mapping = ApprovedMapping(
        proposal=MappingProposal(
            factory_id=factory_id,
            execution_id=ctx.execution_id,
            dataset_revision_id=rev_id,
            entries=tuple(entries),
        ),
        approved_by="test_harness",
        confirmed_source_columns=(),
    )

    with pytest.raises(ValueError, match="non-numeric values"):
        mapper.apply_mapping(context=ctx, artifact=artifact, contract=contract, mapping=mapping)


def test_derived_formulas_are_row_and_order_independent() -> None:
    """Task 03: Derived formulas must be deterministic and independent of unrelated rows."""
    epi = pd.Series([60.0, 70.0, np.nan, -5.0])
    ppi = pd.Series([50.0, 55.0, 40.0, 40.0])
    w_res = compute_weaving_thread_density_sum(epi, ppi)
    assert w_res.iloc[0] == 110.0
    assert w_res.iloc[1] == 125.0
    assert pd.isna(w_res.iloc[2])
    assert pd.isna(w_res.iloc[3])

    # Reverse order yields identical row-wise values
    w_rev = compute_weaving_thread_density_sum(epi.iloc[::-1], ppi.iloc[::-1])
    pd.testing.assert_series_equal(w_res, w_rev.iloc[::-1])

    tex = pd.Series([25.0, 36.0, 0.0, -4.0, np.nan])
    stitch = pd.Series([2.5, 3.0, 2.5, 2.5, 2.5])
    k_res = compute_knitting_tightness_factor(tex, stitch)
    assert k_res.iloc[0] == pytest.approx(20.0)
    assert k_res.iloc[1] == pytest.approx(20.0)
    assert pd.isna(k_res.iloc[2])
    assert pd.isna(k_res.iloc[3])
    assert pd.isna(k_res.iloc[4])

    k_rev = compute_knitting_tightness_factor(tex.iloc[::-1], stitch.iloc[::-1])
    pd.testing.assert_series_equal(k_res, k_rev.iloc[::-1])
