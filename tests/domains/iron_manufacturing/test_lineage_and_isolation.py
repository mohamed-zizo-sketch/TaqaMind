"""Record lineage, duplicate prevention, immutability, and cross-factory isolation tests."""

from __future__ import annotations

import io
from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import (
    ArtifactConflictError,
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetMetadata,
    PipelineContext,
    RecordMatchStatus,
    RegistryNotFoundError,
    calculate_bytes_sha256,
)
from taqamind.domains.iron_manufacturing import (
    IronManufacturingPipeline,
    get_iron_task,
)
from taqamind.domains.iron_manufacturing.adapters.readiness_and_splitting import (
    build_record_scope,
)
from taqamind.domains.iron_manufacturing.common.synthetic import (
    generate_synthetic_eaf_dataset,
)
from taqamind.domains.iron_manufacturing.services.lineage_service import (
    register_dataset_lineage,
)
from tests.domains.iron_manufacturing.conftest import FACTORY_A_ID, FACTORY_B_ID


def _make_engineered_artifact(
    pipeline: IronManufacturingPipeline,
    factory_id,
    df,
    confirmation,
):
    task = get_iron_task("eaf_meltshop", confirmation=confirmation)
    contract = task.dataset_contract
    raw_bytes = df.to_csv(index=False).encode("utf-8")
    sha = calculate_bytes_sha256(raw_bytes)
    ctx = PipelineContext.from_task(
        factory_id=factory_id,
        dataset_revision_id=uuid4(),
        task=task,
        purpose=DatasetPurpose.TRAINING,
    )
    ref = ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha,
        size_bytes=len(raw_bytes),
    )
    pipeline.dataset_storage.put(factory_id=factory_id, reference=ref, stream=io.BytesIO(raw_bytes))
    raw_art = DatasetArtifact(
        metadata=DatasetMetadata.create_raw(context=ctx, source_file_sha256=sha),
        reference=ref,
    )
    reg_res = pipeline.dataset_registry.register(factory_id=factory_id, artifact=raw_art)
    ing = pipeline.ingestion.ingest(context=ctx, artifact=raw_art, contract=contract)
    prop = pipeline.column_mapper.propose_mapping(
        context=ctx, artifact=ing.output_artifact, contract=contract
    )
    from taqamind.core.interfaces import ApprovedMapping

    approved = ApprovedMapping(proposal=prop, approved_by="tester")
    mapped = pipeline.column_mapper.apply_mapping(
        context=ctx, artifact=ing.output_artifact, contract=contract, mapping=approved
    )
    cleaned = pipeline.data_cleaner.clean(
        context=ctx, artifact=mapped.output_artifact, contract=contract
    )
    eng = pipeline.feature_engineer.engineer(
        context=ctx, artifact=cleaned.output_artifact, contract=contract
    )
    return ctx, contract, eng.output_artifact, reg_res


def test_exact_file_detection_and_immutable_storage(
    pipeline: IronManufacturingPipeline,
    synthetic_confirmation,
) -> None:
    df = generate_synthetic_eaf_dataset(n_rows=15, seed=10, include_target=True)
    _, _, _, reg1 = _make_engineered_artifact(pipeline, FACTORY_A_ID, df, synthetic_confirmation)
    _, _, _, reg2 = _make_engineered_artifact(pipeline, FACTORY_A_ID, df, synthetic_confirmation)

    assert not reg1.has_matching_source
    assert reg2.has_matching_source
    assert reg1.artifact.metadata.revision_id in reg2.matching_source_revision_ids

    # Immutable storage rejects overwriting existing artifact_id with different bytes
    raw_ref = reg1.artifact.reference
    different_bytes = b"mutated,csv\n1,2\n"
    mutated_ref = raw_ref.model_copy(
        update={
            "content_sha256": calculate_bytes_sha256(different_bytes),
            "size_bytes": len(different_bytes),
        }
    )
    with pytest.raises(ArtifactConflictError):
        pipeline.dataset_storage.put(
            factory_id=FACTORY_A_ID,
            reference=mutated_ref,
            stream=io.BytesIO(different_bytes),
        )


def test_identical_features_distinct_business_keys_and_corrected_versions(
    pipeline: IronManufacturingPipeline,
    synthetic_confirmation,
) -> None:
    # Two distinct industrial heats with identical feature values but different heat_ids
    df = generate_synthetic_eaf_dataset(n_rows=4, seed=11, include_target=True)
    df.loc[1, df.columns.difference(["heat_id"])] = df.loc[0, df.columns.difference(["heat_id"])]
    df.loc[1, "heat_id"] = "H-DISTINCT-002"

    ctx1, contract, art1, _ = _make_engineered_artifact(
        pipeline, FACTORY_A_ID, df, synthetic_confirmation
    )
    from taqamind.domains.iron_manufacturing.adapters.processing import _read_artifact_df

    eng_df1 = _read_artifact_df(pipeline.dataset_storage, FACTORY_A_ID, art1)
    lin1 = register_dataset_lineage(
        context=ctx1,
        artifact=art1,
        df=eng_df1,
        contract=contract,
        lineage_registry=pipeline.lineage_registry,
    )
    assert lin1.new_records == 4
    assert lin1.manifest.unique_record_count == 4
    assert lin1.manifest.memberships[0].record_id != lin1.manifest.memberships[1].record_id

    # Re-upload exact same records -> EXACT_MATCH
    ctx2, _, art2, _ = _make_engineered_artifact(
        pipeline, FACTORY_A_ID, df, synthetic_confirmation
    )
    eng_df2 = _read_artifact_df(pipeline.dataset_storage, FACTORY_A_ID, art2)
    lin2 = register_dataset_lineage(
        context=ctx2,
        artifact=art2,
        df=eng_df2,
        contract=contract,
        lineage_registry=pipeline.lineage_registry,
    )
    assert lin2.exact_matches == 4
    assert lin2.new_records == 0
    assert lin2.manifest.memberships[0].record_id == lin1.manifest.memberships[0].record_id

    # Correct row 0 target value without confirmation -> rejected
    df_corr = df.copy()
    df_corr.loc[0, "metallic_waste_loss_t"] = float(df_corr.loc[0, "metallic_waste_loss_t"]) + 0.5
    ctx3, _, art3, _ = _make_engineered_artifact(
        pipeline, FACTORY_A_ID, df_corr, synthetic_confirmation
    )
    eng_df3 = _read_artifact_df(pipeline.dataset_storage, FACTORY_A_ID, art3)
    with pytest.raises(ValueError, match="explicit correction confirmation"):
        register_dataset_lineage(
            context=ctx3,
            artifact=art3,
            df=eng_df3,
            contract=contract,
            lineage_registry=pipeline.lineage_registry,
            allow_confirmed_corrections=False,
        )

    # Confirm correction -> retains original record_id and advances version_number to 2
    lin3 = register_dataset_lineage(
        context=ctx3,
        artifact=art3,
        df=eng_df3,
        contract=contract,
        lineage_registry=pipeline.lineage_registry,
        allow_confirmed_corrections=True,
        correction_confirmed_by="qa_engineer",
        correction_reason="Weighbridge tare recalibration",
    )
    assert lin3.corrected_records == 1
    assert lin3.exact_matches == 3
    assert lin3.manifest.memberships[0].record_id == lin1.manifest.memberships[0].record_id
    assert (
        lin3.manifest.memberships[0].record_version_id
        != lin1.manifest.memberships[0].record_version_id
    )
    versions = pipeline.lineage_registry.get_versions(
        scope=build_record_scope(ctx3),
        record_id=lin1.manifest.memberships[0].record_id,
    )
    assert [v.version_number for v in versions] == [1, 2]


def test_missing_business_key_is_never_matched_as_confirmed_new(
    pipeline: IronManufacturingPipeline,
    synthetic_confirmation,
) -> None:
    df = generate_synthetic_eaf_dataset(n_rows=3, seed=12, include_target=True)
    ctx, contract, art, _ = _make_engineered_artifact(
        pipeline, FACTORY_A_ID, df, synthetic_confirmation
    )
    scope = build_record_scope(ctx)
    match = pipeline.lineage_registry.match(
        scope=scope,
        business_key_sha256=None,
        content_sha256="a" * 64,
        content_fingerprint_version="1.0.0",
    )
    assert match.status == RecordMatchStatus.UNVERIFIED
    assert match.requires_review


def test_cross_factory_isolation_enforced(
    pipeline: IronManufacturingPipeline,
    eaf_train_csv: Path,
    eaf_predict_csv: Path,
    synthetic_confirmation,
) -> None:
    outcome = pipeline.train_initial(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=eaf_train_csv,
        confirmation=synthetic_confirmation,
    )
    assert outcome.status == "succeeded"
    assert outcome.active_model_entry is not None

    # Factory B has NO active model and cannot access Factory A's dataset or model
    pred_b = pipeline.predict(
        factory_id=FACTORY_B_ID,
        process_id="eaf_meltshop",
        csv_path=eaf_predict_csv,
        confirmation=synthetic_confirmation,
    )
    assert pred_b.status == "blocked"
    assert "No active model" in (pred_b.blocked_reason or "")

    assert pipeline.get_model_info(factory_id=FACTORY_B_ID, process_id="eaf_meltshop")[
        "active_model"
    ] is None

    with pytest.raises(RegistryNotFoundError):
        pipeline.dataset_registry.get(
            factory_id=FACTORY_B_ID,
            dataset_revision_id=outcome.split_manifest.dataset_revision_id,  # type: ignore[union-attr]
        )
