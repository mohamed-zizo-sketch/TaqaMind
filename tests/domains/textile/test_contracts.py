"""Tests for textile domain registration, contracts, unit mapping, and validation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from taqamind.core.contracts import (
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureOrigin,
    FeatureRole,
    MissingValuePolicy,
)
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.interfaces import (
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    ExecutionStatus,
    PipelineContext,
    calculate_bytes_sha256,
)
from taqamind.core.registry import DomainRegistry, LocalDatasetArtifactStorage
from taqamind.core.validation import ContractSemanticValidator
from taqamind.domains.textile.adapters.unit_mapper import UnitAwareColumnMapperAdapter
from taqamind.domains.textile.domain import TEXTILE_DOMAIN_ID, register_textile
from taqamind.domains.textile.knitting import (
    KNITTING_PROCESS_ID,
    build_knitting_dataset_contract,
    build_synthetic_knitting_confirmation,
    generate_synthetic_knitting_prediction_frame,
)
from taqamind.domains.textile.weaving import (
    WEAVING_POST_PRODUCTION_EXCLUDED_COLUMNS,
    WEAVING_PROCESS_ID,
    build_synthetic_weaving_confirmation,
    build_weaving_dataset_contract,
    generate_synthetic_weaving_prediction_frame,
    generate_synthetic_weaving_training_frame,
)


def test_textile_domain_registration_and_unconfirmed_default() -> None:
    registry = DomainRegistry()
    domain = register_textile(registry)
    assert domain.domain_id == TEXTILE_DOMAIN_ID
    assert {p.process_id for p in domain.processes} == {
        WEAVING_PROCESS_ID,
        KNITTING_PROCESS_ID,
    }
    for proc in domain.processes:
        for task in proc.tasks:
            assert not task.dataset_contract.target.is_confirmed


def test_post_production_predictor_rejected_by_contract() -> None:
    base = build_weaving_dataset_contract(confirmation=build_synthetic_weaving_confirmation())
    bad_feature = FeatureContract(
        name="act_shrink_pct",
        display_name="Actual Shrinkage",
        description="Post-production shrinkage.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    bad_contract = base.model_copy(update={"features": (*base.features, bad_feature)})
    with pytest.raises(ValueError):
        bad_contract.validate_prediction_features()
    assert "act_shrink%" in WEAVING_POST_PRODUCTION_EXCLUDED_COLUMNS


def test_unit_aware_mapper_converts_meters_to_yards_and_blocks_unknown(tmp_path: Path) -> None:
    storage = LocalDatasetArtifactStorage(tmp_path / "datasets")
    mapper = UnitAwareColumnMapperAdapter(storage)
    contract = build_weaving_dataset_contract(confirmation=build_synthetic_weaving_confirmation())
    factory_id = uuid4()

    df = generate_synthetic_weaving_training_frame(num_orders=4, batches_per_order=2)
    # Supply planned_beam_length_yds in meters (m) instead of yards (yd).
    df_meters = df.copy()
    df_meters["planned_beam_length_yds"] = 100.0
    raw_bytes = df_meters.to_csv(index=False).encode("utf-8")
    ref = ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=calculate_bytes_sha256(raw_bytes),
        size_bytes=len(raw_bytes),
    )
    storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(raw_bytes))

    rev_id = uuid4()
    raw_meta = DatasetMetadata(
        revision_id=rev_id,
        root_revision_id=rev_id,
        factory_id=factory_id,
        domain="textile",
        process="weaving",
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_id="weaving_waste_regression",
        task_version="1.0.0",
        target_version=contract.target.version,
        source_file_sha256=ref.content_sha256,
        state=DatasetState.RAW,
        row_count=None,
        columns=(),
    )
    context = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=raw_meta.revision_id,
        domain="textile",
        process="weaving",
        task_id="weaving_waste_regression",
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_version="1.0.0",
        target_version=contract.target.version,
        purpose=DatasetPurpose.TRAINING,
    )
    raw_artifact = DatasetArtifact(metadata=raw_meta, reference=ref)
    ingested = CsvDataIngestion(storage).ingest(
        context=context, artifact=raw_artifact, contract=contract
    )
    assert ingested.output_artifact is not None
    artifact = ingested.output_artifact
    proposal = mapper.propose_mapping(context=context, artifact=artifact, contract=contract)

    # Override source_unit to 'm' for planned_beam_length_yds.
    entries = tuple(
        e.model_copy(update={"source_unit": "m"})
        if e.canonical_column == "planned_beam_length_yds"
        else e
        for e in proposal.entries
    )
    mod_proposal = proposal.model_copy(update={"entries": entries})
    approved = ApprovedMapping(
        proposal=mod_proposal,
        approved_by="test",
        confirmed_source_columns=tuple(e.source_column for e in entries),
    )
    res = mapper.apply_mapping(
        context=context, artifact=artifact, contract=contract, mapping=approved
    )
    assert res.status == ExecutionStatus.SUCCEEDED
    assert res.output_artifact is not None
    with storage.open_verified(
        factory_id=factory_id, reference=res.output_artifact.reference
    ) as stream:
        mapped_df = pd.read_csv(stream)
    assert pytest.approx(float(mapped_df["planned_beam_length_yds"].iloc[0]), rel=1e-4) == 109.36133

    # Unsupported unit conversion fails cleanly.
    bad_entries = tuple(
        e.model_copy(update={"source_unit": "kg"})
        if e.canonical_column == "planned_beam_length_yds"
        else e
        for e in proposal.entries
    )
    bad_approved = ApprovedMapping(
        proposal=proposal.model_copy(update={"entries": bad_entries}),
        approved_by="test",
        confirmed_source_columns=tuple(e.source_column for e in bad_entries),
    )
    bad_res = mapper.apply_mapping(
        context=context, artifact=artifact, contract=contract, mapping=bad_approved
    )
    assert bad_res.status == ExecutionStatus.FAILED


def test_prediction_frame_succeeds_without_target_while_training_requires_target(
    tmp_path: Path,
) -> None:
    storage = LocalDatasetArtifactStorage(tmp_path / "datasets")
    validator = ContractSemanticValidator(storage)
    contract = build_knitting_dataset_contract(confirmation=build_synthetic_knitting_confirmation())
    factory_id = uuid4()

    pred_df = generate_synthetic_knitting_prediction_frame(num_runs=4)
    raw_bytes = pred_df.to_csv(index=False).encode("utf-8")
    ref = ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=calculate_bytes_sha256(raw_bytes),
        size_bytes=len(raw_bytes),
    )
    storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(raw_bytes))
    rev_id2 = uuid4()
    raw_meta = DatasetMetadata(
        revision_id=rev_id2,
        root_revision_id=rev_id2,
        factory_id=factory_id,
        domain="textile",
        process="knitting",
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_id="knitting_waste_regression",
        task_version="1.0.0",
        target_version=contract.target.version,
        source_file_sha256=ref.content_sha256,
        state=DatasetState.RAW,
        row_count=None,
        columns=(),
    )
    raw_artifact = DatasetArtifact(metadata=raw_meta, reference=ref)
    pred_ctx = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=raw_meta.revision_id,
        domain="textile",
        process="knitting",
        task_id="knitting_waste_regression",
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_version="1.0.0",
        target_version=contract.target.version,
        purpose=DatasetPurpose.PREDICTION,
        model_version_id=uuid4(),
    )
    ingested = CsvDataIngestion(storage).ingest(
        context=pred_ctx, artifact=raw_artifact, contract=contract
    )
    assert ingested.output_artifact is not None
    proposal = UnitAwareColumnMapperAdapter(storage).propose_mapping(
        context=pred_ctx, artifact=ingested.output_artifact, contract=contract
    )
    mapped = UnitAwareColumnMapperAdapter(storage).apply_mapping(
        context=pred_ctx,
        artifact=ingested.output_artifact,
        contract=contract,
        mapping=ApprovedMapping(
            proposal=proposal,
            approved_by="test",
            confirmed_source_columns=tuple(e.source_column for e in proposal.entries),
        ),
    )
    assert mapped.output_artifact is not None
    artifact = mapped.output_artifact
    pred_res = validator.validate(context=pred_ctx, artifact=artifact, contract=contract)
    assert pred_res.status == ExecutionStatus.SUCCEEDED

    train_ctx = pred_ctx.model_copy(
        update={"purpose": DatasetPurpose.TRAINING, "model_version_id": None}
    )
    train_res = validator.validate(context=train_ctx, artifact=artifact, contract=contract)
    assert train_res.status in {ExecutionStatus.FAILED, ExecutionStatus.BLOCKED}


def test_weaving_prediction_frame_has_no_target() -> None:
    df = generate_synthetic_weaving_prediction_frame(num_batches=3)
    assert "waste_length_yds" not in df.columns
    assert len(df) == 3
