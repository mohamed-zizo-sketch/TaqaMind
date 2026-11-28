"""WP04 regression tests — Splitting and Leakage Prevention hardening."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
import pytest

from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetContract,
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureOrigin,
    FeatureRole,
    MissingValuePolicy,
    PipelineStage,
    PredictionContext,
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMetadata,
    DatasetState,
    LeakageCheck,
    LeakageCheckStatus,
    PipelineContext,
    RecordScope,
    SplitConfiguration,
    SplitManifest,
    SplitPartition,
    SplitRole,
    SplitStrategy,
    TransformationRecord,
)
from taqamind.core.lineage import LocalRecordLineageRegistry, RecordMembershipBuilder
from taqamind.core.readiness import CoreTrainingReadinessChecker
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.core.splitting import (
    CoreDatasetSplitter,
    CoreLeakageValidator,
    LocalRecordUsageRegistry,
)


def _build_contract() -> DatasetContract:
    return DatasetContract(
        dataset_id="wp04_ds",
        version="1.0.0",
        domain="textile",
        process="weaving",
        description="WP04 splitting contract",
        granularity="batch",
        features=(
            FeatureContract(
                name="record_key",
                display_name="Record Key",
                description="Key",
                data_type=DataType.STRING,
                role=FeatureRole.IDENTIFIER,
                availability=FeatureAvailability.PRE_PRODUCTION,
                origin=FeatureOrigin.RECORDED,
                missing_policy=MissingValuePolicy.REJECT,
            ),
            FeatureContract(
                name="lot_id",
                display_name="Lot ID",
                description="Group",
                data_type=DataType.STRING,
                role=FeatureRole.METADATA,
                availability=FeatureAvailability.PRE_PRODUCTION,
                origin=FeatureOrigin.RECORDED,
                missing_policy=MissingValuePolicy.REJECT,
            ),
            FeatureContract(
                name="run_ts",
                display_name="Run TS",
                description="Timestamp",
                data_type=DataType.DATETIME,
                role=FeatureRole.TIMESTAMP,
                availability=FeatureAvailability.PRE_PRODUCTION,
                origin=FeatureOrigin.RECORDED,
                missing_policy=MissingValuePolicy.REJECT,
            ),
            FeatureContract(
                name="qty",
                display_name="Qty",
                description="Quantity",
                data_type=DataType.FLOAT,
                role=FeatureRole.PREDICTOR,
                availability=FeatureAvailability.PRE_PRODUCTION,
                origin=FeatureOrigin.RECORDED,
                missing_policy=MissingValuePolicy.REJECT,
            ),
        ),
        target=TargetContract(
            target_id="waste_target",
            feature=FeatureContract(
                name="waste",
                display_name="Waste",
                description="Waste",
                data_type=DataType.FLOAT,
                role=FeatureRole.TARGET,
                availability=FeatureAvailability.POST_PRODUCTION,
                origin=FeatureOrigin.MEASURED,
                unit="kg",
                constraints=FeatureConstraints(minimum=0.0),
                missing_policy=MissingValuePolicy.REJECT,
            ),
            task_type=TaskType.REGRESSION,
            target_kind=TargetKind.QUANTITY,
            definition=TargetDefinition(definition="Waste", granularity="batch"),
            confirmation=TargetConfirmation(
                status=ConfirmationStatus.CONFIRMED,
                confirmed_by="wp04_test",
                confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ),
        prediction_context=PredictionContext(description="Batch pred"),
        business_keys=("record_key",),
        grouping_columns=("lot_id",),
        time_column="run_ts",
    )


def _store_engineered_and_readiness(
    tmp_path: Path,
    df: pd.DataFrame,
    contract: DatasetContract,
):
    storage = LocalDatasetArtifactStorage(tmp_path / "blobs")
    lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
    usage_reg = LocalRecordUsageRegistry(tmp_path / "reg")

    factory_id = uuid4()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    digest = sha256(csv_bytes).hexdigest()
    ref = ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=digest,
        size_bytes=len(csv_bytes),
    )
    storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(csv_bytes))

    raw_id = uuid4()
    eng_id = uuid4()
    eng_art = DatasetArtifact(
        metadata=DatasetMetadata(
            revision_id=eng_id,
            root_revision_id=raw_id,
            factory_id=factory_id,
            domain=contract.domain,
            process=contract.process,
            dataset_id=contract.dataset_id,
            contract_version=contract.version,
            task_id=contract.target.target_id,
            task_version="1.0.0",
            target_version=contract.target.version,
            source_file_sha256=digest,
            state=DatasetState.ENGINEERED,
            row_count=len(df),
            columns=tuple(
                DatasetColumnSchema(name=str(col), physical_dtype=str(df[col].dtype))
                for col in df.columns
            ),
            transformation=TransformationRecord(
                execution_id=uuid4(),
                parent_revision_id=raw_id,
                stage=PipelineStage.FEATURE_ENGINEERING,
                operation="feature_engineering",
                implementation_version="1.0.0",
                description="Engineered",
            ),
        ),
        reference=ref,
    )

    scope = RecordScope(
        factory_id=factory_id,
        domain=contract.domain,
        process=contract.process,
        dataset_id=contract.dataset_id,
        identity_namespace="production_record",
    )
    builder = RecordMembershipBuilder(storage, lin_reg)
    summary = builder.build_and_register(scope=scope, artifact=eng_art, contract=contract)

    ctx = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=eng_id,
        domain=contract.domain,
        process=contract.process,
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_id=contract.target.target_id,
        task_version="1.0.0",
        target_version=contract.target.version,
        purpose=DatasetPurpose.TRAINING,
    )
    readiness = CoreTrainingReadinessChecker(storage, lin_reg).assess(
        context=ctx, artifact=eng_art, contract=contract
    )
    return storage, lin_reg, usage_reg, summary, readiness


class TestWP04SplittingAndLeakageHardening:
    """WP04: Temporal NaT, null group keys, and group-temporal ordering guarantees."""

    def test_temporal_split_rejects_missing_timestamps(self, tmp_path: Path):
        contract = _build_contract()
        rows = [
            {
                "record_key": f"R{i:02d}",
                "lot_id": f"L{i}",
                "run_ts": f"2026-01-{i + 1:02d}T00:00:00+00:00",
                "qty": 10.0 + i,
                "waste": 1.0 + i,
            }
            for i in range(15)
        ]
        df_valid = pd.DataFrame(rows)
        storage, lin_reg, _, _, readiness = _store_engineered_and_readiness(
            tmp_path, df_valid, contract
        )
        df_bad = df_valid.copy()
        df_bad.loc[5, "run_ts"] = None
        splitter = CoreDatasetSplitter(storage, lin_reg)
        cfg = SplitConfiguration(
            strategy=SplitStrategy.TEMPORAL,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
            justification="Temporal split",
        )
        with (
            patch(
                "taqamind.core.splitting.dataset_splitter.read_verified_csv",
                return_value=df_bad,
            ),
            pytest.raises(ValueError, match="missing or unparseable timestamps"),
        ):
            splitter.split(readiness=readiness, configuration=cfg)

    def test_group_split_rejects_missing_group_keys(self, tmp_path: Path):
        contract = _build_contract()
        rows = [
            {
                "record_key": f"R{i:02d}",
                "lot_id": f"L{i}",
                "run_ts": f"2026-01-{i + 1:02d}T00:00:00+00:00",
                "qty": 10.0 + i,
                "waste": 1.0 + i,
            }
            for i in range(15)
        ]
        df_valid = pd.DataFrame(rows)
        storage, lin_reg, _, _, readiness = _store_engineered_and_readiness(
            tmp_path, df_valid, contract
        )
        df_bad = df_valid.copy()
        df_bad.loc[4, "lot_id"] = None
        splitter = CoreDatasetSplitter(storage, lin_reg)
        cfg = SplitConfiguration(
            strategy=SplitStrategy.GROUP,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
            justification="Group split",
        )
        with (
            patch(
                "taqamind.core.splitting.dataset_splitter.read_verified_csv",
                return_value=df_bad,
            ),
            pytest.raises(ValueError, match="missing values"),
        ):
            splitter.split(readiness=readiness, configuration=cfg)

    def test_leakage_validator_fails_temporal_order_when_nat_present(self, tmp_path: Path):
        contract = _build_contract()
        rows = [
            {
                "record_key": f"R{i:02d}",
                "lot_id": f"L{i}",
                "run_ts": f"2026-01-{i + 1:02d}T00:00:00+00:00",
                "qty": 10.0 + i,
                "waste": 1.0 + i,
            }
            for i in range(15)
        ]
        df_valid = pd.DataFrame(rows)
        storage, lin_reg, usage_reg, summary, readiness = _store_engineered_and_readiness(
            tmp_path, df_valid, contract
        )
        df_bad = df_valid.copy()
        df_bad.loc[0, "run_ts"] = None
        rids = [m.record_id for m in summary.manifest.memberships]
        manifest = SplitManifest(
            context=readiness.result.context,
            dataset_revision_id=readiness.result.input_artifact.metadata.revision_id,
            configuration=SplitConfiguration(
                strategy=SplitStrategy.TEMPORAL,
                train_ratio=0.6,
                validation_ratio=0.2,
                test_ratio=0.2,
                justification="Temporal split with NaT",
            ),
            train=SplitPartition(role=SplitRole.TRAIN, record_ids=tuple(rids[:9])),
            validation=SplitPartition(role=SplitRole.VALIDATION, record_ids=tuple(rids[9:12])),
            test=SplitPartition(role=SplitRole.TEST, record_ids=tuple(rids[12:])),
        )
        validator = CoreLeakageValidator(storage, lin_reg, usage_reg)
        with patch(
            "taqamind.core.splitting.leakage_validator.read_verified_csv",
            return_value=df_bad,
        ):
            result = validator.validate(readiness=readiness, manifest=manifest)
        temporal_check = next(c for c in result.checks if c.check == LeakageCheck.TEMPORAL_ORDER)
        assert temporal_check.status == LeakageCheckStatus.FAILED

    def test_leakage_validator_fails_group_overlap_when_null_group_present(self, tmp_path: Path):
        contract = _build_contract()
        rows = [
            {
                "record_key": f"R{i:02d}",
                "lot_id": f"L{i}",
                "run_ts": f"2026-01-{i + 1:02d}T00:00:00+00:00",
                "qty": 10.0 + i,
                "waste": 1.0 + i,
            }
            for i in range(15)
        ]
        df_valid = pd.DataFrame(rows)
        storage, lin_reg, usage_reg, summary, readiness = _store_engineered_and_readiness(
            tmp_path, df_valid, contract
        )
        df_bad = df_valid.copy()
        df_bad.loc[0, "lot_id"] = None
        rids = [m.record_id for m in summary.manifest.memberships]
        manifest = SplitManifest(
            context=readiness.result.context,
            dataset_revision_id=readiness.result.input_artifact.metadata.revision_id,
            configuration=SplitConfiguration(
                strategy=SplitStrategy.GROUP,
                train_ratio=0.6,
                validation_ratio=0.2,
                test_ratio=0.2,
                justification="Group split with null",
            ),
            train=SplitPartition(role=SplitRole.TRAIN, record_ids=tuple(rids[:9])),
            validation=SplitPartition(role=SplitRole.VALIDATION, record_ids=tuple(rids[9:12])),
            test=SplitPartition(role=SplitRole.TEST, record_ids=tuple(rids[12:])),
        )
        validator = CoreLeakageValidator(storage, lin_reg, usage_reg)
        with patch(
            "taqamind.core.splitting.leakage_validator.read_verified_csv",
            return_value=df_bad,
        ):
            result = validator.validate(readiness=readiness, manifest=manifest)
        group_check = next(c for c in result.checks if c.check == LeakageCheck.GROUP_OVERLAP)
        assert group_check.status == LeakageCheckStatus.FAILED

    def test_group_temporal_split_passes_leakage_validation(self, tmp_path: Path):
        contract = _build_contract()
        rows = [
            {
                "record_key": f"R{i:02d}",
                "lot_id": f"LOT_{i // 3}",
                "run_ts": f"2026-01-{i + 1:02d}T00:00:00+00:00",
                "qty": 10.0 + i,
                "waste": 1.0 + i,
            }
            for i in range(15)
        ]
        storage, lin_reg, usage_reg, _, readiness = _store_engineered_and_readiness(
            tmp_path, pd.DataFrame(rows), contract
        )
        splitter = CoreDatasetSplitter(storage, lin_reg)
        cfg = SplitConfiguration(
            strategy=SplitStrategy.GROUP_TEMPORAL,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
            justification="Disjoint group-temporal split",
        )
        manifest = splitter.split(readiness=readiness, configuration=cfg)
        validator = CoreLeakageValidator(storage, lin_reg, usage_reg)
        result = validator.validate(readiness=readiness, manifest=manifest)
        assert not result.report.has_blocking_issues
        result.require_training_eligibility()
