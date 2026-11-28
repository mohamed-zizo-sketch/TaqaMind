"""Readiness, splitting, leakage prevention, train-only fit, and evaluation control tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from taqamind.core.interfaces import (
    AcceptancePolicy,
    LeakageCheck,
    LeakageCheckStatus,
    ModelStatus,
    ModelTransitionAction,
    ModelTransitionRequest,
    RecordUsage,
    RecordUsageRole,
    SplitConfiguration,
    SplitManifest,
    SplitPartition,
    SplitRole,
    SplitStrategy,
    validate_model_transition,
)
from taqamind.domains.iron_manufacturing import IronManufacturingPipeline
from taqamind.domains.iron_manufacturing.common.synthetic import (
    generate_synthetic_eaf_dataset,
)
from taqamind.domains.iron_manufacturing.services.lineage_service import (
    build_record_scope,
)
from tests.domains.iron_manufacturing.conftest import FACTORY_A_ID


def test_training_blocked_when_target_unconfirmed_or_insufficient_rows(
    pipeline: IronManufacturingPipeline,
    eaf_train_csv: Path,
    tmp_path: Path,
    synthetic_confirmation,
) -> None:
    # 1. Default unconfirmed target -> BLOCKED before splitting
    unconfirmed_out = pipeline.train_initial(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=eaf_train_csv,
        confirmation=None,
    )
    assert unconfirmed_out.status == "blocked"
    assert unconfirmed_out.readiness_result is not None
    assert not unconfirmed_out.readiness_result.eligible_for_split

    # 2. Insufficient rows (< Core min_rows=12) -> BLOCKED
    tiny_df = generate_synthetic_eaf_dataset(
        n_rows=6, seed=7, include_target=True, start_index=5001
    )
    tiny_csv = tmp_path / "tiny.csv"
    tiny_df.to_csv(tiny_csv, index=False)

    tiny_out = pipeline.train_initial(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=tiny_csv,
        confirmation=synthetic_confirmation,
    )
    assert tiny_out.status == "blocked"
    assert tiny_out.readiness_result is not None
    assert any(
        i.code in {"INSUFFICIENT_ROW_COUNT", "INSUFFICIENT_OBSERVATIONS"}
        for i in tiny_out.readiness_result.result.report.issues  # type: ignore[union-attr]
    )


def test_group_leakage_temporal_leakage_and_historical_exposure_rejected(
    pipeline: IronManufacturingPipeline,
    eaf_train_csv: Path,
    synthetic_confirmation,
) -> None:
    # Run initial training to obtain valid readiness + registered lineage
    outcome = pipeline.train_initial(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=eaf_train_csv,
        confirmation=synthetic_confirmation,
    )
    assert outcome.status == "succeeded"
    assert outcome.readiness_result is not None
    assert outcome.split_manifest is not None
    assert outcome.lineage_summary is not None

    readiness = outcome.readiness_result
    manifest = outcome.split_manifest

    # 1. Re-validating the SAME split manifest now FAILS HISTORICAL_USAGE because test was exposed
    leak_hist = pipeline.leakage_validator.validate(readiness=readiness, manifest=manifest)
    assert not leak_hist.eligible_for_training
    hist_check = next(c for c in leak_hist.checks if c.check == LeakageCheck.HISTORICAL_USAGE)
    assert hist_check.status == LeakageCheckStatus.FAILED

    # 2. Construct a group-leaking split (swap one record between train and test sharing group)
    all_ids = [m.record_id for m in outcome.lineage_summary.manifest.memberships]
    # Interleave record IDs so production_order_id overlaps across train/val/test
    tr_leak = tuple(all_ids[0::3])
    va_leak = tuple(all_ids[1::3])
    te_leak = tuple(all_ids[2::3])
    bad_manifest = SplitManifest(
        context=manifest.context,
        dataset_revision_id=manifest.dataset_revision_id,
        configuration=SplitConfiguration(
            strategy=SplitStrategy.GROUP_TEMPORAL,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            justification="Intentional group and temporal leakage test manifest.",
        ),
        train=SplitPartition(role=SplitRole.TRAIN, record_ids=tr_leak),
        validation=SplitPartition(role=SplitRole.VALIDATION, record_ids=va_leak),
        test=SplitPartition(role=SplitRole.TEST, record_ids=te_leak),
    )
    leak_bad = pipeline.leakage_validator.validate(readiness=readiness, manifest=bad_manifest)
    assert not leak_bad.eligible_for_training
    check_map = {c.check: c.status for c in leak_bad.checks}
    assert check_map[LeakageCheck.GROUP_OVERLAP] == LeakageCheckStatus.FAILED
    assert check_map[LeakageCheck.TEMPORAL_ORDER] == LeakageCheckStatus.FAILED


def test_corrected_record_version_still_detected_as_historically_exposed(
    pipeline: IronManufacturingPipeline,
    eaf_train_csv: Path,
    synthetic_confirmation,
) -> None:
    outcome = pipeline.train_initial(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=eaf_train_csv,
        confirmation=synthetic_confirmation,
    )
    assert outcome.readiness_result is not None
    assert outcome.split_manifest is not None

    scope = build_record_scope(outcome.split_manifest.context)
    exposed_id = outcome.split_manifest.test.record_ids[0]
    history = pipeline.usage_registry.get_history(scope=scope, record_ids=(exposed_id,))
    assert any(u.role == RecordUsageRole.PROTECTED_TEST for u in history)
    assert all(isinstance(u, RecordUsage) for u in history)


def test_rejected_model_cannot_be_activated(
    pipeline: IronManufacturingPipeline,
    eaf_train_csv: Path,
    synthetic_confirmation,
) -> None:
    # Impossible acceptance policy (99.9% improvement required) -> honest REJECTION
    strict_policy = AcceptancePolicy(
        minimum_improvement_pct=99.9,
        justification="Strict acceptance gate test verifying honest rejection.",
        created_at=datetime.now(UTC) - timedelta(seconds=5),
    )
    outcome = pipeline.train_initial(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=eaf_train_csv,
        confirmation=synthetic_confirmation,
        acceptance_policy=strict_policy,
    )
    assert outcome.status == "rejected"
    assert outcome.model_entry is not None
    assert outcome.model_entry.status == ModelStatus.REJECTED
    assert outcome.active_model_entry is None

    # Attempting to activate a REJECTED model must raise ValueError
    req = ModelTransitionRequest(
        scope=outcome.model_entry.version.scope,
        action=ModelTransitionAction.FIRST_ACTIVATION,
        target_model_version_id=outcome.model_entry.version.model_version_id,
        requested_by="tester",
        reason="Should fail because model is rejected.",
    )
    with pytest.raises(ValueError, match="invalid lifecycle status"):
        validate_model_transition(request=req, target=outcome.model_entry, active=None)
    with pytest.raises(ValueError, match="invalid lifecycle status"):
        pipeline.model_registry.transition(request=req)


def test_train_only_preprocessing_and_hurdle_model_on_zero_inflated_scrap(
    pipeline: IronManufacturingPipeline,
    scrap_yard_train_csv: Path,
    synthetic_confirmation,
) -> None:
    outcome = pipeline.train_initial(
        factory_id=FACTORY_A_ID,
        process_id="scrap_yard_reception",
        csv_path=scrap_yard_train_csv,
        confirmation=synthetic_confirmation,
    )
    assert outcome.status == "succeeded"
    assert outcome.training_run is not None
    assert outcome.active_model_entry is not None

    # Verify hurdle regression candidate was automatically included for zero-inflated data
    families = {c.family.value for c in outcome.training_run.configuration.candidates}
    assert "hurdle_regression" in families
    assert "dummy_regressor" in families
    assert "historical_rate" in families

    # Development split must never expose protected test records
    dev = outcome.training_run.development_split
    test_ids = set(outcome.split_manifest.test.record_ids)  # type: ignore[union-attr]
    assert not (set(dev.train.record_ids) & test_ids)
    assert not (set(dev.validation.record_ids) & test_ids)
    assert not hasattr(dev, "test")

    # Verify saved scaler/imputer statistics come ONLY from train rows, not full dataset
    assert outcome.final_evaluation is not None
    assert outcome.final_evaluation.selected_mae < outcome.final_evaluation.baseline_mae

    df_full = pd.read_csv(scrap_yard_train_csv)
    assert len(df_full) == 90
