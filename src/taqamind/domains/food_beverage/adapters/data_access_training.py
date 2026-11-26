"""P1.7: Data access providers, training/evaluation registries, AutoML trainer, and evaluator."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from taqamind.core.contracts import DatasetContract
from taqamind.core.interfaces import (
    CandidateConfiguration,
    CandidateModelReference,
    CandidateResult,
    CandidateStatus,
    ExecutionError,
    FinalEvaluationResult,
    FrozenModelSelection,
    ModelFamily,
    RegressionMetric,
    SplitManifest,
    SplitRole,
    StageTiming,
    TestMetricResult,
    TrainingAuthorization,
    TrainingConfiguration,
    TrainingRunResult,
    ValidationMetric,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.storage import ModelArtifactStorage
from taqamind.core.preprocessing import TabularPreprocessor
from taqamind.domains.food_beverage.adapters._store import (
    atomic_write_model,
    load_model_json,
)

ALLOWLISTED_PARAM_KEYS: dict[ModelFamily, set[str]] = {
    ModelFamily.DUMMY_REGRESSOR: {"strategy"},
    ModelFamily.RIDGE: {"alpha"},
    ModelFamily.ELASTIC_NET: {"alpha", "l1_ratio", "max_iter"},
    ModelFamily.RANDOM_FOREST: {"n_estimators", "max_depth", "min_samples_leaf"},
    ModelFamily.HIST_GRADIENT_BOOSTING: {"max_iter", "max_depth", "learning_rate"},
    ModelFamily.XGBOOST: {"n_estimators", "max_depth", "learning_rate", "subsample"},
}


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: tuple[RegressionMetric, ...],
) -> list[tuple[RegressionMetric, float | None, str | None]]:
    """Compute regression metrics with explicit undefined-denominator/variance handling."""

    results: list[tuple[RegressionMetric, float | None, str | None]] = []
    for m in metrics:
        if m == RegressionMetric.MAE:
            results.append((m, float(mean_absolute_error(y_true, y_pred)), None))
        elif m == RegressionMetric.RMSE:
            results.append((m, float(root_mean_squared_error(y_true, y_pred)), None))
        elif m == RegressionMetric.WAPE:
            denom = float(np.sum(np.abs(y_true)))
            if denom <= 0.0:
                results.append((m, None, "Zero denominator in WAPE."))
            else:
                results.append((m, float(np.sum(np.abs(y_true - y_pred)) / denom), None))
        elif m == RegressionMetric.R2:
            if float(np.var(y_true)) == 0.0:
                results.append((m, None, "Undefined R2 due to zero target variance."))
            else:
                val = float(r2_score(y_true, y_pred))
                results.append(
                    (
                        m,
                        val if math.isfinite(val) else None,
                        None if math.isfinite(val) else "Non-finite R2.",
                    )
                )
    return results


def build_estimator(candidate: CandidateConfiguration, seed: int) -> Any:
    """Instantiate an allowlisted regression estimator."""

    params = candidate.parameter_map
    allowed = ALLOWLISTED_PARAM_KEYS.get(candidate.family, set())
    unknown = set(params.keys()) - allowed
    if unknown:
        raise ValueError(f"Disallowed hyperparameters for {candidate.family}: {sorted(unknown)}")

    if candidate.family == ModelFamily.DUMMY_REGRESSOR:
        return DummyRegressor(strategy=str(params.get("strategy", "median")))
    if candidate.family == ModelFamily.RIDGE:
        return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=seed)
    if candidate.family == ModelFamily.ELASTIC_NET:
        return ElasticNet(
            alpha=float(params.get("alpha", 0.1)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            max_iter=int(params.get("max_iter", 2000)),
            random_state=seed,
        )
    if candidate.family == ModelFamily.RANDOM_FOREST:
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else 8,
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            random_state=seed,
            n_jobs=1,
        )
    if candidate.family == ModelFamily.HIST_GRADIENT_BOOSTING:
        return HistGradientBoostingRegressor(
            max_iter=int(params.get("max_iter", 100)),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else 6,
            learning_rate=float(params.get("learning_rate", 0.08)),
            random_state=seed,
        )
    if candidate.family == ModelFamily.XGBOOST:
        return XGBRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            subsample=float(params.get("subsample", 0.9)),
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(f"Unsupported model family: {candidate.family}")


class LocalDevelopmentDataProvider:
    """Provides strictly TRAIN and VALIDATION partitions (never TEST).

    Candidate for migration to ``taqamind.core.data_access``.
    """

    def __init__(self, partitions: dict[SplitRole, pd.DataFrame]) -> None:
        self._partitions = partitions

    def get_train(self) -> pd.DataFrame:
        return self._partitions[SplitRole.TRAIN].copy()

    def get_validation(self) -> pd.DataFrame:
        return self._partitions[SplitRole.VALIDATION].copy()


class LocalProtectedTestDataProvider:
    """Single-use protected TEST partition provider gated by FrozenModelSelection.

    Candidate for migration to ``taqamind.core.data_access``.
    """

    def __init__(self, test_df: pd.DataFrame) -> None:
        self._test_df = test_df
        self._consumed_splits: set[UUID] = set()

    def release_test(self, *, split_id: UUID, selection: FrozenModelSelection) -> pd.DataFrame:
        _ = selection
        if split_id in self._consumed_splits:
            raise ValueError(f"Protected test partition for split {split_id} already consumed.")
        self._consumed_splits.add(split_id)
        return self._test_df.copy()


class LocalTrainingRegistry:
    """Local JSON registry for TrainingRunResults.

    Candidate for migration to ``taqamind.core.training_registry``.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve() / "training_runs"
        self._root.mkdir(parents=True, exist_ok=True)

    def register(self, run: TrainingRunResult) -> TrainingRunResult:
        fid = run.development_split.context.factory_id
        path = self._root / str(fid) / f"{run.training_run_id}.json"
        atomic_write_model(path, run)
        return run

    def get(self, *, factory_id: UUID, training_run_id: UUID) -> TrainingRunResult:
        path = self._root / str(factory_id) / f"{training_run_id}.json"
        return load_model_json(path, TrainingRunResult)


class LocalEvaluationRegistry:
    """Local JSON registry for FinalEvaluationResults.

    Candidate for migration to ``taqamind.core.evaluation``.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve() / "evaluations"
        self._root.mkdir(parents=True, exist_ok=True)

    def register(self, evaluation: FinalEvaluationResult) -> FinalEvaluationResult:
        fid = evaluation.selection.training_run.development_split.context.factory_id
        path = self._root / str(fid) / f"{evaluation.evaluation_id}.json"
        atomic_write_model(path, evaluation)
        return evaluation


class FoodBeverageControlledAutoMLTrainer:
    """Controlled regression trainer fitting Core TabularPreprocessor strictly on train_df.

    Candidate for migration to ``taqamind.core.automl``.
    """

    def __init__(self, model_storage: ModelArtifactStorage) -> None:
        self._model_storage = model_storage

    def train(
        self,
        *,
        authorization: TrainingAuthorization,
        configuration: TrainingConfiguration,
        contract: DatasetContract,
        dev_provider: LocalDevelopmentDataProvider,
    ) -> TrainingRunResult:
        train_df = dev_provider.get_train()
        validation_df = dev_provider.get_validation()
        target_col = contract.target.feature.name

        y_train = train_df[target_col].to_numpy(dtype=float)
        y_val = validation_df[target_col].to_numpy(dtype=float)

        all_metrics = (configuration.primary_metric, *configuration.secondary_metrics)
        results: list[CandidateResult] = []
        execution_id = authorization.development_split.context.execution_id
        factory_id = authorization.development_split.context.factory_id

        for candidate in configuration.candidates[: configuration.budget.max_trials]:
            t0 = datetime.now(UTC)
            try:
                # Fit Core TabularPreprocessor strictly on train_df inside each trial
                preprocessor = TabularPreprocessor(contract)
                X_train_t = preprocessor.fit_transform(train_df)
                X_val_t = preprocessor.transform(validation_df)

                estimator = build_estimator(candidate, configuration.random_seed)
                estimator.fit(X_train_t, y_train)
                y_pred_val = np.asarray(estimator.predict(X_val_t), dtype=float)

                metric_tuples = compute_regression_metrics(y_val, y_pred_val, all_metrics)
                val_metrics = tuple(
                    ValidationMetric(metric=m, value=v, undefined_reason=r)
                    for m, v, r in metric_tuples
                )

                artifact_id = uuid4()
                bundle = {
                    "preprocessor": preprocessor,
                    "estimator": estimator,
                    "feature_names": preprocessor.ordered_predictors,
                    "target_name": target_col,
                    "target_unit": contract.target.feature.unit,
                    "candidate_id": str(candidate.candidate_id),
                    "family": candidate.family.value,
                }
                buf = BytesIO()
                joblib.dump(bundle, buf, compress=3)
                raw_bytes = buf.getvalue()
                digest = calculate_bytes_sha256(raw_bytes)
                model_ref = CandidateModelReference(
                    artifact_id=artifact_id,
                    factory_id=factory_id,
                    execution_id=execution_id,
                    candidate_id=candidate.candidate_id,
                    content_sha256=digest,
                    size_bytes=len(raw_bytes),
                )
                self._model_storage.put(
                    factory_id=factory_id,
                    reference=model_ref,
                    stream=BytesIO(raw_bytes),
                )
                t1 = datetime.now(UTC)
                results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.SUCCEEDED,
                        validation_metrics=val_metrics,
                        evaluated_row_count=int(len(y_val)),
                        model_reference=model_ref,
                        timing=StageTiming(started_at=t0, finished_at=t1),
                    )
                )
            except Exception as exc:
                t1 = datetime.now(UTC)
                results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.FAILED,
                        timing=StageTiming(started_at=t0, finished_at=t1),
                        error=ExecutionError(code="CANDIDATE_TRAINING_FAILED", message=str(exc)),
                    )
                )

        return TrainingRunResult(
            development_split=authorization.development_split,
            configuration=configuration,
            results=tuple(results),
        )


class FoodBeverageProtectedFinalEvaluator:
    """Evaluate frozen baseline & challenger on protected TEST data without refitting.

    Candidate for migration to ``taqamind.core.evaluation``.
    """

    def __init__(self, model_storage: ModelArtifactStorage) -> None:
        self._model_storage = model_storage

    def evaluate(
        self,
        *,
        selection: FrozenModelSelection,
        manifest: SplitManifest,
        contract: DatasetContract,
        test_provider: LocalProtectedTestDataProvider,
    ) -> FinalEvaluationResult:
        started_at = datetime.now(UTC)
        test_df = test_provider.release_test(split_id=manifest.split_id, selection=selection)
        target_col = contract.target.feature.name
        y_test = test_df[target_col].to_numpy(dtype=float)

        run = selection.training_run
        factory_id = run.development_split.context.factory_id
        all_metrics = (run.configuration.primary_metric, *run.configuration.secondary_metrics)

        def eval_candidate(cand_id: UUID) -> TestMetricResult:
            res = next(r for r in run.results if r.candidate_id == cand_id)
            assert res.model_reference is not None
            with self._model_storage.open_verified(
                factory_id=factory_id,
                reference=res.model_reference,
            ) as stream:
                bundle = joblib.load(stream)
            X_test_t = bundle["preprocessor"].transform(test_df)
            y_pred = np.asarray(bundle["estimator"].predict(X_test_t), dtype=float)
            metric_tuples = compute_regression_metrics(y_test, y_pred, all_metrics)
            return TestMetricResult(
                candidate_id=cand_id,
                model_artifact_id=res.model_reference.artifact_id,
                record_ids=manifest.test.record_ids,
                metrics=tuple(
                    ValidationMetric(metric=m, value=v, undefined_reason=r)
                    for m, v, r in metric_tuples
                ),
            )

        baseline_test = eval_candidate(selection.baseline_candidate_id)
        selected_test = eval_candidate(selection.selected_candidate_id)
        finished_at = datetime.now(UTC)
        return FinalEvaluationResult(
            selection=selection,
            split_manifest=manifest,
            baseline_result=baseline_test,
            selected_result=selected_test,
            timing=StageTiming(started_at=started_at, finished_at=finished_at),
        )
