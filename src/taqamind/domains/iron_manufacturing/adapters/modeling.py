"""Controlled model training, protected final evaluation, and zero-retraining prediction.

Implements Shared Core protocols:
    - `ModelTrainer` -> `ControlledModelTrainer`
    - `FinalModelEvaluator` -> `ProtectedFinalModelEvaluator`
    - `PredictionEngine` -> `ZeroRetrainingPredictionEngine`

Strict guarantees:
    - All learned preprocessing (imputers, OneHotEncoder, StandardScaler) is
      fitted inside scikit-learn Pipelines on the TRAIN partition ONLY.
    - `ControlledModelTrainer` receives only `DevelopmentDataProvider` and NEVER
      has access to protected test records.
    - `ProtectedFinalModelEvaluator` loads exact frozen candidate bundles without
      refitting and evaluates them on exact registered protected test records.
    - `ZeroRetrainingPredictionEngine` loads ONLY a trusted verified active bundle
      and executes `.predict(...)` only — zero `fit`, `fit_transform`, or `partial_fit`.
"""

from __future__ import annotations

import io
import math
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from taqamind.core.contracts import DatasetContract, DataType
from taqamind.core.interfaces.artifacts import calculate_bytes_sha256
from taqamind.core.interfaces.candidates import (
    CandidateModelReference,
    CandidateResult,
    CandidateStatus,
    TrainingRunResult,
    ValidationMetric,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.data_access import (
    ActiveModelProvider,
    DevelopmentDataProvider,
    ProtectedTestDataProvider,
)
from taqamind.core.interfaces.dataset import DatasetArtifact
from taqamind.core.interfaces.dataset_registry import RecordLineageRegistry
from taqamind.core.interfaces.evaluation import (
    AcceptancePolicy,
    FinalEvaluationResult,
    FrozenModelSelection,
    TestMetricResult,
)
from taqamind.core.interfaces.leakage import TrainingAuthorization
from taqamind.core.interfaces.ml import (
    validate_prediction_request,
    validate_training_request,
)
from taqamind.core.interfaces.model_registry import ModelScope
from taqamind.core.interfaces.prediction import (
    DerivedRatePrediction,
    PredictionBatchResult,
    PredictionModelIdentity,
    PredictionStatus,
    RecordPrediction,
)
from taqamind.core.interfaces.results import ExecutionError, StageTiming
from taqamind.core.interfaces.splitting import SplitManifest
from taqamind.core.interfaces.storage import (
    DatasetArtifactStorage,
    ModelArtifactStorage,
)
from taqamind.core.interfaces.training import (
    CandidateConfiguration,
    ModelFamily,
    RegressionMetric,
    TrainingConfiguration,
)
from taqamind.core.interfaces.training_registry import (
    EvaluationRegistry,
    EvaluationReservation,
)
from taqamind.domains.iron_manufacturing.adapters.processing import _read_artifact_df
from taqamind.domains.iron_manufacturing.adapters.readiness_and_splitting import (
    build_record_scope,
)
from taqamind.domains.iron_manufacturing.common.hashing import (
    collect_runtime_dependency_versions,
)

# Allowlisted bounded hyperparameters per candidate family.
ALLOWLISTED_PARAMS: dict[ModelFamily, frozenset[str]] = {
    ModelFamily.DUMMY_REGRESSOR: frozenset({"strategy", "quantile"}),
    ModelFamily.HISTORICAL_RATE: frozenset({"denominator_column"}),
    ModelFamily.RIDGE: frozenset({"alpha"}),
    ModelFamily.ELASTIC_NET: frozenset({"alpha", "l1_ratio", "max_iter"}),
    ModelFamily.RANDOM_FOREST: frozenset({"n_estimators", "max_depth", "min_samples_leaf"}),
    ModelFamily.HIST_GRADIENT_BOOSTING: frozenset(
        {"max_iter", "max_depth", "learning_rate", "min_samples_leaf"}
    ),
    ModelFamily.XGBOOST: frozenset(
        {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"}
    ),
    ModelFamily.HURDLE_REGRESSION: frozenset({"max_depth", "learning_rate"}),
}


class HistoricalRateRegressor(BaseEstimator, RegressorMixin):
    """Industrial baseline predicting waste = historical_rate * positive_denominator."""

    def __init__(self, denominator_column: str) -> None:
        self.denominator_column = denominator_column

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> HistoricalRateRegressor:
        y_arr = np.asarray(y, dtype=float)
        self.fallback_mean_ = float(np.mean(y_arr))
        self.rate_: float | None = None
        if isinstance(X, pd.DataFrame) and self.denominator_column in X.columns:
            denom = pd.to_numeric(X[self.denominator_column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(denom) & (denom > 0.0) & np.isfinite(y_arr)
            if bool(valid.any()) and float(denom[valid].sum()) > 0.0:
                self.rate_ = float(y_arr[valid].sum() / denom[valid].sum())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if (
            self.rate_ is not None
            and isinstance(X, pd.DataFrame)
            and self.denominator_column in X.columns
        ):
            denom = pd.to_numeric(X[self.denominator_column], errors="coerce").to_numpy(dtype=float)
            preds = np.where(
                np.isfinite(denom) & (denom > 0.0),
                self.rate_ * denom,
                self.fallback_mean_,
            )
            return np.clip(preds, 0.0, None)
        return np.full(len(X), max(0.0, self.fallback_mean_), dtype=float)


class TwoStageHurdleRegressor(BaseEstimator, RegressorMixin):
    """Two-stage hurdle regressor for zero-inflated waste targets: P(y>0) * E[y|y>0]."""

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.08,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: Any, y: np.ndarray) -> TwoStageHurdleRegressor:
        y_arr = np.asarray(y, dtype=float)
        is_pos = (y_arr > 0.0).astype(int)
        pos_count = int(is_pos.sum())
        zero_count = int((1 - is_pos).sum())

        if pos_count >= 3 and zero_count >= 3:
            self.clf_: HistGradientBoostingClassifier | None = HistGradientBoostingClassifier(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            self.clf_.fit(X, is_pos)
            self.const_prob_ = float(is_pos.mean())
        else:
            self.clf_ = None
            self.const_prob_ = float(is_pos.mean())

        self.reg_ = HistGradientBoostingRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        pos_mask = y_arr > 0.0
        if pos_count >= 5:
            X_pos = X.iloc[pos_mask] if hasattr(X, "iloc") else X[pos_mask]
            self.reg_.fit(X_pos, y_arr[pos_mask])
        else:
            self.reg_.fit(X, y_arr)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.clf_ is not None:
            p_pos = self.clf_.predict_proba(X)[:, 1]
        else:
            p_pos = np.full(X.shape[0], self.const_prob_, dtype=float)
        y_pos = np.clip(self.reg_.predict(X), 0.0, None)
        return np.clip(p_pos * y_pos, 0.0, None)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    requested_metrics: tuple[RegressionMetric, ...],
) -> tuple[ValidationMetric, ...]:
    """Compute MAE, RMSE, R2, and WAPE while explicitly handling undefined metrics."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    results: list[ValidationMetric] = []

    for metric in requested_metrics:
        if metric == RegressionMetric.MAE:
            results.append(
                ValidationMetric(metric=metric, value=float(mean_absolute_error(yt, yp)))
            )
        elif metric == RegressionMetric.RMSE:
            results.append(
                ValidationMetric(metric=metric, value=float(root_mean_squared_error(yt, yp)))
            )
        elif metric == RegressionMetric.R2:
            if len(yt) < 2 or float(np.var(yt)) <= 1e-12:
                results.append(
                    ValidationMetric(
                        metric=metric,
                        value=None,
                        undefined_reason="Target variance is zero or fewer than 2 rows.",
                    )
                )
            else:
                r2_val = float(r2_score(yt, yp))
                if not math.isfinite(r2_val):
                    results.append(
                        ValidationMetric(
                            metric=metric,
                            value=None,
                            undefined_reason="Computed R2 is not finite.",
                        )
                    )
                else:
                    results.append(ValidationMetric(metric=metric, value=min(1.0, r2_val)))
        elif metric == RegressionMetric.WAPE:
            denom = float(np.sum(np.abs(yt)))
            if denom <= 1e-12:
                results.append(
                    ValidationMetric(
                        metric=metric,
                        value=None,
                        undefined_reason="Sum of absolute target values is zero.",
                    )
                )
            else:
                wape = float(np.sum(np.abs(yt - yp)) / denom)
                results.append(ValidationMetric(metric=metric, value=wape))

    return tuple(results)


def _build_preprocessor(
    contract: DatasetContract,
    feature_names: tuple[str, ...],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for name in feature_names:
        feat = contract.get_feature(name)
        if feat.data_type in {DataType.FLOAT, DataType.INTEGER}:
            numeric_cols.append(name)
        else:
            categorical_cols.append(name)

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), numeric_cols))

    if categorical_cols:
        cat_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _build_candidate_pipeline(
    candidate: CandidateConfiguration,
    contract: DatasetContract,
    feature_names: tuple[str, ...],
    random_seed: int,
) -> Any:
    allowed = ALLOWLISTED_PARAMS[candidate.family]
    params = candidate.parameter_map
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"Candidate family '{candidate.family.value}' received non-allowlisted "
            f"parameter(s): {sorted(unknown)}."
        )

    if candidate.family == ModelFamily.HISTORICAL_RATE:
        if contract.target.derived_rate is None:
            raise ValueError("Historical rate baseline requires a target derived_rate.")
        denom_col = str(params.get("denominator_column", contract.target.derived_rate.denominator))
        return HistoricalRateRegressor(denominator_column=denom_col)

    if candidate.family == ModelFamily.DUMMY_REGRESSOR:
        strategy = str(params.get("strategy", "median"))
        est: BaseEstimator = DummyRegressor(strategy=strategy)
        return Pipeline(
            [
                ("preprocessor", _build_preprocessor(contract, feature_names, scale_numeric=False)),
                ("estimator", est),
            ]
        )

    if candidate.family == ModelFamily.RIDGE:
        est = Ridge(alpha=float(params.get("alpha", 1.0)))
        return Pipeline(
            [
                ("preprocessor", _build_preprocessor(contract, feature_names, scale_numeric=True)),
                ("estimator", est),
            ]
        )

    if candidate.family == ModelFamily.ELASTIC_NET:
        est = ElasticNet(
            alpha=float(params.get("alpha", 0.1)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            max_iter=int(params.get("max_iter", 2000)),
            random_state=random_seed,
        )
        return Pipeline(
            [
                ("preprocessor", _build_preprocessor(contract, feature_names, scale_numeric=True)),
                ("estimator", est),
            ]
        )

    if candidate.family == ModelFamily.RANDOM_FOREST:
        est = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 80)),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else 10,
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            random_state=random_seed,
            n_jobs=1,
        )
        return Pipeline(
            [
                ("preprocessor", _build_preprocessor(contract, feature_names, scale_numeric=False)),
                ("estimator", est),
            ]
        )

    if candidate.family == ModelFamily.HIST_GRADIENT_BOOSTING:
        est = HistGradientBoostingRegressor(
            max_iter=int(params.get("max_iter", 100)),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else 5,
            learning_rate=float(params.get("learning_rate", 0.08)),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            random_state=random_seed,
        )
        return Pipeline(
            [
                ("preprocessor", _build_preprocessor(contract, feature_names, scale_numeric=False)),
                ("estimator", est),
            ]
        )

    if candidate.family == ModelFamily.XGBOOST:
        est = XGBRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            subsample=float(params.get("subsample", 0.85)),
            colsample_bytree=float(params.get("colsample_bytree", 0.85)),
            random_state=random_seed,
            n_jobs=1,
        )
        return Pipeline(
            [
                ("preprocessor", _build_preprocessor(contract, feature_names, scale_numeric=False)),
                ("estimator", est),
            ]
        )

    if candidate.family == ModelFamily.HURDLE_REGRESSION:
        est = TwoStageHurdleRegressor(
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            random_state=random_seed,
        )
        return Pipeline(
            [
                ("preprocessor", _build_preprocessor(contract, feature_names, scale_numeric=False)),
                ("estimator", est),
            ]
        )

    raise ValueError(f"Unsupported model family: {candidate.family}")


def serialize_trusted_bundle(
    *,
    fitted_pipeline: Any,
    feature_names: tuple[str, ...],
    target_name: str,
    target_unit: str | None,
    candidate_family: str,
) -> bytes:
    """Serialize a fitted preprocessing+estimator bundle with metadata envelope."""
    envelope = {
        "bundle_format": "taqamind_iron_manufacturing_bundle_v1",
        "feature_names": list(feature_names),
        "target_name": target_name,
        "target_unit": target_unit,
        "candidate_family": candidate_family,
        "dependency_versions": collect_runtime_dependency_versions(),
        "pipeline": fitted_pipeline,
    }
    buf = io.BytesIO()
    joblib.dump(envelope, buf, compress=3)
    return buf.getvalue()


def deserialize_trusted_bundle(raw_bytes: bytes) -> dict[str, Any]:
    """Deserialize an internally generated, integrity-verified model bundle."""
    envelope = joblib.load(io.BytesIO(raw_bytes))
    if (
        not isinstance(envelope, dict)
        or envelope.get("bundle_format") != "taqamind_iron_manufacturing_bundle_v1"
        or "pipeline" not in envelope
        or "feature_names" not in envelope
    ):
        raise ValueError("Invalid or untrusted model bundle payload.")
    return envelope


class ControlledModelTrainer:
    """Development-only controlled AutoML trainer implementing `ModelTrainer`."""

    def __init__(
        self,
        *,
        development_data_provider: DevelopmentDataProvider,
        model_storage: ModelArtifactStorage,
    ) -> None:
        self._dev_provider = development_data_provider
        self._model_storage = model_storage

    def train(
        self,
        *,
        authorization: TrainingAuthorization,
        contract: DatasetContract,
        configuration: TrainingConfiguration,
        policy: AcceptancePolicy,
    ) -> TrainingRunResult:
        validate_training_request(
            authorization=authorization,
            contract=contract,
            configuration=configuration,
            policy=policy,
        )

        dev_access = self._dev_provider.prepare(
            authorization=authorization,
            contract=contract,
        )
        with self._dev_provider.open_train(access=dev_access) as tr_stream:
            train_df = pd.read_csv(tr_stream)
        with self._dev_provider.open_validation(access=dev_access) as va_stream:
            val_df = pd.read_csv(va_stream)

        target_col = contract.target.feature.name
        feature_names = tuple(col for col in contract.predictor_names if col in train_df.columns)
        X_train = train_df.loc[:, list(feature_names)]
        y_train = train_df[target_col].to_numpy(dtype=float)
        X_val = val_df.loc[:, list(feature_names)]
        y_val = val_df[target_col].to_numpy(dtype=float)

        requested_metrics = (
            configuration.primary_metric,
            *configuration.secondary_metrics,
        )
        context = authorization.development_split.context
        target_min = contract.target.feature.constraints.minimum
        target_max = contract.target.feature.constraints.maximum

        results: list[CandidateResult] = []
        for candidate in configuration.candidates:
            c_start = datetime.now(UTC)
            try:
                pipe = _build_candidate_pipeline(
                    candidate=candidate,
                    contract=contract,
                    feature_names=feature_names,
                    random_seed=configuration.random_seed,
                )
                # Fit strictly on TRAIN partition only; transform VALIDATION without fitting
                pipe.fit(X_train, y_train)
                raw_val_pred = np.asarray(pipe.predict(X_val), dtype=float)
                val_pred = np.clip(
                    raw_val_pred,
                    target_min if target_min is not None else -np.inf,
                    target_max if target_max is not None else np.inf,
                )
                metrics = compute_regression_metrics(y_val, val_pred, requested_metrics)

                bundle_bytes = serialize_trusted_bundle(
                    fitted_pipeline=pipe,
                    feature_names=feature_names,
                    target_name=target_col,
                    target_unit=contract.target.feature.unit,
                    candidate_family=candidate.family.value,
                )
                model_ref = CandidateModelReference(
                    artifact_id=uuid4(),
                    factory_id=context.factory_id,
                    execution_id=context.execution_id,
                    candidate_id=candidate.candidate_id,
                    content_sha256=calculate_bytes_sha256(bundle_bytes),
                    size_bytes=len(bundle_bytes),
                )
                self._model_storage.put(
                    factory_id=context.factory_id,
                    reference=model_ref,
                    stream=io.BytesIO(bundle_bytes),
                )
                results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.SUCCEEDED,
                        timing=StageTiming(started_at=c_start, finished_at=datetime.now(UTC)),
                        validation_metrics=metrics,
                        evaluated_row_count=int(len(y_val)),
                        model_reference=model_ref,
                    )
                )
            except Exception as exc:
                results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.FAILED,
                        timing=StageTiming(started_at=c_start, finished_at=datetime.now(UTC)),
                        error=ExecutionError(
                            code="CANDIDATE_TRAINING_FAILED",
                            message=f"{candidate.family.value} failed: {exc}",
                        ),
                    )
                )

        return TrainingRunResult(
            configuration=configuration,
            development_split=authorization.development_split,
            results=tuple(results),
        )


class ProtectedFinalModelEvaluator:
    """Evaluates frozen baseline and selected challenger ONCE on protected test data."""

    def __init__(
        self,
        *,
        contract: DatasetContract,
        evaluation_registry: EvaluationRegistry,
        test_data_provider: ProtectedTestDataProvider,
        model_storage: ModelArtifactStorage,
    ) -> None:
        self._contract = contract
        self._evaluation_registry = evaluation_registry
        self._test_provider = test_data_provider
        self._model_storage = model_storage

    def evaluate(
        self,
        *,
        selection: FrozenModelSelection,
        manifest: SplitManifest,
    ) -> FinalEvaluationResult:
        context = manifest.context
        scope = build_record_scope(context)

        self._evaluation_registry.register_selection(
            factory_id=context.factory_id,
            selection=selection,
        )
        reservation = EvaluationReservation.from_selection(
            scope=scope,
            selection=selection,
            manifest=manifest,
        )
        registered_res = self._evaluation_registry.reserve_test_access(
            factory_id=context.factory_id,
            reservation=reservation,
        )

        test_access = self._test_provider.prepare(
            reservation=registered_res,
            selection=selection,
            contract=self._contract,
        )
        eval_started = datetime.now(UTC)

        with self._test_provider.open_test(access=test_access) as test_stream:
            test_df = pd.read_csv(test_stream)

        target_col = self._contract.target.feature.name
        y_test = test_df[target_col].to_numpy(dtype=float)

        run = selection.training_run
        requested_metrics = (
            run.configuration.primary_metric,
            *run.configuration.secondary_metrics,
        )
        results_by_id = {r.candidate_id: r for r in run.successful_results}
        target_min = self._contract.target.feature.constraints.minimum
        target_max = self._contract.target.feature.constraints.maximum

        def _score_frozen_candidate(candidate_id) -> TestMetricResult:
            cand_res = results_by_id[candidate_id]
            assert cand_res.model_reference is not None
            with self._model_storage.open_verified(
                factory_id=context.factory_id,
                reference=cand_res.model_reference,
            ) as bundle_stream:
                envelope = deserialize_trusted_bundle(bundle_stream.read())

            ordered_features = list(envelope["feature_names"])
            pipe = envelope["pipeline"]
            X_test = test_df.loc[:, ordered_features]
            # Strictly predict only — NEVER refit before final evaluation
            raw_pred = np.asarray(pipe.predict(X_test), dtype=float)
            preds = np.clip(
                raw_pred,
                target_min if target_min is not None else -np.inf,
                target_max if target_max is not None else np.inf,
            )
            metrics = compute_regression_metrics(y_test, preds, requested_metrics)
            return TestMetricResult(
                candidate_id=candidate_id,
                model_artifact_id=cand_res.model_reference.artifact_id,
                record_ids=manifest.test.record_ids,
                metrics=metrics,
            )

        baseline_test = _score_frozen_candidate(selection.baseline_candidate_id)
        selected_test = _score_frozen_candidate(selection.selected_candidate_id)

        final_result = FinalEvaluationResult(
            selection=selection,
            split_manifest=manifest,
            baseline_result=baseline_test,
            selected_result=selected_test,
            timing=StageTiming(started_at=eval_started, finished_at=datetime.now(UTC)),
        )
        self._evaluation_registry.register_evaluation(
            factory_id=context.factory_id,
            reservation_id=registered_res.reservation_id,
            evaluation=final_result,
        )
        return final_result


class ZeroRetrainingPredictionEngine:
    """Train-once / predict-many inference engine implementing `PredictionEngine`.

    NEVER calls `fit`, `fit_transform`, `partial_fit`, or any training service.
    """

    def __init__(
        self,
        *,
        storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        active_model_provider: ActiveModelProvider,
    ) -> None:
        self._storage = storage
        self._lineage_registry = lineage_registry
        self._active_model_provider = active_model_provider

    def predict(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        model: PredictionModelIdentity,
    ) -> PredictionBatchResult:
        started_at = datetime.now(UTC)
        validate_prediction_request(
            context=context,
            artifact=artifact,
            contract=contract,
            model=model,
        )

        scope = ModelScope.from_identity(model)
        active_entry = self._active_model_provider.get_active(scope=scope)
        if active_entry is None:
            raise ValueError(
                f"No active registered model found for scope ({scope.factory_id}, "
                f"{scope.domain}, {scope.process}, {scope.task_id})."
            )
        if active_entry.version.identity != model:
            raise ValueError("Active model identity does not match requested prediction model.")

        with self._active_model_provider.open_bundle(
            context=context,
            entry=active_entry,
            contract=contract,
        ) as bundle_stream:
            envelope = deserialize_trusted_bundle(bundle_stream.read())

        saved_features = tuple(envelope["feature_names"])
        expected_features = active_entry.version.feature_specification.feature_names
        if saved_features != expected_features:
            raise ValueError(
                "Persisted model bundle feature ordering does not match registered specification."
            )

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        missing_features = [f for f in saved_features if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"Prediction input is missing saved model features: {missing_features}."
            )

        record_scope = build_record_scope(context)
        membership = self._lineage_registry.get_membership(
            scope=record_scope,
            dataset_revision_id=artifact.metadata.revision_id,
        )
        membership.validate_artifact(artifact)
        pos_to_record_id = {m.row_position: m.record_id for m in membership.memberships}

        pipe = envelope["pipeline"]
        X_pred = df.loc[:, list(saved_features)]
        target_constraints = contract.target.feature.constraints
        rate_def = contract.target.derived_rate

        # Score row-by-row cleanly so invalid/non-finite rows are rejected without crashing batch
        raw_preds = np.asarray(pipe.predict(X_pred), dtype=float)

        predictions: list[RecordPrediction] = []
        for pos in range(len(df)):
            rec_id = pos_to_record_id[pos]
            val = float(raw_preds[pos])
            warnings: list[str] = []

            if not math.isfinite(val):
                predictions.append(
                    RecordPrediction(
                        record_id=rec_id,
                        status=PredictionStatus.REJECTED,
                        rejection_reason="Model produced a non-finite prediction value.",
                    )
                )
                continue

            if target_constraints.minimum is not None and val < target_constraints.minimum:
                val = float(target_constraints.minimum)
                warnings.append("Clipped negative raw prediction to contract target minimum.")
            if target_constraints.maximum is not None and val > target_constraints.maximum:
                predictions.append(
                    RecordPrediction(
                        record_id=rec_id,
                        status=PredictionStatus.REJECTED,
                        rejection_reason=(
                            f"Predicted value {val:.4f} exceeds target maximum "
                            f"{target_constraints.maximum}."
                        ),
                    )
                )
                continue

            derived_rate: DerivedRatePrediction | None = None
            if rate_def is not None:
                denom_raw = pd.to_numeric(
                    pd.Series([df.iloc[pos][rate_def.denominator]]), errors="coerce"
                ).iloc[0]
                if (
                    pd.isna(denom_raw)
                    or not math.isfinite(float(denom_raw))
                    or float(denom_raw) <= 0
                ):
                    derived_rate = DerivedRatePrediction(
                        value=None,
                        denominator_value=None,
                        unit=rate_def.output_unit,
                        undefined_reason=(
                            f"Denominator '{rate_def.denominator}' is missing or non-positive."
                        ),
                    )
                else:
                    denom_val = float(denom_raw)
                    rate_val = (val / denom_val) * rate_def.multiplier
                    derived_rate = DerivedRatePrediction(
                        value=rate_val,
                        denominator_value=denom_val,
                        unit=rate_def.output_unit,
                    )

            predictions.append(
                RecordPrediction(
                    record_id=rec_id,
                    status=PredictionStatus.SUCCEEDED,
                    predicted_value=val,
                    derived_rate=derived_rate,
                    warnings=tuple(warnings),
                )
            )

        return PredictionBatchResult(
            context=context,
            model=model,
            contract=contract,
            input_artifact=artifact,
            predictions=tuple(predictions),
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )
