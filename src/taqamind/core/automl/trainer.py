"""Core P1.7 ControlledAutoMLTrainer and leak-free TabularPreprocessor."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from io import BytesIO
from typing import Any, Self
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge

from taqamind.core.automl._metrics import compute_regression_metrics
from taqamind.core.automl.data_access import (
    convert_dataframe_by_contract,
    read_partition_dataframe,
)
from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import DataType
from taqamind.core.interfaces.artifacts import calculate_bytes_sha256
from taqamind.core.interfaces.candidates import (
    CandidateModelReference,
    CandidateResult,
    CandidateStatus,
    TrainingRunResult,
)
from taqamind.core.interfaces.data_access import DevelopmentDataProvider
from taqamind.core.interfaces.evaluation import AcceptancePolicy
from taqamind.core.interfaces.leakage import TrainingAuthorization
from taqamind.core.interfaces.ml import ModelTrainer
from taqamind.core.interfaces.results import ExecutionError, StageTiming
from taqamind.core.interfaces.storage import ModelArtifactStorage
from taqamind.core.interfaces.training import (
    CandidateConfiguration,
    DevelopmentSplit,
    ModelFamily,
    TrainingConfiguration,
)
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter


class TabularPreprocessor:
    """Domain-agnostic tabular preprocessor fit strictly on the training partition."""

    def __init__(self, *, contract: DatasetContract) -> None:
        self._predictor_names = tuple(contract.predictor_names)
        self._feature_types = {
            f.name: f.data_type for f in contract.features if f.name in self._predictor_names
        }
        self.numeric_cols_: tuple[str, ...] = ()
        self.categorical_cols_: tuple[str, ...] = ()
        self.numeric_medians_: dict[str, float] = {}
        self.categorical_values_: dict[str, tuple[str, ...]] = {}
        self.feature_names_out_: tuple[str, ...] = ()
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> Self:
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        for col in self._predictor_names:
            dtype = self._feature_types.get(col)
            if dtype in {DataType.INTEGER, DataType.FLOAT, DataType.BOOLEAN}:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        self.numeric_cols_ = tuple(numeric_cols)
        self.categorical_cols_ = tuple(categorical_cols)
        self.numeric_medians_ = {}
        for col in self.numeric_cols_:
            series = (
                pd.to_numeric(df[col], errors="coerce")
                if col in df.columns
                else pd.Series(dtype=float)
            )
            med = float(series.median()) if not series.dropna().empty else 0.0
            self.numeric_medians_[col] = med if np.isfinite(med) else 0.0

        self.categorical_values_ = {}
        out_names: list[str] = list(self.numeric_cols_)
        for col in self.categorical_cols_:
            if col in df.columns:
                vals = sorted({str(v) for v in df[col].fillna("__MISSING__").astype(str).tolist()})
            else:
                vals = ["__MISSING__"]
            self.categorical_values_[col] = tuple(vals)
            for v in vals:
                out_names.append(f"{col}__{v}")

        self.feature_names_out_ = tuple(out_names)
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TabularPreprocessor must be fit before calling transform.")
        n_rows = len(df)
        cols: list[np.ndarray] = []

        for col in self.numeric_cols_:
            if col in df.columns:
                arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            else:
                arr = np.full(n_rows, self.numeric_medians_[col], dtype=float)
            arr = np.where(np.isfinite(arr), arr, self.numeric_medians_[col])
            cols.append(arr)

        for col in self.categorical_cols_:
            raw = (
                df[col].fillna("__MISSING__").astype(str).tolist()
                if col in df.columns
                else ["__MISSING__"] * n_rows
            )
            raw_arr = np.array(raw, dtype=object)
            for val in self.categorical_values_[col]:
                cols.append((raw_arr == val).astype(float))

        if not cols:
            return np.zeros((n_rows, 0), dtype=float)
        return np.column_stack(cols)


class HistoricalRateEstimator:
    """Baseline estimator predicting target = historical_rate * denominator."""

    def __init__(self, *, denominator_index: int) -> None:
        self._denominator_index = denominator_index
        self.rate_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        denom = X[:, self._denominator_index]
        mask = np.isfinite(denom) & (denom > 0)
        if np.any(mask):
            denom_sum = float(np.sum(denom[mask]))
            self.rate_ = float(np.sum(y[mask]) / denom_sum) if denom_sum > 0 else 0.0
        else:
            self.rate_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        denom = np.maximum(0.0, X[:, self._denominator_index])
        return np.maximum(0.0, self.rate_ * denom)


class HurdleRegressorEstimator:
    """Two-stage zero-inflated hurdle regressor: P(y > 0) * E[y | y > 0]."""

    def __init__(self, *, random_state: int = 42, max_iter: int = 200) -> None:
        self._random_state = random_state
        self._max_iter = max_iter
        self._clf: LogisticRegression | None = None
        self._const_prob: float = 0.0
        self._reg: HistGradientBoostingRegressor | DummyRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        y_pos_mask = y > 0
        n_pos = int(np.sum(y_pos_mask))
        if n_pos == 0:
            self._const_prob = 0.0
            self._clf = None
            self._reg = DummyRegressor(strategy="constant", constant=0.0).fit(X, y)
            return self
        if n_pos == len(y):
            self._const_prob = 1.0
            self._clf = None
        else:
            self._clf = LogisticRegression(
                max_iter=self._max_iter,
                random_state=self._random_state,
            ).fit(X, y_pos_mask.astype(int))
            self._const_prob = float(np.mean(y_pos_mask))

        X_pos = X[y_pos_mask]
        y_pos = y[y_pos_mask]
        if len(y_pos) < 5:
            self._reg = DummyRegressor(strategy="mean").fit(X_pos, y_pos)
        else:
            self._reg = HistGradientBoostingRegressor(
                random_state=self._random_state,
                max_iter=min(100, self._max_iter),
            ).fit(X_pos, y_pos)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._reg is not None
        if self._clf is None:
            prob = np.full(len(X), self._const_prob, dtype=float)
        else:
            prob = self._clf.predict_proba(X)[:, 1]
        amount = np.maximum(0.0, self._reg.predict(X))
        return np.maximum(0.0, prob * amount)


class ControlledAutoMLTrainer(ModelTrainer):
    """Fits only allowlisted model families on approved train/validation partitions."""

    def __init__(
        self,
        *,
        development_data_provider: DevelopmentDataProvider,
        model_storage: ModelArtifactStorage,
        type_converter: ContractTypeConverter | None = None,
    ) -> None:
        self._data_provider = development_data_provider
        self._model_storage = model_storage
        self._converter = type_converter or ContractTypeConverter()

    def _build_estimator(
        self,
        *,
        candidate: CandidateConfiguration,
        preprocessor: TabularPreprocessor,
        contract: DatasetContract,
        random_seed: int,
    ) -> Any:
        params = dict(candidate.parameter_map)
        family = candidate.family

        if family == ModelFamily.DUMMY_REGRESSOR:
            return DummyRegressor(strategy=str(params.get("strategy", "median")))
        if family == ModelFamily.HISTORICAL_RATE:
            assert contract.target.derived_rate is not None
            denom_name = contract.target.derived_rate.denominator
            denom_idx = preprocessor.feature_names_out_.index(denom_name)
            return HistoricalRateEstimator(denominator_index=denom_idx)
        if family == ModelFamily.RIDGE:
            return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=random_seed)
        if family == ModelFamily.ELASTIC_NET:
            return ElasticNet(
                alpha=float(params.get("alpha", 0.1)),
                l1_ratio=float(params.get("l1_ratio", 0.5)),
                max_iter=int(params.get("max_iter", 2000)),
                random_state=random_seed,
            )
        if family == ModelFamily.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 50)),
                max_depth=params.get("max_depth", 8),
                random_state=random_seed,
                n_jobs=1,
            )
        if family == ModelFamily.HIST_GRADIENT_BOOSTING:
            return HistGradientBoostingRegressor(
                max_iter=int(params.get("max_iter", 80)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                max_depth=params.get("max_depth", 6),
                random_state=random_seed,
            )
        if family == ModelFamily.XGBOOST:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=int(params.get("n_estimators", 50)),
                max_depth=int(params.get("max_depth", 4)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                random_state=random_seed,
                n_jobs=1,
                verbosity=0,
            )
        if family == ModelFamily.HURDLE_REGRESSION:
            return HurdleRegressorEstimator(
                random_state=random_seed,
                max_iter=int(params.get("max_iter", 200)),
            )
        raise ValueError(f"Unsupported model family '{family}'.")

    def train(
        self,
        *,
        authorization: TrainingAuthorization,
        contract: DatasetContract,
        configuration: TrainingConfiguration,
        policy: AcceptancePolicy,
    ) -> TrainingRunResult:
        _ = policy
        configuration.validate_contract(contract)
        access = self._data_provider.prepare(authorization=authorization, contract=contract)

        with self._data_provider.open_train(access=access) as train_stream:
            train_raw = read_partition_dataframe(train_stream)
        with self._data_provider.open_validation(access=access) as val_stream:
            val_raw = read_partition_dataframe(val_stream)

        train_df = convert_dataframe_by_contract(
            train_raw,
            contract=contract,
            converter=self._converter,
        )
        val_df = convert_dataframe_by_contract(
            val_raw,
            contract=contract,
            converter=self._converter,
        )

        target_col = contract.target.feature.name
        y_train = pd.to_numeric(train_df[target_col], errors="raise").to_numpy(dtype=float)
        y_val = pd.to_numeric(val_df[target_col], errors="raise").to_numpy(dtype=float)
        target_min = contract.target.feature.constraints.minimum
        must_be_non_negative = target_min is not None and target_min >= 0.0

        preprocessor = TabularPreprocessor(contract=contract).fit(train_df)
        X_train = preprocessor.transform(train_df)
        X_val = preprocessor.transform(val_df)

        requested_metrics = (configuration.primary_metric, *configuration.secondary_metrics)
        context = authorization.development_split.context
        candidate_results: list[CandidateResult] = []

        for candidate in configuration.candidates:
            started_at = datetime.now(UTC)
            try:
                estimator = self._build_estimator(
                    candidate=candidate,
                    preprocessor=preprocessor,
                    contract=contract,
                    random_seed=configuration.random_seed,
                )
                estimator.fit(X_train, y_train)
                y_pred = np.asarray(estimator.predict(X_val), dtype=float)
                if must_be_non_negative:
                    y_pred = np.maximum(0.0, y_pred)

                val_metrics = compute_regression_metrics(
                    y_true=y_val,
                    y_pred=y_pred,
                    metrics=requested_metrics,
                )
                bundle = {
                    "preprocessor": preprocessor,
                    "estimator": estimator,
                    "family": candidate.family.value,
                    "target_name": target_col,
                    "predictor_names": tuple(contract.predictor_names),
                    "feature_names_out": preprocessor.feature_names_out_,
                    "contract_version": str(contract.version),
                    "must_be_non_negative": must_be_non_negative,
                }
                bundle_bytes = pickle.dumps(bundle)
                artifact_id = uuid4()
                sha256 = calculate_bytes_sha256(bundle_bytes)
                model_ref = CandidateModelReference(
                    artifact_id=artifact_id,
                    factory_id=context.factory_id,
                    execution_id=context.execution_id,
                    candidate_id=candidate.candidate_id,
                    content_sha256=sha256,
                    size_bytes=len(bundle_bytes),
                )
                self._model_storage.put(
                    factory_id=context.factory_id,
                    reference=model_ref,
                    stream=BytesIO(bundle_bytes),
                )
                finished_at = datetime.now(UTC)
                candidate_results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.SUCCEEDED,
                        timing=StageTiming(started_at=started_at, finished_at=finished_at),
                        validation_metrics=val_metrics,
                        evaluated_row_count=len(y_val),
                        model_reference=model_ref,
                    )
                )
            except ImportError as exc:
                finished_at = datetime.now(UTC)
                candidate_results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.SKIPPED,
                        timing=StageTiming(started_at=started_at, finished_at=finished_at),
                        skip_reason=f"Missing optional dependency: {exc}",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                finished_at = datetime.now(UTC)
                candidate_results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.FAILED,
                        timing=StageTiming(started_at=started_at, finished_at=finished_at),
                        error=ExecutionError(
                            code="CANDIDATE_TRAINING_FAILED",
                            message=f"{type(exc).__name__}: {exc}",
                        ),
                    )
                )

        return TrainingRunResult(
            configuration=configuration,
            development_split=DevelopmentSplit(
                split_id=access.authorization.development_split.split_id,
                context=context,
                dataset_revision_id=access.train.dataset_revision_id,
                train=access.authorization.development_split.train,
                validation=access.authorization.development_split.validation,
            ),
            results=tuple(candidate_results),
        )


__all__ = [
    "ControlledAutoMLTrainer",
    "HistoricalRateEstimator",
    "HurdleRegressorEstimator",
    "TabularPreprocessor",
]
