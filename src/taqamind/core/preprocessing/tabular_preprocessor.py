"""Leakage-safe sklearn preprocessing transformer fit strictly on training data."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import DataType, FeatureRole, MissingValuePolicy
from taqamind.core.contracts.feature import FeatureContract


class _EpochSecondsTransformer(BaseEstimator, TransformerMixin):
    """Deterministically convert date/datetime columns to numeric epoch seconds."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(X)
        columns: list[np.ndarray] = []
        for col in frame.columns:
            parsed = pd.to_datetime(frame[col], errors="coerce", utc=True)
            epoch = pd.Series(np.nan, index=frame.index, dtype="float64")
            valid = parsed.notna()
            if bool(valid.any()):
                ns = parsed.loc[valid].astype("int64")
                epoch.loc[valid] = (ns / 1_000_000_000.0).to_numpy()
            columns.append(epoch.to_numpy(dtype="float64"))
        if not columns:
            return np.empty((len(frame), 0), dtype="float64")
        return np.column_stack(columns)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray([f"{name}_epoch_s" for name in input_features], dtype=object)


class TabularPreprocessor:
    """Fit preprocessing on training rows only and transform unseen rows safely."""

    def __init__(
        self,
        contract: DatasetContract,
        *,
        scale_numeric: bool = True,
    ) -> None:
        self._contract = contract
        self._scale_numeric = scale_numeric
        self._ordered_predictors = self._resolve_predictors(contract)
        if not self._ordered_predictors:
            raise ValueError(
                "DatasetContract has no predictors available at prediction time."
            )
        self._transformer = self._build_column_transformer()
        self._is_fitted = False

    @staticmethod
    def _resolve_predictors(contract: DatasetContract) -> tuple[FeatureContract, ...]:
        return tuple(
            feature
            for feature in contract.features
            if feature.role == FeatureRole.PREDICTOR
            and contract.is_available_at_prediction(feature)
        )

    @property
    def ordered_predictors(self) -> tuple[str, ...]:
        """Return ordered canonical predictor names used by the fitted transformer."""
        return tuple(feature.name for feature in self._ordered_predictors)

    @property
    def is_fitted(self) -> bool:
        """Return whether the preprocessor has been fit on training data."""
        return self._is_fitted

    def _build_column_transformer(self) -> ColumnTransformer:
        transformers: list[tuple[str, object, list[str]]] = []

        numeric_impute = [
            f.name
            for f in self._ordered_predictors
            if f.data_type in {DataType.INTEGER, DataType.FLOAT}
        ]
        if numeric_impute:
            steps: list[tuple[str, object]] = [
                ("imputer", SimpleImputer(strategy="median")),
            ]
            if self._scale_numeric:
                steps.append(("scaler", StandardScaler()))
            transformers.append(("numeric", Pipeline(steps), numeric_impute))

        categorical_impute = [
            f.name
            for f in self._ordered_predictors
            if f.data_type in {DataType.STRING, DataType.CATEGORY}
            and f.missing_policy != MissingValuePolicy.ALLOW
        ]
        if categorical_impute:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("categorical_freq", cat_pipe, categorical_impute))

        categorical_allow = [
            f.name
            for f in self._ordered_predictors
            if f.data_type in {DataType.STRING, DataType.CATEGORY}
            and f.missing_policy == MissingValuePolicy.ALLOW
        ]
        if categorical_allow:
            allow_pipe = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("categorical_allow", allow_pipe, categorical_allow))

        boolean_cols = [
            f.name
            for f in self._ordered_predictors
            if f.data_type == DataType.BOOLEAN
        ]
        if boolean_cols:
            bool_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            drop="if_binary",
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            )
            transformers.append(("boolean", bool_pipe, boolean_cols))

        temporal_cols = [
            f.name
            for f in self._ordered_predictors
            if f.data_type in {DataType.DATE, DataType.DATETIME}
        ]
        if temporal_cols:
            temp_steps: list[tuple[str, object]] = [
                ("epoch", _EpochSecondsTransformer()),
                ("imputer", SimpleImputer(strategy="median")),
            ]
            if self._scale_numeric:
                temp_steps.append(("scaler", StandardScaler()))
            transformers.append(("temporal", Pipeline(temp_steps), temporal_cols))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.ordered_predictors if col not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing required predictors: {missing}")

        subset = df.loc[:, list(self.ordered_predictors)].copy()
        for feature in self._ordered_predictors:
            col = feature.name
            if feature.data_type in {DataType.INTEGER, DataType.FLOAT}:
                subset[col] = pd.to_numeric(subset[col], errors="coerce").astype("float64")
            elif feature.data_type in {DataType.STRING, DataType.CATEGORY}:
                subset[col] = subset[col].where(subset[col].notna(), np.nan)
                non_null = subset[col].notna()
                subset.loc[non_null, col] = subset.loc[non_null, col].astype(str)
            elif feature.data_type == DataType.BOOLEAN:
                subset[col] = subset[col].where(subset[col].notna(), np.nan)
        return subset

    def fit(self, X_train: pd.DataFrame) -> Self:
        """Fit imputers, scalers, and encoders strictly on training data."""
        prepared = self._prepare_frame(X_train)
        self._transformer.fit(prepared)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using already-fitted training parameters only."""
        if not self._is_fitted:
            raise RuntimeError(
                "TabularPreprocessor must be fit on training data before transform."
            )
        prepared = self._prepare_frame(X)
        return np.asarray(self._transformer.transform(prepared), dtype="float64")

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on training data and return transformed matrix."""
        self.fit(X_train)
        return self.transform(X_train)

    def get_feature_names_out(self) -> tuple[str, ...]:
        """Return output feature names after encoding."""
        if not self._is_fitted:
            raise RuntimeError("TabularPreprocessor must be fit first.")
        return tuple(str(name) for name in self._transformer.get_feature_names_out())
