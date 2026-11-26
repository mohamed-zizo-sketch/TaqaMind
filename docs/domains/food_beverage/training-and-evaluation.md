# TaqaMind — Food & Beverage Training and Evaluation

---

## 1. Controlled Candidate Search

- **Baselines:** `DummyRegressor(strategy="median")` (and `HistoricalRate` supported in Core enum).
- **Challengers (Allowlisted Hyperparameters Only):**
  - `Ridge` (`alpha`)
  - `ElasticNet` (`alpha`, `l1_ratio`, `max_iter`)
  - `RandomForestRegressor` (`n_estimators`, `max_depth`, `min_samples_leaf`)
  - `HistGradientBoostingRegressor` (`max_iter`, `max_depth`, `learning_rate`)
  - `XGBRegressor` (`n_estimators`, `max_depth`, `learning_rate`, `subsample`)

## 2. Metrics & Leakage Guarantees

- **Primary Metric:** `MAE` (in target units: `kg` or `L`).
- **Secondary Metrics:** `RMSE`, `WAPE` (explicitly undefined if `sum(|y_true|) == 0`), `R²` (negative `R²` reported honestly without clipping).
- **Preprocessing Isolation:** `ColumnTransformer` (`SimpleImputer` + `StandardScaler` + `OneHotEncoder`) is fitted strictly inside `TRAIN` and applied via `.transform()` to `VALIDATION` and `TEST`.
- **Protected Test Lockdown:** `FrozenModelSelection` locks the selected challenger and baseline before `TEST` access; `RecordUsageRole.FINAL_TEST` prevents historical test rows from re-entering future independent test splits.
- **Acceptance Policy:** Candidate is `ACCEPTED` only if it improves baseline `MAE` by at least `minimum_improvement_pct` and satisfies `maximum_test_mae` (if configured).
