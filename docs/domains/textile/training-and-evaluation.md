# Training, Evaluation & Acceptance Policy

## Splitting Strategy

- **Weaving:** `SplitStrategy.GROUP` grouped by `order_id` (`70% / 15% / 15%`). Prevents same-order beam batches from leaking across train/validation/test.
- **Knitting:** `SplitStrategy.GROUP_TEMPORAL` grouped by `order_lot_id` and ordered by `production_date` (`70% / 15% / 15%`). Prevents both lot overlap and temporal look-ahead.

## Controlled AutoML Candidates (`config.py`)

- **Baselines (`CandidateRole.BASELINE`):**
  - `DummyRegressor(strategy="median")`
  - `_HistoricalRateEstimator` (train-only ratio `sum(y) / sum(planned_qty)`)
- **Challengers (`CandidateRole.CHALLENGER`):**
  - `Ridge`, `ElasticNet`, `HistGradientBoostingRegressor`, `_TwoStageHurdleEstimator` (`P(y > 0 | X) * E[y | y > 0, X]`), `RandomForestRegressor`, `XGBRegressor`

## Leakage & Evaluation Guarantees

1. `TabularPreprocessor` is fit **strictly on `train_df`** inside each candidate trial.
2. `TextileProtectedFinalEvaluator` reserves test records once via `LocalEvaluationRegistry`; re-evaluating exposed test records raises `RegistryConflictError`.
3. `AcceptancePolicy` requires `policy.created_at <= context.created_at` and positive improvement over the best baseline on protected test MAE before activation is allowed.
