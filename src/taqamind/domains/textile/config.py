"""Default split, training, and acceptance policy configurations for Textile V1."""

from __future__ import annotations

from taqamind.core.contracts import TaskType
from taqamind.core.interfaces import (
    AcceptancePolicy,
    CandidateConfiguration,
    CandidateRole,
    ModelFamily,
    ModelParameter,
    RegressionMetric,
    SplitConfiguration,
    SplitStrategy,
    TrainingBudget,
    TrainingConfiguration,
)
from taqamind.domains.textile.knitting.contracts import KNITTING_PROCESS_ID
from taqamind.domains.textile.weaving.contracts import WEAVING_PROCESS_ID


def build_default_split_configuration(
    process_id: str,
    *,
    random_seed: int = 42,
) -> SplitConfiguration:
    """Return the process-appropriate leakage-safe split configuration."""
    if process_id == WEAVING_PROCESS_ID:
        return SplitConfiguration(
            strategy=SplitStrategy.GROUP,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=random_seed,
            justification=(
                "Weaving records lack full calendar dates; group split by order_id "
                "prevents same-order leakage across partitions."
            ),
        )
    if process_id == KNITTING_PROCESS_ID:
        return SplitConfiguration(
            strategy=SplitStrategy.GROUP_TEMPORAL,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=random_seed,
            justification=(
                "Knitting shift runs have scheduled production_date and order_lot_id; "
                "group-temporal split prevents both lot leakage and temporal look-ahead."
            ),
        )
    raise ValueError(f"Unsupported textile process '{process_id}'.")


def build_default_training_configuration(
    *,
    random_seed: int = 42,
    include_heavy: bool = True,
    max_training_seconds: int = 120,
) -> TrainingConfiguration:
    """Return controlled AutoML configuration with explicit baselines and challengers."""
    candidates: list[CandidateConfiguration] = [
        CandidateConfiguration(
            family=ModelFamily.DUMMY_REGRESSOR,
            role=CandidateRole.BASELINE,
            parameters=(ModelParameter(name="strategy", value="median"),),
        ),
        CandidateConfiguration(
            family=ModelFamily.HISTORICAL_RATE,
            role=CandidateRole.BASELINE,
        ),
        CandidateConfiguration(
            family=ModelFamily.RIDGE,
            role=CandidateRole.CHALLENGER,
            parameters=(ModelParameter(name="alpha", value=1.0),),
        ),
        CandidateConfiguration(
            family=ModelFamily.ELASTIC_NET,
            role=CandidateRole.CHALLENGER,
            parameters=(
                ModelParameter(name="alpha", value=0.2),
                ModelParameter(name="l1_ratio", value=0.5),
            ),
        ),
        CandidateConfiguration(
            family=ModelFamily.HIST_GRADIENT_BOOSTING,
            role=CandidateRole.CHALLENGER,
            parameters=(
                ModelParameter(name="max_iter", value=80),
                ModelParameter(name="max_depth", value=5),
                ModelParameter(name="learning_rate", value=0.08),
            ),
        ),
        CandidateConfiguration(
            family=ModelFamily.HURDLE_REGRESSION,
            role=CandidateRole.CHALLENGER,
            parameters=(ModelParameter(name="max_iter", value=200),),
        ),
    ]
    if include_heavy:
        candidates.extend(
            [
                CandidateConfiguration(
                    family=ModelFamily.RANDOM_FOREST,
                    role=CandidateRole.CHALLENGER,
                    parameters=(
                        ModelParameter(name="n_estimators", value=50),
                        ModelParameter(name="max_depth", value=8),
                        ModelParameter(name="min_samples_leaf", value=2),
                    ),
                ),
                CandidateConfiguration(
                    family=ModelFamily.XGBOOST,
                    role=CandidateRole.CHALLENGER,
                    parameters=(
                        ModelParameter(name="n_estimators", value=60),
                        ModelParameter(name="max_depth", value=4),
                        ModelParameter(name="learning_rate", value=0.08),
                        ModelParameter(name="subsample", value=0.9),
                    ),
                ),
            ]
        )

    return TrainingConfiguration(
        task_type=TaskType.REGRESSION,
        primary_metric=RegressionMetric.MAE,
        secondary_metrics=(RegressionMetric.RMSE, RegressionMetric.WAPE, RegressionMetric.R2),
        candidates=tuple(candidates),
        budget=TrainingBudget(
            max_trials=len(candidates),
            max_training_seconds=max_training_seconds,
            max_parallel_jobs=1,
        ),
        random_seed=random_seed,
        justification="Controlled AutoML regression suite for zero-inflated textile waste.",
    )


def build_default_acceptance_policy(
    *,
    minimum_improvement_pct: float = 0.5,
    maximum_test_mae: float | None = None,
) -> AcceptancePolicy:
    """Return default model acceptance policy requiring improvement over baseline."""
    return AcceptancePolicy(
        minimum_improvement_pct=minimum_improvement_pct,
        maximum_test_mae=maximum_test_mae,
        justification=(
            "Challenger must outperform best baseline on protected test MAE "
            "before activation is permitted."
        ),
    )
