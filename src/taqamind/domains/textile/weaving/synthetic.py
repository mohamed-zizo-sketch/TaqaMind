"""Explicit synthetic dataset generators for offline weaving verification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_weaving_training_frame(
    *,
    num_orders: int = 30,
    batches_per_order: int = 3,
    seed: int = 42,
    zero_inflation_prob: float = 0.65,
) -> pd.DataFrame:
    """Generate synthetic pre-production weaving beam batches with planted signal.

    Uses alphanumeric string batch_record_id and order_id keys and canonical units (yd).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    constructions = ("40x40/110x90", "30x30/80x60", "20x20/60x50")

    for order_idx in range(1, num_orders + 1):
        order_id = f"ORD-{order_idx:04d}"
        construction = constructions[order_idx % len(constructions)]
        warp_count = float(rng.choice([20.0, 30.0, 40.0, 50.0]))
        weft_count = float(rng.choice([20.0, 30.0, 40.0]))
        epi = float(rng.integers(60, 130))
        ppi = float(rng.integers(50, 100))
        base_allowance = float(rng.uniform(2.0, 8.0))
        base_shrink = float(rng.uniform(1.0, 5.0))

        for batch_idx in range(1, batches_per_order + 1):
            batch_id = f"WB-{order_idx:04d}-{batch_idx:02d}"
            beam_len = float(rng.uniform(400.0, 2500.0))
            grey_len = beam_len * float(rng.uniform(0.95, 1.05))
            finish_len = grey_len * (1.0 - base_shrink / 100.0)

            # Planted deterministic signal + small noise so models beat DummyRegressor.
            density_stress = (epi + ppi) / 100.0
            yarn_thinness = (warp_count + weft_count) / 40.0
            expected_waste = (
                0.012 * beam_len * density_stress * (0.7 + 0.3 * yarn_thinness)
                + float(rng.normal(0.0, 1.5))
            )
            if rng.random() < zero_inflation_prob:
                waste = 0.0
            else:
                waste = float(max(0.5, round(expected_waste, 3)))

            rows.append(
                {
                    "batch_record_id": batch_id,
                    "order_id": order_id,
                    "planned_beam_length_yds": round(beam_len, 3),
                    "planned_finish_fabric_yds": round(finish_len, 3),
                    "planned_grey_fabric_yds": round(grey_len, 3),
                    "fabric_allowance_pct": round(base_allowance, 2),
                    "shrink_allowance_pct": round(base_shrink, 2),
                    "warp_count_ne": warp_count,
                    "weft_count_ne": weft_count,
                    "warp_ends_per_inch": epi,
                    "weft_picks_per_inch": ppi,
                    "fabric_construction": construction,
                    "waste_length_yds": waste,
                }
            )
    return pd.DataFrame(rows)


def generate_synthetic_weaving_prediction_frame(
    *,
    num_batches: int = 6,
    seed: int = 99,
    start_index: int = 1,
) -> pd.DataFrame:
    """Generate prediction-only weaving rows (without target or post-production fields)."""
    df = generate_synthetic_weaving_training_frame(
        num_orders=max(2, num_batches),
        batches_per_order=1,
        seed=seed,
    ).iloc[:num_batches].copy()
    df["batch_record_id"] = [
        f"WB-PRED-{start_index + idx:04d}" for idx in range(len(df))
    ]
    df["order_id"] = [
        f"ORD-PRED-{start_index + idx:04d}" for idx in range(len(df))
    ]
    return df.drop(columns=["waste_length_yds"]).reset_index(drop=True)
