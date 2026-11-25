"""Explicit synthetic dataset generators for offline knitting verification."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def generate_synthetic_knitting_training_frame(
    *,
    num_lots: int = 28,
    runs_per_lot: int = 3,
    seed: int = 42,
    zero_inflation_prob: float = 0.70,
) -> pd.DataFrame:
    """Generate synthetic knitting shift runs with dates, lot groups, and zero-inflation."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    brands = ("Fukuhara", "Mayer & Cie", "Terrot", "Lisky")
    start_date = date(2026, 1, 1)

    day_offset = 0
    for lot_idx in range(1, num_lots + 1):
        lot_id = f"LOT-{lot_idx:04d}"
        brand = brands[lot_idx % len(brands)]
        mc_dia = float(rng.choice([26.0, 30.0, 34.0]))
        mc_gauge = float(rng.choice([20.0, 24.0, 28.0]))
        tex_1 = float(rng.choice([19.68, 21.09, 24.60, 29.52]))
        stitch_mm = float(rng.uniform(2.5, 3.6))
        gsm = float(rng.uniform(135.0, 240.0))
        feeders = int(rng.choice([60, 72, 90, 96]))

        for run_idx in range(1, runs_per_lot + 1):
            run_id = f"KN-{lot_idx:04d}-{run_idx:02d}"
            run_date = (start_date + timedelta(days=day_offset)).isoformat()
            day_offset += 1
            planned_kg = float(rng.uniform(40.0, 320.0))
            tightness = (tex_1 ** 0.5) / (stitch_mm / 10.0)
            gauge_penalty = 1.0 + 0.15 * max(0.0, mc_gauge - 20.0)
            brand_mult = 1.35 if brand in {"Terrot", "Lisky"} else 0.75
            signal = (
                0.015 * planned_kg * max(0.5, tightness - 0.8) * gauge_penalty * brand_mult
                + float(rng.normal(0.0, 0.25))
            )
            waste_kg = (
                0.0
                if rng.random() < zero_inflation_prob
                else float(max(0.2, round(signal, 3)))
            )

            rows.append(
                {
                    "run_id": run_id,
                    "order_lot_id": lot_id,
                    "production_date": run_date,
                    "planned_qty_kg": round(planned_kg, 3),
                    "machine_brand": brand,
                    "machine_diameter_in": mc_dia,
                    "machine_gauge_e": mc_gauge,
                    "yarn_count_tex_1": tex_1,
                    "stitch_length_mm": round(stitch_mm, 3),
                    "target_gsm": round(gsm, 1),
                    "number_of_feeders": feeders,
                    "waste_qty_kg": waste_kg,
                }
            )
    return pd.DataFrame(rows)


def generate_synthetic_knitting_prediction_frame(
    *,
    num_runs: int = 6,
    seed: int = 77,
    start_index: int = 1,
) -> pd.DataFrame:
    """Generate prediction-only knitting shift runs (without target or post-shift totals)."""
    df = generate_synthetic_knitting_training_frame(
        num_lots=max(2, num_runs),
        runs_per_lot=1,
        seed=seed,
    ).iloc[:num_runs].copy()
    df["run_id"] = [f"KN-PRED-{start_index + idx:04d}" for idx in range(len(df))]
    df["order_lot_id"] = [f"LOT-PRED-{start_index + idx:04d}" for idx in range(len(df))]
    return df.drop(columns=["waste_qty_kg"]).reset_index(drop=True)
