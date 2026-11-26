"""Physics-informed synthetic Knitting Energy generator (ENGINEERING VERIFICATION ONLY)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def _build_knitting_energy_base(
    *,
    num_lots: int,
    runs_per_lot: int,
    seed: int,
    include_target: bool,
    id_prefix: str = "",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    machine_types = ("circular", "flat")

    for l_idx in range(num_lots):
        lot_id = f"KEL-{l_idx + 1:04d}"
        prod_date = (start + timedelta(days=l_idx)).isoformat()
        for r_idx in range(runs_per_lot):
            m_type = str(machine_types[(l_idx + r_idx) % len(machine_types)])
            machine_id = f"KNIT-{((l_idx + r_idx) % 6) + 1:02d}"
            rated_kw = 5.5 if m_type == "circular" else 2.5
            dia_in = float(rng.choice([30.0, 34.0])) if m_type == "circular" else 48.0
            gauge_e = float(rng.choice([20.0, 24.0, 28.0, 32.0]))
            feeders = int(rng.choice([72, 96, 120])) if m_type == "circular" else 4
            speed_rpm = float(rng.uniform(22.0, 34.0)) if m_type == "circular" else 18.0
            util_pct = float(np.clip(rng.normal(86.0, 4.5), 65.0, 98.0))
            duration_hr = float(np.round(rng.uniform(4.0, 12.0), 2))
            qty_kg = float(np.round(duration_hr * (18.0 if m_type == "circular" else 4.5), 2))
            tex = float(rng.choice([19.7, 24.6, 29.5]))
            stitch_mm = float(np.round(rng.uniform(2.5, 3.4), 2))
            gsm = float(rng.choice([140.0, 180.0, 220.0, 260.0]))
            temp_c = float(np.round(rng.uniform(22.0, 30.0), 1))

            row: dict[str, object] = {
                "run_id": f"{id_prefix}KER-{l_idx + 1:04d}-{r_idx + 1:02d}",
                "order_lot_id": f"{id_prefix}{lot_id}",
                "production_date": prod_date,
                "machine_id": machine_id,
                "knitting_machine_type": m_type,
                "machine_rated_power_kw": rated_kw,
                "planned_qty_kg": qty_kg,
                "planned_operating_duration_hr": duration_hr,
                "machine_diameter_in": dia_in,
                "machine_gauge_e": gauge_e,
                "number_of_feeders": feeders,
                "planned_machine_speed_rpm": round(speed_rpm, 1),
                "planned_utilization_pct": round(util_pct, 2),
                "yarn_count_tex_1": tex,
                "stitch_length_mm": stitch_mm,
                "target_gsm": gsm,
                "ambient_temp_c": temp_c,
            }
            if include_target:
                # Physics-grounded signal: drive load scaled by gauge friction, speed & feeders
                gauge_factor = 0.65 + 0.008 * (gauge_e - 20.0) + 0.004 * (speed_rpm - 20.0)
                base_kwh = rated_kw * duration_hr * (util_pct / 100.0) * gauge_factor
                noise = float(rng.normal(0.0, 0.25))
                row["total_electrical_energy_kwh"] = round(max(0.4, base_kwh + noise), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def generate_synthetic_knitting_energy_training_frame(
    *,
    num_lots: int = 30,
    runs_per_lot: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate physics-calibrated Knitting Energy training data (verification only)."""
    return _build_knitting_energy_base(
        num_lots=num_lots,
        runs_per_lot=runs_per_lot,
        seed=seed,
        include_target=True,
    )


def generate_synthetic_knitting_energy_prediction_frame(
    *,
    num_runs: int = 6,
    seed: int = 77,
) -> pd.DataFrame:
    """Generate Knitting Energy prediction data strictly WITHOUT the target column."""
    return _build_knitting_energy_base(
        num_lots=max(1, num_runs // 2),
        runs_per_lot=2,
        seed=seed,
        include_target=False,
        id_prefix="PRED_",
    ).iloc[:num_runs].reset_index(drop=True)
