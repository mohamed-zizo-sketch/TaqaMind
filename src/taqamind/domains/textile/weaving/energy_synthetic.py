"""Physics-informed synthetic Weaving Energy generator (ENGINEERING VERIFICATION ONLY)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def _build_weaving_energy_base(
    *,
    num_orders: int,
    batches_per_order: int,
    seed: int,
    include_target: bool,
    id_prefix: str = "",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    loom_types = ("air_jet", "rapier", "water_jet")

    for o_idx in range(num_orders):
        order_id = f"WEO-{o_idx + 1:04d}"
        sched_date = (start + timedelta(days=o_idx)).isoformat()
        for b_idx in range(batches_per_order):
            loom_type = str(loom_types[(o_idx + b_idx) % len(loom_types)])
            machine_id = f"LOOM-{((o_idx + b_idx) % 6) + 1:02d}"
            rated_kw = {"air_jet": 5.5, "rapier": 4.0, "water_jet": 3.0}[loom_type]
            width_cm = float(rng.choice([150.0, 190.0, 220.0, 280.0]))
            beam_m = float(np.round(rng.uniform(400.0, 2200.0), 1))
            speed_rpm = {"air_jet": 750.0, "rapier": 480.0, "water_jet": 650.0}[loom_type]
            speed_rpm += float(rng.normal(0.0, 25.0))
            util_pct = float(np.clip(rng.normal(84.0, 5.0), 60.0, 98.0))
            duration_hr = float(np.round(rng.uniform(4.0, 14.0), 2))
            air_bar = (
                float(np.round(rng.uniform(4.5, 6.2), 2))
                if loom_type == "air_jet"
                else 0.0
            )
            warp_ne = float(rng.choice([20.0, 30.0, 40.0]))
            weft_ne = float(rng.choice([20.0, 30.0, 40.0]))
            epi = float(rng.integers(60, 125))
            ppi = float(rng.integers(48, 96))
            temp_c = float(np.round(rng.uniform(22.0, 31.0), 1))
            rh_pct = float(np.round(rng.uniform(58.0, 76.0), 1))

            row: dict[str, object] = {
                "batch_record_id": f"{id_prefix}WEB-{o_idx + 1:04d}-{b_idx + 1:02d}",
                "order_id": f"{id_prefix}{order_id}",
                "scheduled_date": sched_date,
                "machine_id": machine_id,
                "loom_type": loom_type,
                "machine_rated_power_kw": rated_kw,
                "planned_beam_length_m": beam_m,
                "planned_operating_duration_hr": duration_hr,
                "fabric_width_cm": width_cm,
                "planned_loom_speed_rpm": round(speed_rpm, 1),
                "planned_utilization_pct": round(util_pct, 2),
                "compressed_air_pressure_bar": air_bar,
                "warp_count_ne": warp_ne,
                "weft_count_ne": weft_ne,
                "warp_ends_per_inch": epi,
                "weft_picks_per_inch": ppi,
                "ambient_temp_c": temp_c,
                "relative_humidity_pct": rh_pct,
            }
            if include_target:
                # Physics-grounded signal: motor load + width/speed + air-jet compressor load
                load_factor = 0.62 + 0.0004 * (width_cm - 150.0) + 0.0002 * (ppi - 48.0)
                drive_kwh = rated_kw * duration_hr * (util_pct / 100.0) * load_factor
                air_kwh = (
                    (0.18 * air_bar * (width_cm / 190.0) * (speed_rpm / 700.0) * duration_hr)
                    if loom_type == "air_jet"
                    else 0.0
                )
                noise = float(rng.normal(0.0, 0.35))
                row["total_electrical_energy_kwh"] = round(max(0.5, drive_kwh + air_kwh + noise), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def generate_synthetic_weaving_energy_training_frame(
    *,
    num_orders: int = 24,
    batches_per_order: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate physics-calibrated Weaving Energy training data (verification only)."""
    return _build_weaving_energy_base(
        num_orders=num_orders,
        batches_per_order=batches_per_order,
        seed=seed,
        include_target=True,
    )


def generate_synthetic_weaving_energy_prediction_frame(
    *,
    num_batches: int = 6,
    seed: int = 99,
) -> pd.DataFrame:
    """Generate Weaving Energy prediction data strictly WITHOUT the target column."""
    return _build_weaving_energy_base(
        num_orders=max(1, num_batches // 2),
        batches_per_order=2,
        seed=seed,
        include_target=False,
        id_prefix="PRED_",
    ).iloc[:num_batches].reset_index(drop=True)
