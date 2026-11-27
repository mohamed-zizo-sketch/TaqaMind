"""Physics-calibrated synthetic Energy generators for Food & Beverage (VERIFICATION ONLY)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def _build_bakery_energy_base(
    *,
    num_lines: int,
    batches_per_line: int,
    seed: int,
    include_target: bool,
    id_prefix: str = "",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    categories = ("baladi_bread", "toast_loaf", "biscuits")
    heating_types = ("electric", "gas_fired")

    for l_idx in range(num_lines):
        line_id = f"{id_prefix}BAK-LINE-{l_idx + 1:04d}"
        prod_date = (start + timedelta(days=l_idx)).isoformat()
        for b_idx in range(batches_per_line):
            category = str(categories[(l_idx + b_idx) % len(categories)])
            heating = str(heating_types[(l_idx + b_idx) % len(heating_types)])
            machine_id = f"OVEN-{((l_idx + b_idx) % 4) + 1:02d}"
            rated_kw = 95.0 if heating == "electric" else 28.0
            qty_kg = float(np.round(rng.uniform(400.0, 2200.0), 1))
            temp_c = float(np.round(rng.uniform(180.0, 260.0), 1))
            duration_hr = float(np.round(rng.uniform(2.0, 8.0), 2))
            util_pct = float(np.clip(rng.normal(82.0, 6.0), 55.0, 98.0))
            ambient_c = float(np.round(rng.uniform(22.0, 36.0), 1))

            row: dict[str, object] = {
                "batch_id": f"{id_prefix}BAK-B-{l_idx + 1:04d}-{b_idx + 1:02d}",
                "production_line_id": line_id,
                "production_timestamp": prod_date,
                "machine_id": machine_id,
                "product_category": category,
                "oven_heating_type": heating,
                "planned_batch_quantity_kg": qty_kg,
                "machine_rated_power_kw": rated_kw,
                "planned_operating_duration_hr": duration_hr,
                "planned_utilization_pct": round(util_pct, 2),
                "ambient_temp_c": ambient_c,
                "planned_processing_temp_c": temp_c,
            }
            if include_target:
                load_factor = (
                    0.66
                    + 0.00008 * (qty_kg - 400.0)
                    + 0.0012 * (temp_c - 180.0)
                    + 0.003 * max(0.0, ambient_c - 25.0)
                )
                base_kwh = rated_kw * duration_hr * (util_pct / 100.0) * load_factor
                noise = float(rng.normal(0.0, 0.8))
                row["total_electrical_energy_kwh"] = round(max(1.0, base_kwh + noise), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_dairy_energy_base(
    *,
    num_lines: int,
    batches_per_line: int,
    seed: int,
    include_target: bool,
    id_prefix: str = "",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    categories = ("fluid_milk", "yogurt", "white_cheese")
    stages = ("pasteurization_chiller", "uht_line", "cold_storage")

    for l_idx in range(num_lines):
        line_id = f"{id_prefix}DRY-LINE-{l_idx + 1:04d}"
        prod_date = (start + timedelta(days=l_idx)).isoformat()
        for b_idx in range(batches_per_line):
            category = str(categories[(l_idx + b_idx) % len(categories)])
            stage = str(stages[(l_idx + b_idx) % len(stages)])
            machine_id = f"CHILL-{((l_idx + b_idx) % 4) + 1:02d}"
            rated_kw = {
                "pasteurization_chiller": 65.0,
                "uht_line": 88.0,
                "cold_storage": 45.0,
            }[stage]
            qty_kg = float(np.round(rng.uniform(1000.0, 8000.0), 1))
            temp_c = float(np.round(rng.uniform(4.0, 138.0), 1))
            duration_hr = float(np.round(rng.uniform(3.0, 10.0), 2))
            util_pct = float(np.clip(rng.normal(85.0, 5.0), 60.0, 98.0))
            ambient_c = float(np.round(rng.uniform(20.0, 38.0), 1))

            row: dict[str, object] = {
                "batch_id": f"{id_prefix}DRY-B-{l_idx + 1:04d}-{b_idx + 1:02d}",
                "production_line_id": line_id,
                "production_timestamp": prod_date,
                "machine_id": machine_id,
                "product_category": category,
                "chiller_stage_type": stage,
                "planned_batch_quantity_kg": qty_kg,
                "machine_rated_power_kw": rated_kw,
                "planned_operating_duration_hr": duration_hr,
                "planned_utilization_pct": round(util_pct, 2),
                "ambient_temp_c": ambient_c,
                "planned_pasteurization_temp_c": temp_c,
            }
            if include_target:
                chiller_cop_penalty = 0.008 * max(0.0, ambient_c - 22.0)
                load_factor = 0.68 + 0.00003 * (qty_kg - 1000.0) + chiller_cop_penalty
                base_kwh = rated_kw * duration_hr * (util_pct / 100.0) * load_factor
                noise = float(rng.normal(0.0, 1.0))
                row["total_electrical_energy_kwh"] = round(max(1.0, base_kwh + noise), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_beverage_energy_base(
    *,
    num_lines: int,
    batches_per_line: int,
    seed: int,
    include_target: bool,
    id_prefix: str = "",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 1)
    rows: list[dict[str, object]] = []
    categories = ("carbonated_soft_drink", "bottled_water", "fruit_juice")
    line_types = ("pet_bottling", "canning", "glass_bottling")

    for l_idx in range(num_lines):
        line_id = f"{id_prefix}BEV-LINE-{l_idx + 1:04d}"
        prod_date = (start + timedelta(days=l_idx)).isoformat()
        for b_idx in range(batches_per_line):
            category = str(categories[(l_idx + b_idx) % len(categories)])
            line_type = str(line_types[(l_idx + b_idx) % len(line_types)])
            machine_id = f"FILL-{((l_idx + b_idx) % 4) + 1:02d}"
            rated_kw = {
                "pet_bottling": 115.0,
                "canning": 75.0,
                "glass_bottling": 90.0,
            }[line_type]
            qty_kg = float(np.round(rng.uniform(2000.0, 15000.0), 1))
            speed_bph = float(np.round(rng.uniform(6000.0, 24000.0), 0))
            duration_hr = float(np.round(rng.uniform(3.0, 12.0), 2))
            util_pct = float(np.clip(rng.normal(83.0, 5.5), 58.0, 98.0))
            ambient_c = float(np.round(rng.uniform(22.0, 38.0), 1))

            row: dict[str, object] = {
                "batch_id": f"{id_prefix}BEV-B-{l_idx + 1:04d}-{b_idx + 1:02d}",
                "production_line_id": line_id,
                "production_timestamp": prod_date,
                "machine_id": machine_id,
                "product_category": category,
                "packaging_line_type": line_type,
                "planned_batch_quantity_kg": qty_kg,
                "machine_rated_power_kw": rated_kw,
                "planned_operating_duration_hr": duration_hr,
                "planned_utilization_pct": round(util_pct, 2),
                "ambient_temp_c": ambient_c,
                "planned_line_speed_bph": speed_bph,
            }
            if include_target:
                blow_air = 12.0 * duration_hr if line_type == "pet_bottling" else 0.0
                load_factor = (
                    0.64
                    + 0.000012 * (speed_bph - 6000.0)
                    + 0.004 * max(0.0, ambient_c - 24.0)
                )
                base_kwh = rated_kw * duration_hr * (util_pct / 100.0) * load_factor + blow_air
                noise = float(rng.normal(0.0, 1.2))
                row["total_electrical_energy_kwh"] = round(max(1.0, base_kwh + noise), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def generate_synthetic_bakery_energy_training_frame(
    *, num_lines: int = 20, batches_per_line: int = 3, seed: int = 42
) -> pd.DataFrame:
    """Generate physics-calibrated Bakery Energy training data (verification only)."""
    return _build_bakery_energy_base(
        num_lines=num_lines, batches_per_line=batches_per_line, seed=seed, include_target=True
    )


def generate_synthetic_bakery_energy_prediction_frame(
    *, num_batches: int = 6, seed: int = 99
) -> pd.DataFrame:
    """Generate physics-calibrated Bakery Energy prediction data without target column."""
    return _build_bakery_energy_base(
        num_lines=num_batches,
        batches_per_line=1,
        seed=seed,
        include_target=False,
        id_prefix="PRED-",
    )


def generate_synthetic_dairy_energy_training_frame(
    *, num_lines: int = 20, batches_per_line: int = 3, seed: int = 42
) -> pd.DataFrame:
    """Generate physics-calibrated Dairy Energy training data (verification only)."""
    return _build_dairy_energy_base(
        num_lines=num_lines, batches_per_line=batches_per_line, seed=seed, include_target=True
    )


def generate_synthetic_dairy_energy_prediction_frame(
    *, num_batches: int = 6, seed: int = 99
) -> pd.DataFrame:
    """Generate physics-calibrated Dairy Energy prediction data without target column."""
    return _build_dairy_energy_base(
        num_lines=num_batches,
        batches_per_line=1,
        seed=seed,
        include_target=False,
        id_prefix="PRED-",
    )


def generate_synthetic_beverage_energy_training_frame(
    *, num_lines: int = 20, batches_per_line: int = 3, seed: int = 42
) -> pd.DataFrame:
    """Generate physics-calibrated Beverage Energy training data (verification only)."""
    return _build_beverage_energy_base(
        num_lines=num_lines, batches_per_line=batches_per_line, seed=seed, include_target=True
    )


def generate_synthetic_beverage_energy_prediction_frame(
    *, num_batches: int = 6, seed: int = 99
) -> pd.DataFrame:
    """Generate physics-calibrated Beverage Energy prediction data without target column."""
    return _build_beverage_energy_base(
        num_lines=num_batches,
        batches_per_line=1,
        seed=seed,
        include_target=False,
        id_prefix="PRED-",
    )
