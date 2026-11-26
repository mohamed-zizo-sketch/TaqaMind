"""Customer CSV template generator for Food & Beverage processes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    DataType,
    FeatureRole,
)

SAMPLE_VALUES: dict[str, object] = {
    "factory_id": "eg_cairo_bakery_01",
    "batch_id": "BATCH-2026-0001",
    "production_timestamp": "2026-11-25T06:00:00Z",
    "production_line_id": "LINE_01",
    "product_category": "standard_product",
    "planned_batch_quantity_kg": 1000.0,
    "planned_raw_material_kg": 1080.0,
    "planned_batch_volume_l": 2500.0,
    "planned_processing_temp_c": 185.0,
    "planned_processing_duration_min": 30.0,
    "planned_shift": "morning",
    "target_ph_level": 4.5,
    "waste_quantity_kg": 18.5,
    "waste_quantity_l": 24.0,
}


def generate_csv_template(
    contract: DatasetContract,
    purpose: DatasetPurpose,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate a training or prediction CSV template from a DatasetContract."""

    if purpose == DatasetPurpose.TRAINING:
        columns = [
            f
            for f in contract.all_columns
            if f.role != FeatureRole.METADATA
            and (f.role == FeatureRole.TARGET or contract.is_available_at_prediction(f))
        ]
    elif purpose == DatasetPurpose.PREDICTION:
        columns = [
            f
            for f in contract.features
            if f.role in {FeatureRole.IDENTIFIER, FeatureRole.TIMESTAMP, FeatureRole.PREDICTOR}
            and contract.is_available_at_prediction(f)
        ]
    else:
        raise ValueError(f"Template generation only supports TRAINING or PREDICTION, got {purpose}")

    row: dict[str, object] = {}
    for col in columns:
        if col.name in SAMPLE_VALUES:
            row[col.name] = SAMPLE_VALUES[col.name]
        elif col.data_type in {DataType.FLOAT, DataType.INTEGER}:
            row[col.name] = 1.0 if col.data_type == DataType.FLOAT else 1
        elif col.data_type == DataType.DATETIME:
            row[col.name] = "2026-11-25T06:00:00Z"
        else:
            row[col.name] = "sample_value"

    df = pd.DataFrame([row])
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df
