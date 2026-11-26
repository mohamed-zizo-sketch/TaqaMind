"""Customer data templates and YAML/JSON dictionary export for textile domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from taqamind.core.contracts import DatasetContract, DatasetPurpose, FeatureOrigin
from taqamind.domains.textile.knitting.contracts import build_knitting_dataset_contract
from taqamind.domains.textile.knitting.synthetic import (
    generate_synthetic_knitting_prediction_frame,
    generate_synthetic_knitting_training_frame,
)
from taqamind.domains.textile.weaving.contracts import build_weaving_dataset_contract
from taqamind.domains.textile.weaving.synthetic import (
    generate_synthetic_weaving_prediction_frame,
    generate_synthetic_weaving_training_frame,
)


def contract_to_dictionary_payload(contract: DatasetContract) -> dict[str, Any]:
    """Serialize contract field rules into a customer-facing data dictionary."""
    entries = []
    for feature in contract.all_columns:
        entries.append(
            {
                "name": feature.name,
                "display_name": feature.display_name,
                "role": feature.role.value,
                "data_type": feature.data_type.value,
                "unit": feature.unit,
                "availability": feature.availability.value,
                "origin": feature.origin.value,
                "required_for_training": (
                    feature.name in contract.required_columns(DatasetPurpose.TRAINING)
                ),
                "required_for_prediction": (
                    feature.name in contract.required_columns(DatasetPurpose.PREDICTION)
                ),
                "nullable": feature.nullable,
                "missing_policy": feature.missing_policy.value,
                "aliases": list(feature.aliases),
                "derived_from": list(feature.derived_from),
                "constraints": feature.constraints.model_dump(mode="json", exclude_none=True),
                "description": feature.description,
            }
        )
    return {
        "domain": contract.domain,
        "process": contract.process,
        "dataset_id": contract.dataset_id,
        "version": contract.version,
        "granularity": contract.granularity,
        "business_keys": list(contract.business_keys),
        "grouping_columns": list(contract.grouping_columns),
        "time_column": contract.time_column,
        "target": {
            "name": contract.target.feature.name,
            "unit": contract.target.feature.unit,
            "kind": contract.target.target_kind.value,
            "confirmation_status": contract.target.confirmation.status.value,
            "derived_rate": (
                contract.target.derived_rate.model_dump(mode="json")
                if contract.target.derived_rate is not None
                else None
            ),
        },
        "columns": entries,
    }


def export_textile_templates(output_dir: str | Path) -> dict[str, list[str]]:
    """Export weaving & knitting training/prediction CSV templates and JSON dictionaries."""
    base = Path(output_dir)
    created: dict[str, list[str]] = {"weaving": [], "knitting": []}

    specs = (
        (
            "weaving",
            build_weaving_dataset_contract(),
            generate_synthetic_weaving_training_frame(num_orders=4, batches_per_order=2),
            generate_synthetic_weaving_prediction_frame(num_batches=4),
        ),
        (
            "knitting",
            build_knitting_dataset_contract(),
            generate_synthetic_knitting_training_frame(num_lots=4, runs_per_lot=2),
            generate_synthetic_knitting_prediction_frame(num_runs=4),
        ),
    )

    for process_name, contract, train_df, pred_df in specs:
        proc_dir = base / process_name
        proc_dir.mkdir(parents=True, exist_ok=True)

        # Exclude derived features from raw customer input templates (computed in Core).
        raw_train_cols = [
            f.name
            for f in contract.all_columns
            if f.origin != FeatureOrigin.DERIVED and f.name in train_df.columns
        ]
        raw_pred_cols = [
            f.name
            for f in contract.features
            if f.origin != FeatureOrigin.DERIVED and f.name in pred_df.columns
        ]

        train_path = proc_dir / f"{process_name}_historical_training_template.csv"
        pred_path = proc_dir / f"{process_name}_prediction_only_template.csv"
        dict_path = proc_dir / f"{process_name}_data_dictionary.json"

        train_df[raw_train_cols].to_csv(train_path, index=False)
        pred_df[raw_pred_cols].to_csv(pred_path, index=False)
        dict_path.write_text(
            json.dumps(contract_to_dictionary_payload(contract), indent=2),
            encoding="utf-8",
        )
        created[process_name].extend(
            [str(train_path), str(pred_path), str(dict_path)]
        )

    return created
