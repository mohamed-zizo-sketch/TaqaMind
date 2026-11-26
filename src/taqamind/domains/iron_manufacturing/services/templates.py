"""Customer CSV template, contract YAML, and feature dictionary generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    FeatureOrigin,
)
from taqamind.core.interfaces.serialization import serialize_contract
from taqamind.domains.iron_manufacturing.common.energy_synthetic import (
    generate_synthetic_iron_energy_dataset,
)
from taqamind.domains.iron_manufacturing.common.normalization import safe_export_csv
from taqamind.domains.iron_manufacturing.common.synthetic import generate_synthetic_dataset
from taqamind.domains.iron_manufacturing.domain import (
    PROCESS_DEFAULT_TASKS,
    build_iron_manufacturing_domain,
    get_iron_dataset_contract,
)
from taqamind.domains.iron_manufacturing.energy_contracts import (
    get_iron_energy_dataset_contract,
)


def _raw_columns_for_purpose(contract: DatasetContract, purpose: DatasetPurpose) -> list[str]:
    """Return ordered raw (non-derived) columns expected in a customer CSV template."""
    cols: list[str] = []
    for feat in contract.features:
        if feat.origin == FeatureOrigin.DERIVED:
            continue
        if purpose == DatasetPurpose.PREDICTION and not contract.is_available_at_prediction(feat):
            continue
        cols.append(feat.name)
    if purpose == DatasetPurpose.TRAINING:
        cols.append(contract.target.feature.name)
    return cols


def build_feature_dictionary_records(contract: DatasetContract) -> list[dict[str, Any]]:
    """Build structured feature dictionary entries for a DatasetContract."""
    records: list[dict[str, Any]] = []
    for col in contract.all_columns:
        records.append(
            {
                "canonical_name": col.name,
                "display_name": col.display_name,
                "description": col.description,
                "process": contract.process,
                "granularity": contract.granularity,
                "data_type": col.data_type.value,
                "role": col.role.value,
                "availability": col.availability.value,
                "origin": col.origin.value,
                "unit": col.unit,
                "required": col.required,
                "nullable": col.nullable,
                "missing_policy": col.missing_policy.value,
                "aliases": list(col.aliases),
                "source_system": col.source_system,
                "derived_from": list(col.derived_from),
                "constraints": col.constraints.model_dump(mode="json"),
                "is_business_key": col.name in contract.business_keys,
                "is_grouping_column": col.name in contract.grouping_columns,
                "is_time_column": col.name == contract.time_column,
            }
        )
    return records


def _write_process_template_bundle(
    *,
    process_id: str,
    contract: DatasetContract,
    sample_hist: Any,
    sample_pred: Any,
    output_dir: Path | str,
    filename_prefix: str,
) -> dict[str, Path]:
    """Write historical CSV, prediction CSV, feature dictionary, and mapping example."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_cols = _raw_columns_for_purpose(contract, DatasetPurpose.TRAINING)
    pred_cols = _raw_columns_for_purpose(contract, DatasetPurpose.PREDICTION)

    hist_df = sample_hist.reindex(columns=hist_cols)
    pred_df = sample_pred.reindex(columns=pred_cols)

    hist_path = safe_export_csv(
        hist_df, out_dir / f"{filename_prefix}_historical_training_template.csv"
    )
    pred_path = safe_export_csv(
        pred_df, out_dir / f"{filename_prefix}_prediction_only_template.csv"
    )

    dict_path = out_dir / f"{filename_prefix}_feature_dictionary.json"
    dict_path.write_text(
        json.dumps(build_feature_dictionary_records(contract), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mapping_example = {
        "process_id": process_id,
        "dataset_id": contract.dataset_id,
        "contract_version": contract.version,
        "example_source_to_canonical_mappings": {
            (col.aliases[0] if col.aliases else col.name): {
                "canonical_column": col.name,
                "source_unit": col.unit,
                "canonical_unit": col.unit,
                "role": col.role.value,
                "availability": col.availability.value,
            }
            for col in contract.all_columns
            if col.origin != FeatureOrigin.DERIVED
        },
    }
    mapping_path = out_dir / f"{filename_prefix}_mapping_example.yaml"
    mapping_path.write_text(
        yaml.safe_dump(mapping_example, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return {
        "historical_template": hist_path,
        "prediction_template": pred_path,
        "feature_dictionary": dict_path,
        "mapping_example": mapping_path,
    }


def generate_process_templates(
    process_id: str,
    output_dir: Path | str,
    *,
    sample_rows: int = 3,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate V1 waste historical CSV, prediction CSV, and schema artifacts."""
    contract = get_iron_dataset_contract(process_id)
    sample_hist = generate_synthetic_dataset(
        process_id, n_rows=sample_rows, seed=seed, include_target=True
    )
    sample_pred = generate_synthetic_dataset(
        process_id, n_rows=sample_rows, seed=seed + 100, include_target=False
    )
    return _write_process_template_bundle(
        process_id=process_id,
        contract=contract,
        sample_hist=sample_hist,
        sample_pred=sample_pred,
        output_dir=output_dir,
        filename_prefix=process_id,
    )


def generate_energy_process_templates(
    process_id: str,
    output_dir: Path | str,
    *,
    sample_rows: int = 3,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate V2 electrical energy historical CSV, prediction CSV, and schema artifacts."""
    contract = get_iron_energy_dataset_contract(process_id)
    sample_hist = generate_synthetic_iron_energy_dataset(
        process_id, n_rows=sample_rows, seed=seed, include_target=True
    )
    sample_pred = generate_synthetic_iron_energy_dataset(
        process_id, n_rows=sample_rows, seed=seed + 100, include_target=False
    )
    return _write_process_template_bundle(
        process_id=process_id,
        contract=contract,
        sample_hist=sample_hist,
        sample_pred=sample_pred,
        output_dir=output_dir,
        filename_prefix=f"{process_id}_energy",
    )


def generate_all_contracts_and_templates(
    contracts_root: Path | str = Path("contracts/domains/iron_manufacturing"),
) -> dict[str, Any]:
    """Export the full DomainContract YAML and templates for all Iron Manufacturing processes."""
    root = Path(contracts_root)
    templates_dir = root / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    # Local .gitignore override so tracked template CSVs are allowed while raw data stays ignored.
    local_gitignore = templates_dir / ".gitignore"
    local_gitignore.write_text("!*_template.csv\n", encoding="utf-8")

    domain = build_iron_manufacturing_domain()
    domain_yaml_path = root / "domain_contract_v1.yaml"
    domain_yaml_path.write_text(
        yaml.safe_dump(serialize_contract(domain), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    generated: dict[str, Any] = {"domain_contract": str(domain_yaml_path), "processes": {}}
    for process_id in PROCESS_DEFAULT_TASKS:
        files = generate_process_templates(process_id, templates_dir)
        generated["processes"][process_id] = {k: str(v) for k, v in files.items()}

    return generated
