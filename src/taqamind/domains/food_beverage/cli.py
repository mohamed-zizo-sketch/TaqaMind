"""Command-line interface for the Food & Beverage industrial domain pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import UUID

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import TrainingMode
from taqamind.domains.food_beverage.config import (
    SUPPORTED_PROCESSES,
    FoodBeverageStorageConfig,
)
from taqamind.domains.food_beverage.domain import make_synthetic_target_confirmation
from taqamind.domains.food_beverage.pipeline import FoodBeveragePipeline
from taqamind.domains.food_beverage.services.templates import generate_csv_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m taqamind.domains.food_beverage.cli",
        description="TaqaMind Food & Beverage Industrial ML Pipeline CLI",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("artifacts/food_beverage"),
        help="Root directory for factory-scoped artifacts and registries.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Ingest and profile a CSV dataset.")
    p_inspect.add_argument("--csv", type=Path, required=True)
    p_inspect.add_argument("--factory-id", type=UUID, required=True)
    p_inspect.add_argument("--process", choices=SUPPORTED_PROCESSES, required=True)

    # template
    p_tpl = sub.add_parser("template", help="Generate a training or prediction CSV template.")
    p_tpl.add_argument("--process", choices=SUPPORTED_PROCESSES, required=True)
    p_tpl.add_argument("--purpose", choices=["training", "prediction"], required=True)
    p_tpl.add_argument("--output", type=Path, required=True)

    # validate
    p_val = sub.add_parser("validate", help="Run mapping and semantic validation on a CSV.")
    p_val.add_argument("--csv", type=Path, required=True)
    p_val.add_argument("--factory-id", type=UUID, required=True)
    p_val.add_argument("--process", choices=SUPPORTED_PROCESSES, required=True)
    p_val.add_argument("--purpose", choices=["training", "prediction"], default="training")

    # train
    p_train = sub.add_parser("train", help="Run initial factory training and protected evaluation.")
    p_train.add_argument("--csv", type=Path, required=True)
    p_train.add_argument("--factory-id", type=UUID, required=True)
    p_train.add_argument("--process", choices=SUPPORTED_PROCESSES, required=True)
    p_train.add_argument(
        "--synthetic-confirm",
        action="store_true",
        help="Use explicit synthetic target confirmation (for synthetic/testing runs only).",
    )

    # retrain
    p_retrain = sub.add_parser("retrain", help="Run explicit retraining on updated factory data.")
    p_retrain.add_argument("--csv", type=Path, required=True)
    p_retrain.add_argument("--factory-id", type=UUID, required=True)
    p_retrain.add_argument("--process", choices=SUPPORTED_PROCESSES, required=True)
    p_retrain.add_argument("--synthetic-confirm", action="store_true")

    # predict
    p_pred = sub.add_parser("predict", help="Run inference-only prediction using active model.")
    p_pred.add_argument("--csv", type=Path, required=True)
    p_pred.add_argument("--factory-id", type=UUID, required=True)
    p_pred.add_argument("--process", choices=SUPPORTED_PROCESSES, required=True)
    p_pred.add_argument("--output", type=Path, default=None)

    # model-info
    p_info = sub.add_parser("model-info", help="Show active and registered models for a factory.")
    p_info.add_argument("--factory-id", type=UUID, required=True)
    p_info.add_argument("--process", choices=SUPPORTED_PROCESSES, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pipe = FoodBeveragePipeline(storage_config=FoodBeverageStorageConfig(root_dir=args.storage_dir))

    if args.command == "template":
        purpose = (
            DatasetPurpose.TRAINING if args.purpose == "training" else DatasetPurpose.PREDICTION
        )
        contract = pipe.get_contract(args.process)
        df = generate_csv_template(contract, purpose, output_path=args.output)
        print(json.dumps({"status": "ok", "output": str(args.output), "columns": list(df.columns)}))
        return 0

    if args.command == "inspect":
        prep = pipe.prepare_dataset(
            csv_path=args.csv,
            factory_id=args.factory_id,
            process_id=args.process,
            purpose=DatasetPurpose.RESEARCH,
        )
        prof = prep.profiling.profile
        print(
            json.dumps(
                {
                    "status": "ok",
                    "rows": prof.row_count if prof else 0,
                    "columns": [c.name for c in prof.columns] if prof else [],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate":
        purpose = (
            DatasetPurpose.TRAINING if args.purpose == "training" else DatasetPurpose.PREDICTION
        )
        prep = pipe.prepare_dataset(
            csv_path=args.csv,
            factory_id=args.factory_id,
            process_id=args.process,
            purpose=purpose,
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "rows": prep.engineered_artifact.metadata.row_count,
                    "revision_id": str(prep.engineered_artifact.metadata.revision_id),
                },
                indent=2,
            )
        )
        return 0

    if args.command in {"train", "retrain"}:
        conf = make_synthetic_target_confirmation() if args.synthetic_confirm else None
        intent = TrainingMode.RETRAINING if args.command == "retrain" else TrainingMode.INITIAL
        res = pipe.train_model(
            csv_path=args.csv,
            factory_id=args.factory_id,
            process_id=args.process,
            target_confirmation=conf,
            intent=intent,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "decision": res.evaluation.decision.value,
                    "relative_improvement_pct": res.evaluation.test_improvement_pct,
                    "model_version_id": str(res.registry_entry.version.model_version_id)
                    if res.registry_entry
                    else None,
                    "model_status": res.registry_entry.status.value if res.registry_entry else None,
                    "reused_existing": res.reused_existing,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "predict":
        batch = pipe.predict(
            csv_path=args.csv,
            factory_id=args.factory_id,
            process_id=args.process,
        )
        payload = batch.model_dump(mode="json")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": batch.status.value,
                    "model_version_id": str(batch.model.model_version_id),
                    "record_count": len(batch.predictions),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "model-info":
        info = pipe.get_model_info(factory_id=args.factory_id, process_id=args.process)
        print(json.dumps(info, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
