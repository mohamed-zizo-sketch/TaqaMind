"""Command-line interface for the TaqaMind Textile domain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import ModelScope, TrainingMode
from taqamind.domains.textile.pipeline import TextilePipelineService, resolve_process_spec
from taqamind.domains.textile.templates import export_textile_templates


def _parse_uuid(value: str) -> UUID:
    return UUID(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taqamind.domains.textile.cli",
        description="TaqaMind Textile Manufacturing (Weaving & Knitting) ML CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect textile process contract & features.")
    p_inspect.add_argument("--process", choices=["weaving", "knitting"], required=True)

    # template
    p_tpl = sub.add_parser("template", help="Export customer CSV templates & data dictionaries.")
    p_tpl.add_argument(
        "--output-dir",
        type=Path,
        default=Path("contracts/domains/textile/templates"),
        help="Destination directory for templates.",
    )

    # validate
    p_val = sub.add_parser("validate", help="Ingest, map, clean, and validate a CSV.")
    p_val.add_argument("--process", choices=["weaving", "knitting"], required=True)
    p_val.add_argument("--factory-id", type=_parse_uuid, required=True)
    p_val.add_argument("--csv", type=Path, required=True)
    p_val.add_argument(
        "--purpose",
        choices=["training", "prediction"],
        default="prediction",
    )
    p_val.add_argument("--storage-dir", type=Path, default=Path("artifacts/textile/store"))
    p_val.add_argument("--synthetic-confirm", action="store_true")

    # train
    p_train = sub.add_parser("train", help="Run full training, protected eval, and activation.")
    p_train.add_argument("--process", choices=["weaving", "knitting"], required=True)
    p_train.add_argument("--factory-id", type=_parse_uuid, required=True)
    p_train.add_argument("--csv", type=Path, required=True)
    p_train.add_argument("--storage-dir", type=Path, default=Path("artifacts/textile/store"))
    p_train.add_argument(
        "--synthetic-confirm",
        action="store_true",
        help="Use synthetic target confirmation (tests/demos only).",
    )

    # retrain
    p_retrain = sub.add_parser("retrain", help="Explicitly retrain from a prior model version.")
    p_retrain.add_argument("--process", choices=["weaving", "knitting"], required=True)
    p_retrain.add_argument("--factory-id", type=_parse_uuid, required=True)
    p_retrain.add_argument("--csv", type=Path, required=True)
    p_retrain.add_argument("--previous-model-id", type=_parse_uuid, required=True)
    p_retrain.add_argument("--reason", required=True)
    p_retrain.add_argument("--storage-dir", type=Path, default=Path("artifacts/textile/store"))
    p_retrain.add_argument("--synthetic-confirm", action="store_true")

    # predict
    p_pred = sub.add_parser("predict", help="Predict waste using the active factory model.")
    p_pred.add_argument("--process", choices=["weaving", "knitting"], required=True)
    p_pred.add_argument("--factory-id", type=_parse_uuid, required=True)
    p_pred.add_argument("--csv", type=Path, required=True)
    p_pred.add_argument("--storage-dir", type=Path, default=Path("artifacts/textile/store"))
    p_pred.add_argument("--output-json", type=Path, default=None)
    p_pred.add_argument("--synthetic-confirm", action="store_true")

    # model-info
    p_info = sub.add_parser("model-info", help="Show active and registered model versions.")
    p_info.add_argument("--process", choices=["weaving", "knitting"], required=True)
    p_info.add_argument("--factory-id", type=_parse_uuid, required=True)
    p_info.add_argument("--storage-dir", type=Path, default=Path("artifacts/textile/store"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        task_id, dataset_id, contract, _ = resolve_process_spec(args.process)
        payload = {
            "domain": "textile",
            "process": args.process,
            "task_id": task_id,
            "dataset_id": dataset_id,
            "contract_version": contract.version,
            "target": contract.target.feature.name,
            "target_confirmed": contract.target.is_confirmed,
            "predictors": list(contract.predictor_names),
            "business_keys": list(contract.business_keys),
            "grouping_columns": list(contract.grouping_columns),
            "time_column": contract.time_column,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "template":
        written = export_textile_templates(args.output_dir)
        print(json.dumps({k: str(v) for k, v in written.items()}, indent=2))
        return 0

    service = TextilePipelineService(args.storage_dir)

    if args.command == "validate":
        purpose = (
            DatasetPurpose.TRAINING if args.purpose == "training" else DatasetPurpose.PREDICTION
        )
        _, _, contract, _ = resolve_process_spec(
            args.process, use_synthetic_confirmation=args.synthetic_confirm
        )
        prep = service.ingest_and_prepare(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            purpose=purpose,
            contract=contract,
            use_synthetic_confirmation=args.synthetic_confirm,
        )
        print(
            json.dumps(
                {
                    "status": "VALIDATED",
                    "engineered_revision_id": str(prep.engineered_artifact.metadata.revision_id),
                    "rows": prep.engineered_artifact.metadata.row_count,
                },
                indent=2,
            )
        )
        return 0

    if args.command in {"train", "retrain"}:
        _, _, contract, _ = resolve_process_spec(
            args.process, use_synthetic_confirmation=args.synthetic_confirm
        )
        prep = service.ingest_and_prepare(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            purpose=DatasetPurpose.TRAINING,
            contract=contract,
            use_synthetic_confirmation=args.synthetic_confirm,
        )
        mode = TrainingMode.RETRAINING if args.command == "retrain" else TrainingMode.INITIAL
        prev_id = args.previous_model_id if args.command == "retrain" else None
        reason = args.reason if args.command == "retrain" else "CLI initial factory training"
        res = service.train(
            factory_id=args.factory_id,
            process_id=args.process,
            prepared=prep,
            contract=contract,
            mode=mode,
            previous_model_version_id=prev_id,
            reason=reason,
        )
        print(
            json.dumps(
                {
                    "decision": res.evaluation.decision.value,
                    "activated": res.activated,
                    "model_version_id": (
                        str(res.model_version.model_version_id) if res.model_version else None
                    ),
                    "baseline_mae": res.evaluation.baseline_mae,
                    "selected_mae": res.evaluation.selected_mae,
                    "improvement_pct": res.evaluation.test_improvement_pct,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "predict":
        _, _, contract, _ = resolve_process_spec(
            args.process, use_synthetic_confirmation=args.synthetic_confirm
        )
        batch = service.predict(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            contract=contract,
        )
        payload = batch.model_dump(mode="json")
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "model_version_id": str(batch.model.model_version_id),
                    "total_rows": len(batch.predictions),
                    "succeeded": sum(1 for p in batch.predictions if p.status.value == "succeeded"),
                    "rejected": sum(1 for p in batch.predictions if p.status.value == "rejected"),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "model-info":
        task_id, _, _, _ = resolve_process_spec(args.process)
        scope = ModelScope(
            factory_id=args.factory_id,
            domain="textile",
            process=args.process,
            task_id=task_id,
        )
        active = service.stores.model_registry.get_active(scope=scope)
        versions = service.stores.model_registry.list_versions(scope=scope)
        print(
            json.dumps(
                {
                    "active_model_version_id": (
                        str(active.version.model_version_id) if active else None
                    ),
                    "registered_versions": [
                        {
                            "model_version_id": str(v.version.model_version_id),
                            "version_number": v.version.version_number,
                            "status": v.status.value,
                            "family": v.version.candidate.family.value,
                        }
                        for v in versions
                    ],
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
