"""Lightweight command-line interface for the iron_manufacturing domain.

Usage:
    python -m taqamind.domains.iron_manufacturing.cli --help
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetPurpose,
    TargetConfirmation,
)
from taqamind.core.interfaces.evaluation import AcceptancePolicy
from taqamind.domains.iron_manufacturing.domain import (
    PROCESS_DEFAULT_TASKS,
    make_synthetic_confirmation,
)
from taqamind.domains.iron_manufacturing.pipeline import IronManufacturingPipeline


def _parse_uuid(value: str) -> UUID:
    return UUID(value)


def _resolve_confirmation(args: argparse.Namespace) -> TargetConfirmation | None:
    if getattr(args, "synthetic_confirmation", False):
        return make_synthetic_confirmation(
            confirmed_by=getattr(args, "confirmed_by", None) or "cli_synthetic_demo",
            notes="Explicit synthetic CLI fixture confirmation — NOT production-confirmed.",
        )
    confirmed_by = getattr(args, "confirmed_by", None)
    if confirmed_by:
        confirmed_at_raw = getattr(args, "confirmed_at", None)
        confirmed_at = (
            datetime.fromisoformat(confirmed_at_raw)
            if confirmed_at_raw
            else datetime.now(UTC)
        )
        return TargetConfirmation(
            status=ConfirmationStatus.CONFIRMED,
            confirmed_by=confirmed_by,
            confirmed_at=confirmed_at,
            notes=getattr(args, "confirmation_notes", None) or "Confirmed via CLI argument.",
        )
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m taqamind.domains.iron_manufacturing.cli",
        description=(
            "TaqaMind Iron Manufacturing industrial ML pipeline CLI "
            "(pre-production waste/scrap quantity prediction)."
        ),
    )
    parser.add_argument(
        "--storage-root",
        default="artifacts/iron_manufacturing/store",
        help="Local storage root for artifacts and registries.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect CSV file, profile, and mapping proposal.")
    p_inspect.add_argument("--factory-id", required=True, type=_parse_uuid)
    p_inspect.add_argument("--process", required=True, choices=sorted(PROCESS_DEFAULT_TASKS))
    p_inspect.add_argument("--csv", required=True, type=Path)

    # template
    p_tpl = sub.add_parser("template", help="Generate historical and prediction CSV templates.")
    p_tpl.add_argument("--process", required=True, choices=sorted(PROCESS_DEFAULT_TASKS))
    p_tpl.add_argument(
        "--output-dir",
        default="contracts/domains/iron_manufacturing/templates",
        type=Path,
    )
    p_tpl.add_argument("--sample-rows", type=int, default=3)

    # validate
    p_val = sub.add_parser(
        "validate", help="Validate CSV against DatasetContract without training."
    )
    p_val.add_argument("--factory-id", required=True, type=_parse_uuid)
    p_val.add_argument("--process", required=True, choices=sorted(PROCESS_DEFAULT_TASKS))
    p_val.add_argument("--csv", required=True, type=Path)
    p_val.add_argument(
        "--purpose",
        default="training",
        choices=["training", "prediction", "evaluation"],
    )
    p_val.add_argument("--confirm-mappings", action="store_true")
    p_val.add_argument("--synthetic-confirmation", action="store_true")
    p_val.add_argument("--confirmed-by", default=None)

    # train
    p_train = sub.add_parser("train", help="Run initial controlled training for a factory.")
    p_train.add_argument("--factory-id", required=True, type=_parse_uuid)
    p_train.add_argument("--process", required=True, choices=sorted(PROCESS_DEFAULT_TASKS))
    p_train.add_argument("--csv", required=True, type=Path)
    p_train.add_argument("--requested-by", default="cli_user")
    p_train.add_argument("--reason", default="Initial CLI training execution")
    p_train.add_argument("--confirm-mappings", action="store_true")
    p_train.add_argument("--synthetic-confirmation", action="store_true")
    p_train.add_argument("--confirmed-by", default=None)
    p_train.add_argument("--min-improvement-pct", type=float, default=5.0)
    p_train.add_argument("--no-auto-activate", action="store_true")

    # predict
    p_pred = sub.add_parser("predict", help="Predict using persisted ACTIVE model (zero fit).")
    p_pred.add_argument("--factory-id", required=True, type=_parse_uuid)
    p_pred.add_argument("--process", required=True, choices=sorted(PROCESS_DEFAULT_TASKS))
    p_pred.add_argument("--csv", required=True, type=Path)
    p_pred.add_argument("--output-csv", default=None, type=Path)
    p_pred.add_argument("--confirm-mappings", action="store_true")
    p_pred.add_argument("--synthetic-confirmation", action="store_true")
    p_pred.add_argument("--confirmed-by", default=None)

    # retrain
    p_retrain = sub.add_parser("retrain", help="Explicitly retrain a new challenger version.")
    p_retrain.add_argument("--factory-id", required=True, type=_parse_uuid)
    p_retrain.add_argument("--process", required=True, choices=sorted(PROCESS_DEFAULT_TASKS))
    p_retrain.add_argument("--csv", required=True, type=Path)
    p_retrain.add_argument("--requested-by", default="cli_user")
    p_retrain.add_argument("--reason", default="Explicit CLI retraining request")
    p_retrain.add_argument("--confirm-mappings", action="store_true")
    p_retrain.add_argument("--synthetic-confirmation", action="store_true")
    p_retrain.add_argument("--confirmed-by", default=None)
    p_retrain.add_argument("--allow-corrections", action="store_true")
    p_retrain.add_argument("--correction-reason", default=None)
    p_retrain.add_argument("--min-improvement-pct", type=float, default=5.0)

    # model-info
    p_info = sub.add_parser("model-info", help="Show active model and registered version history.")
    p_info.add_argument("--factory-id", required=True, type=_parse_uuid)
    p_info.add_argument("--process", required=True, choices=sorted(PROCESS_DEFAULT_TASKS))

    # rollback
    p_rb = sub.add_parser("rollback", help="Rollback active model to a verified archived version.")
    p_rb.add_argument("--factory-id", required=True, type=_parse_uuid)
    p_rb.add_argument("--process", required=True, choices=sorted(PROCESS_DEFAULT_TASKS))
    p_rb.add_argument("--target-version-id", required=True, type=_parse_uuid)
    p_rb.add_argument("--requested-by", default="cli_admin")
    p_rb.add_argument("--reason", default="Explicit CLI rollback")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    pipeline = IronManufacturingPipeline(storage_root=args.storage_root)

    if args.command == "inspect":
        result = pipeline.inspect_dataset(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "succeeded" else 1

    if args.command == "template":
        files = pipeline.generate_templates(
            process_id=args.process,
            output_dir=args.output_dir,
            sample_rows=args.sample_rows,
        )
        print(json.dumps({k: str(v) for k, v in files.items()}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "validate":
        res = pipeline.validate_dataset(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            purpose=DatasetPurpose(args.purpose),
            confirmation=_resolve_confirmation(args),
            confirm_all_proposed_mappings=True,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["status"] in {"ready", "limited"} else 1

    if args.command == "train":
        policy = AcceptancePolicy(
            minimum_improvement_pct=args.min_improvement_pct,
            justification=f"CLI acceptance policy (min improvement {args.min_improvement_pct}%).",
        )
        outcome = pipeline.train_initial(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            confirmation=_resolve_confirmation(args),
            requested_by=args.requested_by,
            reason=args.reason,
            confirm_all_proposed_mappings=True,
            acceptance_policy=policy,
            auto_activate_first=not args.no_auto_activate,
        )
        print(json.dumps(outcome.to_summary_dict(), indent=2, ensure_ascii=False))
        return 0 if outcome.status in {"succeeded", "reused"} else 1

    if args.command == "predict":
        outcome = pipeline.predict(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            confirmation=_resolve_confirmation(args),
            confirm_all_proposed_mappings=True,
            output_csv_path=args.output_csv,
        )
        print(json.dumps(outcome.to_summary_dict(), indent=2, ensure_ascii=False))
        return 0 if outcome.status in {"succeeded", "partial"} else 1

    if args.command == "retrain":
        policy = AcceptancePolicy(
            minimum_improvement_pct=args.min_improvement_pct,
            justification=f"CLI retraining policy (min improvement {args.min_improvement_pct}%).",
        )
        outcome = pipeline.retrain(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            confirmation=_resolve_confirmation(args),
            requested_by=args.requested_by,
            reason=args.reason,
            confirm_all_proposed_mappings=True,
            acceptance_policy=policy,
            allow_confirmed_corrections=args.allow_corrections,
            correction_reason=args.correction_reason,
        )
        print(json.dumps(outcome.to_summary_dict(), indent=2, ensure_ascii=False))
        return 0 if outcome.status in {"succeeded", "reused"} else 1

    if args.command == "model-info":
        info = pipeline.get_model_info(
            factory_id=args.factory_id,
            process_id=args.process,
        )
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    if args.command == "rollback":
        entry = pipeline.rollback_model(
            factory_id=args.factory_id,
            process_id=args.process,
            target_archived_version_id=args.target_version_id,
            requested_by=args.requested_by,
            reason=args.reason,
        )
        print(
            json.dumps(
                {
                    "status": entry.status.value,
                    "active_model_version_id": str(entry.version.model_version_id),
                    "version_number": entry.version.version_number,
                },
                indent=2,
            )
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
