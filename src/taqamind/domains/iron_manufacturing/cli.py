"""Lightweight command-line interface for the iron_manufacturing domain.

Usage:
    python -m taqamind.domains.iron_manufacturing.cli --help
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetPurpose,
    TargetConfirmation,
)
from taqamind.core.interfaces import (
    AcceptancePolicy,
    ModelScope,
    ModelTransitionAction,
    ModelTransitionRequest,
    TrainingMode,
)
from taqamind.domains.iron_manufacturing.config import DOMAIN_ID
from taqamind.domains.iron_manufacturing.domain import (
    PROCESS_DEFAULT_TASKS,
    PROCESS_ENERGY_TASKS,
    make_synthetic_confirmation,
)
from taqamind.domains.iron_manufacturing.energy_contracts import (
    get_iron_energy_dataset_contract,
)
from taqamind.domains.iron_manufacturing.energy_pipeline import IronEnergyPipelineService
from taqamind.domains.iron_manufacturing.pipeline import IronManufacturingPipeline
from taqamind.domains.iron_manufacturing.services.templates import (
    generate_energy_process_templates,
)


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
    parser.add_argument(
        "--module",
        default="waste",
        choices=["waste", "energy"],
        help="Target module: V1 waste prediction ('waste') or V2 electrical energy ('energy').",
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


def _run_energy_cli(args: argparse.Namespace) -> int:
    """Execute V2 electrical energy CLI commands via IronEnergyPipelineService."""
    import pandas as pd

    if args.command == "template":
        files = generate_energy_process_templates(
            process_id=args.process,
            output_dir=args.output_dir,
            sample_rows=args.sample_rows,
        )
        print(json.dumps({k: str(v) for k, v in files.items()}, indent=2, ensure_ascii=False))
        return 0

    service = IronEnergyPipelineService(Path(args.storage_root))
    confirmation = _resolve_confirmation(args)
    contract = get_iron_energy_dataset_contract(args.process, confirmation=confirmation)
    task_id = PROCESS_ENERGY_TASKS[args.process]

    if args.command == "inspect":
        payload = {
            "status": "succeeded",
            "domain": DOMAIN_ID,
            "module": "energy_efficiency",
            "process": args.process,
            "task_id": task_id,
            "dataset_id": contract.dataset_id,
            "contract_version": contract.version,
            "target": contract.target.feature.name,
            "target_unit": contract.target.feature.unit,
            "target_confirmed": contract.target.is_confirmed,
            "predictors": list(contract.predictor_names),
            "business_keys": list(contract.business_keys),
            "grouping_columns": list(contract.grouping_columns),
            "time_column": contract.time_column,
            "csv": str(args.csv),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "validate":
        prepared = service.ingest_and_prepare(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            purpose=DatasetPurpose(args.purpose),
            contract=contract,
        )
        print(
            json.dumps(
                {
                    "status": "ready",
                    "revision_id": str(prepared.artifact.metadata.revision_id),
                    "row_count": prepared.artifact.metadata.row_count,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command in {"train", "retrain"}:
        prepared = service.ingest_and_prepare(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            purpose=DatasetPurpose.TRAINING,
            contract=contract,
        )
        policy = AcceptancePolicy(
            minimum_improvement_pct=args.min_improvement_pct,
            justification=f"CLI V2 energy policy (min improvement {args.min_improvement_pct}%).",
            created_at=prepared.context.created_at - timedelta(seconds=1),
        )
        mode = TrainingMode.RETRAINING if args.command == "retrain" else TrainingMode.INITIAL
        prev_id = None
        if mode == TrainingMode.RETRAINING:
            scope = ModelScope(
                factory_id=args.factory_id,
                domain=DOMAIN_ID,
                process=args.process,
                task_id=task_id,
            )
            active = service.core.model_registry.get_active(scope=scope)
            prev_id = active.version.model_version_id if active is not None else None
        outcome = service.train_and_evaluate(
            factory_id=args.factory_id,
            process_id=args.process,
            prepared=prepared,
            contract=contract,
            policy=policy,
            mode=mode,
            previous_model_version_id=prev_id,
            auto_activate=not getattr(args, "no_auto_activate", False),
        )
        print(
            json.dumps(
                {
                    "decision": outcome.evaluation.decision.value,
                    "activated": outcome.activated,
                    "model_version_id": str(outcome.model_entry.version.model_version_id),
                    "version_number": outcome.model_entry.version.version_number,
                    "status": outcome.model_entry.status.value,
                    "baseline_mae": outcome.evaluation.baseline_mae,
                    "selected_mae": outcome.evaluation.selected_mae,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "predict":
        batch = service.predict(
            factory_id=args.factory_id,
            process_id=args.process,
            csv_path=args.csv,
            contract=contract,
        )
        if args.output_csv is not None:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "status": p.status.value,
                    "predicted_value": p.predicted_value,
                    "derived_rate": p.derived_rate.value if p.derived_rate is not None else None,
                    "rejection_reason": p.rejection_reason,
                }
                for p in batch.predictions
            ]
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(
            json.dumps(
                {
                    "status": batch.status.value,
                    "model_version_id": str(batch.model.model_version_id),
                    "successful_count": batch.successful_count,
                    "rejected_count": batch.rejected_count,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if batch.successful_count > 0 else 1

    scope = ModelScope(
        factory_id=args.factory_id,
        domain=DOMAIN_ID,
        process=args.process,
        task_id=task_id,
    )
    if args.command == "model-info":
        active = service.core.model_registry.get_active(scope=scope)
        versions = service.core.model_registry.list_versions(scope=scope)
        print(
            json.dumps(
                {
                    "domain": DOMAIN_ID,
                    "process": args.process,
                    "task_id": task_id,
                    "active_model_version_id": (
                        str(active.version.model_version_id) if active is not None else None
                    ),
                    "versions": [
                        {
                            "model_version_id": str(v.version.model_version_id),
                            "version_number": v.version.version_number,
                            "status": v.status.value,
                            "family": v.version.candidate.family,
                        }
                        for v in versions
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "rollback":
        active = service.core.model_registry.get_active(scope=scope)
        entry = service.core.model_registry.transition(
            request=ModelTransitionRequest(
                scope=scope,
                action=ModelTransitionAction.ROLLBACK,
                target_model_version_id=args.target_version_id,
                expected_active_model_version_id=(
                    active.version.model_version_id if active is not None else None
                ),
                requested_by=args.requested_by,
                reason=args.reason,
            )
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

    return 2


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.module == "energy":
        return _run_energy_cli(args)
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
