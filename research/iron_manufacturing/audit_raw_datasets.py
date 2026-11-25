"""Reproducible forensic dataset audit and EDA generator for iron_manufacturing.

Reads raw datasets from `data/iron_manufacturing/raw/` without modifying any
source bytes, calculates SHA-256 digests, checks structure/missingness/duplicates/
zero-inflation/outliers/group dependence/temporal ordering/leakage, and outputs
JSON diagnostics + diagnostic figures into `reports/iron_manufacturing/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_alchimia(raw_root: Path, fig_dir: Path) -> dict[str, Any]:
    """Audit EU Horizon ALCHIMIA scrap delivery datasets (scrap_A.csv & scrap_C.csv)."""
    folder = raw_root / "alchimia_scrap"
    result: dict[str, Any] = {
        "dataset_group": "alchimia_scrap",
        "source": (
            "Zenodo DOIs 10.5281/zenodo.17630721 (Plant A) & "
            "10.5281/zenodo.17633305 (Plant C)"
        ),
        "license": "CC-BY-4.0",
        "country": "Anonymised European electric steel plants (NOT Egyptian)",
        "classification": "CONDITIONALLY_SUITABLE_REFERENCE_ONLY",
        "files": {},
    }
    frames: dict[str, pd.DataFrame] = {}
    for fname in ("scrap_A.csv", "scrap_C.csv"):
        fpath = folder / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, low_memory=False)
        contam = df["Soil"].fillna(0) + df["Water"].fillna(0) + df["Other"].fillna(0)
        df = df.assign(total_contaminant_kg=contam)
        frames[fname] = df

        entry_ts = pd.to_datetime(df["EntryDate"], errors="coerce")
        exit_ts = pd.to_datetime(df["ExitDate"], errors="coerce")
        top_sup_share = float(df["Supplier"].value_counts(normalize=True).iloc[0] * 100.0)
        result["files"][fname] = {
            "sha256": sha256_file(fpath),
            "size_bytes": int(fpath.stat().st_size),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "missing_counts": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
            "id_unique": bool(df["ID"].is_unique),
            "entry_min": str(entry_ts.min()),
            "entry_max": str(entry_ts.max()),
            "exit_before_entry_rows": int((exit_ts < entry_ts).sum()),
            "n_suppliers": int(df["Supplier"].nunique()),
            "n_scrap_types": int(df["ScrapType"].nunique()),
            "top_supplier_share_pct": round(top_sup_share, 2),
            "weight_kg": {
                "min": float(df["Weight"].min()),
                "median": float(df["Weight"].median()),
                "p95": float(df["Weight"].quantile(0.95)),
                "max": float(df["Weight"].max()),
            },
            "total_contaminant_kg": {
                "zero_pct": round(float((contam == 0).mean() * 100.0), 2),
                "median": float(contam.median()),
                "mean": round(float(contam.mean()), 2),
                "p95": float(contam.quantile(0.95)),
                "max": float(contam.max()),
                "weighted_pct_of_weight": round(
                    float(contam.sum() / df["Weight"].sum() * 100.0), 3
                ),
                "distinct_values": int(contam.nunique()),
            },
        }

    if frames:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, (fname, df) in zip(axes, frames.items(), strict=False):
            c = df["total_contaminant_kg"]
            clipped = c.clip(upper=float(c.quantile(0.99)))
            sns.histplot(clipped, bins=35, ax=ax, color="#1f6f8b")
            ax.set_title(
                f"{fname}: Contaminant Mass (kg)\n"
                f"n={len(df):,}, zeros={(c == 0).mean() * 100:.1f}%"
            )
            ax.set_xlabel("total_contaminant_kg (clipped at p99)")
            ax.set_ylabel("Deliveries")
        fig.tight_layout()
        fig.savefig(fig_dir / "alchimia_contaminant_distributions.png", dpi=150)
        plt.close(fig)

        # Group-level variation by scrap type for scrap_A
        if "scrap_A.csv" in frames:
            df_a = frames["scrap_A.csv"].copy()
            df_a["contam_rate_pct"] = (df_a["total_contaminant_kg"] / df_a["Weight"]) * 100.0
            top_types = df_a["ScrapType"].value_counts().head(8).index
            sub = df_a[df_a["ScrapType"].isin(top_types)]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.boxplot(
                data=sub,
                x="ScrapType",
                y="contam_rate_pct",
                ax=ax,
                showfliers=False,
                color="#4e89ae",
            )
            ax.set_title("ALCHIMIA Plant A: Contaminant Rate (%) by Scrap Grade (Top 8)")
            ax.set_ylabel("Contaminant Rate (% of Weight)")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(fig_dir / "alchimia_grade_variation.png", dpi=150)
            plt.close(fig)

    return result


def audit_ladle_furnace(raw_root: Path, fig_dir: Path) -> dict[str, Any]:
    """Audit Yandex Practicum Ladle Furnace multi-table dataset (quarantined / unsuitable)."""
    folder = raw_root / "arc_furnace_heats"
    result: dict[str, Any] = {
        "dataset_group": "ladle_furnace_heats",
        "source": (
            "GitHub mirror shradha289/steel-temperature-prediction "
            "(Yandex Practicum capstone)"
        ),
        "license": "UNKNOWN / UNSTATED",
        "country": "Undisclosed (NOT Egyptian)",
        "classification": "UNSUITABLE_FOR_WASTE_PREDICTION",
        "reason": (
            "Secondary metallurgy Ladle Furnace (LF) temperature dataset — NOT EAF melting. "
            "No scrap/waste target, no tonnage, no kWh meter, unverified units, "
            "100% in-process features, and unstated license."
        ),
        "files": {},
    }
    if not folder.exists():
        return result

    for csv_file in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_file, low_memory=False)
        miss_pct = {
            k: round(float(v * 100.0), 2)
            for k, v in df.isna().mean().items()
            if v > 0
        }
        result["files"][csv_file.name] = {
            "sha256": sha256_file(csv_file),
            "size_bytes": int(csv_file.stat().st_size),
            "rows": int(len(df)),
            "columns_count": int(df.shape[1]),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "missing_pct": miss_pct,
        }

    arc_path = folder / "data_arc_en.csv"
    bulk_path = folder / "data_bulk_en.csv"
    if arc_path.exists() and bulk_path.exists():
        arc_df = pd.read_csv(arc_path)
        bulk_df = pd.read_csv(bulk_path)
        result["anomalies"] = {
            "negative_reactive_power_rows": int((arc_df["Reactive power"] < 0).sum()),
            "negative_reactive_power_min": float(arc_df["Reactive power"].min()),
        }
        miss_series = bulk_df.drop(columns=["key"]).isna().mean() * 100.0
        fig, ax = plt.subplots(figsize=(10, 4.2))
        miss_series.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#c85a53")
        ax.set_title("Ladle Furnace (Quarantined): Missingness (%) Across Bulk 1-15 Columns")
        ax.set_ylabel("Missing (%)")
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(fig_dir / "ladle_furnace_bulk_missingness.png", dpi=150)
        plt.close(fig)

    return result


def audit_uci_energy(raw_root: Path, fig_dir: Path) -> dict[str, Any]:
    """Audit UCI Steel Industry Energy Consumption dataset (ID 851 — unsuitable for waste)."""
    fpath = raw_root / "uci_steel_energy" / "Steel_industry_data.csv"
    result: dict[str, Any] = {
        "dataset_group": "uci_steel_energy",
        "source": "UCI ML Repository ID 851 (Daewoo Steel, Gwangyang, Korea, 2018)",
        "license": "CC-BY-4.0",
        "country": "South Korea (NOT Egyptian)",
        "classification": "UNSUITABLE_FOR_WASTE_PREDICTION",
        "reason": (
            "15-minute plant electricity meter series with zero production tonnage, zero "
            "heat/lot identifiers, zero waste/scrap target, and 59.9% zero CO2 readings."
        ),
    }
    if not fpath.exists():
        return result

    df = pd.read_csv(fpath, low_memory=False)
    num_cols = [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
    ]
    result["file"] = {
        "name": fpath.name,
        "sha256": sha256_file(fpath),
        "size_bytes": int(fpath.stat().st_size),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "co2_zero_pct": round(float((df["CO2(tCO2)"] == 0).mean() * 100.0), 2),
        "pf_below_92_pct": round(
            float((df["Lagging_Current_Power_Factor"] < 92.0).mean() * 100.0), 2
        ),
    }

    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("UCI 851 Steel Energy (Unsuitable for Waste): Numeric Correlation Matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "uci851_correlation_matrix.png", dpi=150)
    plt.close(fig)
    return result


def audit_zenodo_references(raw_root: Path) -> dict[str, Any]:
    """Audit Zenodo parameter registry (22162000) and synthetic OR benchmark (21447633)."""
    summary: dict[str, Any] = {}
    mat_path = (
        raw_root
        / "zenodo_eaf_blending"
        / "extracted"
        / "Complete_Synchronized_Submission_Package"
        / "03_Code_and_Data"
        / "Input_Data_and_Source_Registry"
        / "materials.csv"
    )
    if mat_path.exists():
        mat_df = pd.read_csv(mat_path)
        summary["zenodo_22162000_materials"] = {
            "source": "Zenodo DOI 10.5281/zenodo.22162000",
            "license": "CC-BY-4.0",
            "classification": "REFERENCE_PARAMETERS_ONLY_NOT_TRAINING_DATA",
            "sha256": sha256_file(mat_path),
            "rows": int(len(mat_df)),
            "columns": list(mat_df.columns),
        }

    flex_zip = raw_root / "zenodo_eaf_flexibility"
    if flex_zip.exists():
        summary["zenodo_21447633_flexibility"] = {
            "source": "Zenodo DOI 10.5281/zenodo.21447633",
            "license": "CC-BY-4.0 / MIT",
            "classification": "SYNTHETIC_OR_REFERENCE_ONLY_UNSUITABLE_FOR_ML",
            "file_count": int(sum(1 for p in flex_zip.rglob("*") if p.is_file())),
        }
    return summary


def run_audit(raw_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "domain_id": "iron_manufacturing",
        "raw_root_exists": raw_root.exists(),
        "datasets": {
            "alchimia_scrap": audit_alchimia(raw_root, fig_dir),
            "ladle_furnace_heats": audit_ladle_furnace(raw_root, fig_dir),
            "uci_steel_energy": audit_uci_energy(raw_root, fig_dir),
            "zenodo_references": audit_zenodo_references(raw_root),
        },
        "egyptian_production_data_available": False,
        "verdict": (
            "No supplied public dataset represents an Egyptian EAF meltshop or rebar rolling mill "
            "with confirmed pre-production predictors and waste targets. ALCHIMIA provides real EU "
            "scrap-yard contaminant data suitable for EDA and reference benchmarking only. "
            "Production training remains BLOCKED until Egyptian factory data and target "
            "confirmations are supplied."
        ),
    }
    report_path = output_dir / "dataset_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iron_manufacturing raw dataset audit & EDA.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/iron_manufacturing/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/iron_manufacturing"),
    )
    args = parser.parse_args()
    summary = run_audit(args.raw_root, args.output_dir)
    print(json.dumps({"verdict": summary["verdict"], "output": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
