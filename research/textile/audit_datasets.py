"""Reproducible audit and EDA script for raw textile datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import openpyxl
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a file without modifying bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_weaving(raw_path: Path, plot_dir: Path) -> dict[str, Any]:
    """Audit full_weaving_dataset.csv and generate distribution plots."""
    df = pd.read_csv(raw_path, keep_default_na=True, low_memory=False)
    target = pd.to_numeric(df["Rej_and_cut_Piece"], errors="coerce")
    beam = pd.to_numeric(df["Rec_Beam_length(yds)"], errors="coerce")
    pos_target = target[target > 0]
    exact_dups = int(df.duplicated().sum())
    unique_rows = int(len(df) - exact_dups)
    id_counts = df["ID"].value_counts()

    exceed_mask = (target > beam) & target.notna() & beam.notna()
    act_shrink_na = int((df["act_shrink%"].astype(str).str.strip().str.casefold() == "na").sum())

    # Check if repeated ID rows have varying or identical target values.
    dup_groups = df[df.duplicated(keep=False)].groupby(list(df.columns), dropna=False).size()
    max_group_repeat = int(dup_groups.max()) if not dup_groups.empty else 1

    # Check whether within-ID rows have distinct targets (e.g. multiple beam cuts vs replicated).
    per_id_target_nunique = df.groupby("ID")["Rej_and_cut_Piece"].nunique()
    ids_with_varying_target = int((per_id_target_nunique > 1).sum())

    # Plot target distribution (positives log scale + zero share) and top predictors.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    zero_count = int((target == 0).sum())
    pos_count = int((target > 0).sum())
    axes[0].bar(["Zero Scrap (0 yds)", "Positive Scrap (>0 yds)"], [zero_count, pos_count])
    axes[0].set_title("Weaving Target Zero-Inflation")
    axes[0].set_ylabel("Row Count")

    axes[1].hist(np.log1p(pos_target.dropna()), bins=30, edgecolor="black")
    axes[1].set_title("Weaving Positive Reject Length log1p(yds)")
    axes[1].set_xlabel("log1p(Rej_and_cut_Piece)")
    axes[1].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(plot_dir / "weaving_target_distribution.png", dpi=140)
    plt.close(fig)

    # Plot correlation among deduplicated numeric pre-production columns and target.
    dedup = df.drop_duplicates().copy()
    num_cols = [
        "Req_Finish_Fabrics",
        "Fabric_Allowance",
        "Rec_Beam_length(yds)",
        "Shrink_allow",
        "Req_grey_fabric",
        "Req_beam_length(yds)",
        "warp_count",
        "weft_count",
        "epi",
        "ppi",
        "Rej_and_cut_Piece",
    ]
    corr = dedup[num_cols].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)), num_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(num_cols)), num_cols, fontsize=8)
    ax.set_title("Weaving Deduplicated Pre-Production + Target Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(plot_dir / "weaving_correlation_matrix.png", dpi=140)
    plt.close(fig)

    return {
        "file": raw_path.name,
        "size_bytes": raw_path.stat().st_size,
        "sha256": file_sha256(raw_path),
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": list(df.columns),
        "exact_duplicate_rows": exact_dups,
        "unique_rows": unique_rows,
        "max_exact_group_repeat": max_group_repeat,
        "unique_order_ids": int(df["ID"].nunique()),
        "ids_with_varying_target": ids_with_varying_target,
        "top_id_repeat_count": int(id_counts.max()),
        "zero_rejection_rows": zero_count,
        "positive_rejection_rows": pos_count,
        "null_rejection_rows": int(target.isna().sum()),
        "rejection_gt_beam_rows": int(exceed_mask.sum()),
        "max_rejection_to_beam_ratio": float((target[beam > 0] / beam[beam > 0]).max()),
        "act_shrink_literal_na_rows": act_shrink_na,
        "previous_pdn_nulls": int(df["Previous_pdn"].isna().sum()),
        "total_pdn_mc_nulls": int(df["Total_pdn_m/c"].isna().sum()),
        "positive_target_skewness": float(pos_target.skew()),
        "positive_target_median_yds": float(pos_target.median()),
        "positive_target_p95_yds": float(pos_target.quantile(0.95)),
        "positive_target_max_yds": float(pos_target.max()),
        "unique_months": sorted(df["Month"].dropna().astype(str).unique().tolist()),
    }


def audit_knitting(raw_path: Path, plot_dir: Path) -> dict[str, Any]:
    """Audit knitting_dataset.csv (Latin-1) and generate distribution plots."""
    df = pd.read_csv(raw_path, encoding="latin1", keep_default_na=True, low_memory=False)
    # Strip NBSP from column names for inspection while keeping exact column names list.
    raw_columns = list(df.columns)
    clean_cols = [c.replace("\xa0", " ").strip() for c in raw_columns]
    col_map = dict(zip(clean_cols, raw_columns, strict=True))

    rej_col = col_map["Reject Qty"]
    shift_col = col_map["Shift Total"]
    target = pd.to_numeric(df[rej_col], errors="coerce")
    shift_total = pd.to_numeric(df[shift_col], errors="coerce")

    null_count = int(target.isna().sum())
    zero_count = int((target == 0).sum())
    pos_target = target[target > 0]
    pos_count = int(len(pos_target))
    exact_dups = int(df.duplicated().sum())

    valid_ratio_mask = target.notna() & shift_total.notna() & (shift_total > 0)
    ratios = target[valid_ratio_mask] / shift_total[valid_ratio_mask]
    ratio_gt_1 = int((ratios > 1.0).sum())
    ratio_gt_0_5 = int((ratios > 0.5).sum())

    # Check structural sparsity of tex_2 and tex_3.
    tex2_null = float(df[col_map["tex_2"]].isna().mean() * 100) if "tex_2" in col_map else None
    tex3_null = float(df[col_map["tex_3"]].isna().mean() * 100) if "tex_3" in col_map else None

    # Check run_id cardinality if present.
    run_id_nunique = int(df[col_map["run_id"]].nunique()) if "run_id" in col_map else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(
        ["NULL", "Zero (0 kg)", "Positive (>0 kg)"],
        [null_count, zero_count, pos_count],
    )
    axes[0].set_title("Knitting Reject Qty Completeness & Zero-Inflation")
    axes[0].set_ylabel("Row Count")

    axes[1].hist(pos_target.dropna(), bins=25, edgecolor="black")
    axes[1].set_title("Knitting Positive Reject Qty (kg)")
    axes[1].set_xlabel("Reject Qty (kg)")
    axes[1].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(plot_dir / "knitting_target_distribution.png", dpi=140)
    plt.close(fig)

    # Missingness bar plot across columns with missing values.
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_nonzero = missing_pct[missing_pct > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(missing_nonzero)), missing_nonzero.values)
    ax.set_yticks(
        range(len(missing_nonzero)),
        [c.replace("\xa0", " ").strip() for c in missing_nonzero.index],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Missing Percentage (%)")
    ax.set_title("Knitting Column Missingness")
    fig.tight_layout()
    fig.savefig(plot_dir / "knitting_missingness.png", dpi=140)
    plt.close(fig)

    return {
        "file": raw_path.name,
        "size_bytes": raw_path.stat().st_size,
        "sha256": file_sha256(raw_path),
        "rows": int(len(df)),
        "columns_count": int(len(raw_columns)),
        "columns": clean_cols,
        "columns_with_nbsp": [repr(c) for c in raw_columns if "\xa0" in c],
        "exact_duplicate_rows": exact_dups,
        "null_rejection_rows": null_count,
        "zero_rejection_rows": zero_count,
        "positive_rejection_rows": pos_count,
        "max_rejection_to_shift_ratio": float(ratios.max()),
        "ratio_gt_100pct_count": ratio_gt_1,
        "ratio_gt_50pct_count": ratio_gt_0_5,
        "shift_total_le_5kg_count": int((shift_total <= 5.0).sum()),
        "tex_2_null_pct": tex2_null,
        "tex_3_null_pct": tex3_null,
        "run_id_nunique": run_id_nunique,
        "positive_target_skewness": float(pos_target.skew()),
        "positive_target_median_kg": float(pos_target.median()),
        "positive_target_p95_kg": float(pos_target.quantile(0.95)),
        "positive_target_max_kg": float(pos_target.max()),
    }


def audit_excel(raw_path: Path) -> dict[str, Any]:
    """Audit raw_data_production_reports_20222.xlsx for date-corrupted yarn counts."""
    wb = openpyxl.load_workbook(raw_path, read_only=True, data_only=True)
    sheet_name = wb.sheetnames[0]
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    headers = list(next(rows_iter))
    yarn_idx = next(
        i for i, h in enumerate(headers) if h is not None and "yarn" in str(h).casefold()
    )

    non_empty_rows = 0
    date_cells = 0
    date_value_counts: dict[str, int] = {}
    for row in rows_iter:
        if all(cell is None for cell in row):
            continue
        non_empty_rows += 1
        val = row[yarn_idx] if yarn_idx < len(row) else None
        if isinstance(val, datetime):
            date_cells += 1
            key = val.strftime("%Y-%m-%d")
            date_value_counts[key] = date_value_counts.get(key, 0) + 1
    wb.close()

    return {
        "file": raw_path.name,
        "size_bytes": raw_path.stat().st_size,
        "sha256": file_sha256(raw_path),
        "sheet_names": wb.sheetnames,
        "active_sheet": sheet_name,
        "headers": [str(h) if h is not None else None for h in headers],
        "non_empty_data_rows": non_empty_rows,
        "yarn_count_datetime_cells": date_cells,
        "yarn_count_datetime_breakdown": dict(
            sorted(date_value_counts.items(), key=lambda item: -item[1])
        ),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "textile" / "raw"
    report_dir = repo_root / "reports" / "textile"
    plot_dir = report_dir / "eda"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "weaving": audit_weaving(raw_dir / "full_weaving_dataset.csv", plot_dir),
        "knitting": audit_knitting(raw_dir / "knitting_dataset.csv", plot_dir),
        "excel": audit_excel(raw_dir / "raw_data_production_reports_20222.xlsx"),
    }
    out_path = report_dir / "eda_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
