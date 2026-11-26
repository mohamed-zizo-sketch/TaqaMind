"""Comprehensive Dataset Forensics and Exploratory Data Analysis for Food & Beverage Domain.

Generates empirical statistics, data quality diagnostics, and EDA figures
across all pre-downloaded raw datasets in data/food_beverage/raw/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

RAW_DIR = Path("data/food_beverage/raw")
REPORT_DIR = Path("reports/food_beverage")
FIG_DIR = REPORT_DIR / "figures"
RESEARCH_DIR = Path("research/food_beverage")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_brewery(out: dict) -> None:
    path = RAW_DIR / "brewery_data.csv"
    df = pd.read_csv(path)
    df["Brew_Date_dt"] = pd.to_datetime(df["Brew_Date"], errors="coerce")
    df["Total_Loss"] = (
        df["Loss_During_Brewing"]
        + df["Loss_During_Fermentation"]
        + df["Loss_During_Bottling_Kegging"]
    )

    stats = {
        "file": path.name,
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns) - 2),
        "column_names": [c for c in df.columns if c not in ("Brew_Date_dt", "Total_Loss")],
        "missing_counts": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "exact_duplicate_rows": int(df.drop(columns=["Brew_Date_dt", "Total_Loss"]).duplicated().sum()),
        "unique_batch_ids": int(df["Batch_ID"].nunique()),
        "date_range": [
            str(df["Brew_Date_dt"].min().date()),
            str(df["Brew_Date_dt"].max().date()),
        ],
        "categorical_cardinality": {
            "Beer_Style": df["Beer_Style"].value_counts().to_dict(),
            "SKU": df["SKU"].value_counts().to_dict(),
            "Location": df["Location"].value_counts().to_dict(),
            "Ingredient_Ratio_nunique": int(df["Ingredient_Ratio"].nunique()),
        },
        "numeric_summary": df[
            [
                "Fermentation_Time",
                "Temperature",
                "pH_Level",
                "Gravity",
                "Alcohol_Content",
                "Bitterness",
                "Color",
                "Volume_Produced",
                "Total_Sales",
                "Quality_Score",
                "Brewhouse_Efficiency",
                "Loss_During_Brewing",
                "Loss_During_Fermentation",
                "Loss_During_Bottling_Kegging",
                "Total_Loss",
            ]
        ]
        .describe()
        .round(4)
        .to_dict(),
        "zero_loss_counts": {
            "Loss_During_Brewing_zero": int((df["Loss_During_Brewing"] == 0).sum()),
            "Loss_During_Fermentation_zero": int((df["Loss_During_Fermentation"] == 0).sum()),
            "Loss_During_Bottling_Kegging_zero": int((df["Loss_During_Bottling_Kegging"] == 0).sum()),
            "Total_Loss_zero": int((df["Total_Loss"] == 0).sum()),
        },
        "correlations_with_total_loss": df.select_dtypes(include=[np.number])
        .corr()["Total_Loss"]
        .round(4)
        .to_dict(),
    }
    out["brewery_data"] = stats

    # Plot 1: Brewery target distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col, title in zip(
        axes.flat,
        [
            "Loss_During_Brewing",
            "Loss_During_Fermentation",
            "Loss_During_Bottling_Kegging",
            "Total_Loss",
        ],
        [
            "Loss During Brewing",
            "Loss During Fermentation",
            "Loss During Bottling/Kegging",
            "Sum of Stage Losses (Total_Loss)",
        ],
        strict=True,
    ):
        sns.histplot(df[col], bins=25, kde=True, ax=ax, color="#2b6cb0")
        ax.set_title(title)
        ax.set_xlabel("Reported Loss Value (unit unspecified in raw CSV)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "brewery_loss_distributions.png", dpi=150)
    plt.close()

    # Plot 2: Correlation heatmap
    num_cols = [
        "Fermentation_Time",
        "Temperature",
        "pH_Level",
        "Gravity",
        "Volume_Produced",
        "Quality_Score",
        "Brewhouse_Efficiency",
        "Loss_During_Brewing",
        "Loss_During_Fermentation",
        "Loss_During_Bottling_Kegging",
        "Total_Loss",
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Brewery Dataset Correlation Matrix (n=583)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "brewery_correlation_matrix.png", dpi=150)
    plt.close()

    # Plot 3: Group variation & temporal
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=df, x="Beer_Style", y="Total_Loss", ax=axes[0])
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Total_Loss by Beer Style")
    monthly = df.set_index("Brew_Date_dt").resample("ME")["Total_Loss"].mean()
    monthly.plot(ax=axes[1], marker="o", color="#c53030")
    axes[1].set_title("Monthly Mean Total_Loss Over Time")
    axes[1].set_ylabel("Mean Total_Loss")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "brewery_group_and_temporal.png", dpi=150)
    plt.close()


def analyze_dairy_energy(out: dict) -> None:
    f1 = RAW_DIR / "01_DATA_Figure_2_specific_energy_consumption.csv"
    f2 = RAW_DIR / "02_DATA_thermal_loads_in_kW.csv"
    f3 = RAW_DIR / "03_DATA_electric_boiler_consumption_in_kW.csv"

    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df3 = pd.read_csv(f3)

    df2["timestamp_dt"] = pd.to_datetime(df2["timestamp"])
    df3["timestamp_dt"] = pd.to_datetime(df3["timestamp"])

    out["dairy_energy"] = {
        "sec_monthly": {
            "file": f1.name,
            "sha256": sha256_file(f1),
            "rows": int(len(df1)),
            "columns": int(len(df1.columns)),
            "missing": {k: int(v) for k, v in df1.isna().sum().items() if v > 0},
            "year_months": df1["year_month"].tolist(),
            "processed_milk_m3_sum": float(df1["Processed milk [m^3]"].sum()),
            "water_m3_sum": float(df1["Water consumption [m^3]"].sum()),
        },
        "thermal_loads_1min": {
            "file": f2.name,
            "sha256": sha256_file(f2),
            "rows": int(len(df2)),
            "columns": int(len(df2.columns) - 1),
            "missing": {k: int(v) for k, v in df2.isna().sum().items() if v > 0},
            "time_range": [str(df2["timestamp_dt"].min()), str(df2["timestamp_dt"].max())],
            "negative_values_counts": {
                c: int((df2[c] < 0).sum())
                for c in df2.select_dtypes(include=[np.number]).columns
                if (df2[c] < 0).sum() > 0
            },
            "summary_kw": df2.select_dtypes(include=[np.number]).describe().round(2).to_dict(),
        },
        "electric_boiler_1h": {
            "file": f3.name,
            "sha256": sha256_file(f3),
            "rows": int(len(df3)),
            "columns": int(len(df3.columns) - 1),
            "missing": {k: int(v) for k, v in df3.isna().sum().items() if v > 0},
            "time_range": [str(df3["timestamp_dt"].min()), str(df3["timestamp_dt"].max())],
            "summary_kw": df3.select_dtypes(include=[np.number]).describe().round(2).to_dict(),
        },
    }

    # Plot Dairy Energy Profiles
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    df1.plot(x="year_month", y="Processed milk [m^3]", kind="bar", ax=axes[0], color="#2f855a", legend=False)
    axes[0].set_title("Norwegian Dairy: Monthly Processed Milk (m^3)")
    axes[0].tick_params(axis="x", rotation=45)

    hourly_thermal = df2.set_index("timestamp_dt")[
        ["Glycol Process Cooling [kW]", "Hot Water CIP [kW]", "Hot Water production [kW]"]
    ].resample("D").mean()
    hourly_thermal.plot(ax=axes[1], alpha=0.85)
    axes[1].set_title("Daily Mean Thermal Loads (kW) - 2024")
    axes[1].set_ylabel("Power (kW)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dairy_energy_profiles.png", dpi=150)
    plt.close()


def analyze_bakery_scheduling(out: dict) -> None:
    bakery_stats = {}
    for label, fname in [
        ("BK15", "Babor M-Hitzmann B-2022-Small and medium-sized bakery production data for scheduling-BK15.xlsx"),
        ("BK50", "Babor M-Hitzmann B-2022-Small and medium-sized bakery production data for scheduling-BK50.xlsx"),
    ]:
        path = RAW_DIR / fname
        desc = pd.read_excel(path, sheet_name="Description")
        df = pd.read_excel(path, sheet_name=label)
        bakery_stats[label] = {
            "file": fname,
            "sha256": sha256_file(path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
            "unique_products": int(df["product_id"].nunique()),
            "unique_machines": int(df["machine_id"].nunique()),
            "unique_stages": df["stage_name"].unique().tolist(),
            "total_batches": int(df["batches"].sum()) if "batches" in df else None,
            "duration_PT_summary": df["duration_PT"].describe().round(2).to_dict(),
            "description_rows": desc.dropna(how="all").to_dict(orient="records"),
        }
    out["bakery_scheduling"] = bakery_stats

    # Plot BK50 stage duration distribution
    path50 = RAW_DIR / "Babor M-Hitzmann B-2022-Small and medium-sized bakery production data for scheduling-BK50.xlsx"
    df50 = pd.read_excel(path50, sheet_name="BK50")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=df50, x="stage_name", y="duration_PT", ax=ax)
    ax.set_title("Bakery BK50: Processing Duration (min) by Production Stage")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bakery_bk50_stage_durations.png", dpi=150)
    plt.close()


def analyze_itac(out: dict) -> None:
    path = RAW_DIR / "ITAC_Database_20261124.xlsx"
    assess = pd.read_excel(path, sheet_name="ASSESS")
    recc = pd.read_excel(path, sheet_name="RECC")

    # Filter strictly to NAICS 311 (Food) and 312 (Beverage & Tobacco)
    assess["NAICS_str"] = assess["NAICS"].astype(str)
    fb_assess = assess[assess["NAICS_str"].str.startswith(("311", "312"))].copy()
    fb_ids = set(fb_assess["ID"])
    fb_recc = recc[recc["ID"].isin(fb_ids)].copy()

    out["itac_food_beverage"] = {
        "file": path.name,
        "sha256": sha256_file(path),
        "total_assess_rows": int(len(assess)),
        "total_recc_rows": int(len(recc)),
        "fb_naics_311_312_assess_rows": int(len(fb_assess)),
        "fb_naics_311_312_recc_rows": int(len(fb_recc)),
        "naics_top_subsectors": fb_assess["NAICS_str"].str[:4].value_counts().head(10).to_dict(),
        "fiscal_year_range": [
            int(fb_assess["FY"].min()) if fb_assess["FY"].notna().any() else None,
            int(fb_assess["FY"].max()) if fb_assess["FY"].notna().any() else None,
        ],
        "assess_columns_sample": list(fb_assess.columns)[:25],
        "recc_columns_sample": list(fb_recc.columns)[:25],
        "implemented_recc_count": int((fb_recc["IMPSTATUS"] == "I").sum()) if "IMPSTATUS" in fb_recc else None,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    analyze_brewery(out)
    analyze_dairy_energy(out)
    analyze_bakery_scheduling(out)
    analyze_itac(out)

    summary_path = RESEARCH_DIR / "forensics_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {summary_path} and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
