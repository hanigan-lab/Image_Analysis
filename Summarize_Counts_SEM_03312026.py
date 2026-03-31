import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib import cm  # for RdBu colormap
from typing import Optional

# =========================
# Auto input/output wiring
# =========================

# Use the directory where you ran:  python Summarize_Counts_02062026.py
RUN_DIR = Path.cwd()

# Find the input using partial name "_Image.csv"
matches = sorted(RUN_DIR.glob("*_Image.csv"))
if not matches:
    raise FileNotFoundError(
        f"No input CSV matching '*_Image.csv' found in: {RUN_DIR}\n"
        f"Put the combined CellProfiler Image CSV in this folder (or cd into it) and rerun."
    )
if len(matches) > 1:
    print("Warning: multiple '*_Image.csv' files found. Using the first (sorted):")
    for p in matches:
        print(f"  - {p.name}")
INPUT_CSV = matches[0]

# Make a new output directory in the run directory
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = RUN_DIR / f"cellprofiler_{stamp}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Outputs
OUTPUT_PER_DATE_PER_WELL = OUT_DIR / "combined_per_date_per_well.csv"
OUTPUT_PER_DATE_TOTALS   = OUT_DIR / "combined_per_date_totals.csv"

OUTPUT_RATIO_CSV         = OUT_DIR / "ratio_values.csv"
OUTPUT_NORMALIZED_CSV    = OUT_DIR / "normalized_ratios.csv"
OUTPUT_NORMALIZED_TO_8TH_CSV = OUT_DIR / "normalized_to_8th.csv"

PLOT_RATIO_PATH          = OUT_DIR / "ratio_all_wells.png"
PLOT_NORMALIZED_PATH     = OUT_DIR / "normalized_ratio_all_wells.png"
PLOT_NORMALIZED_TO_8TH_PATH = OUT_DIR / "normalized_to_8th_plot.png"

OUTPUT_GFP_NORM_CSV      = OUT_DIR / "normalized_gfp_counts.csv"
PLOT_GFP_NORM_PATH       = OUT_DIR / "normalized_gfp_counts_all_wells.png"
OUTPUT_MCH_NORM_CSV      = OUT_DIR / "normalized_mcherry_counts.csv"
PLOT_MCH_NORM_PATH       = OUT_DIR / "normalized_mcherry_counts_all_wells.png"

# Map wells to descriptive names for legend
WELL_LABELS = {
    "A1": "AAVS1",
    "A2": "PCNA",
    "A3": "OXA1L Biallelic",
    "B1": "OXA1L 5S Selective",
    "B2": "OXA1L 4S Selective",
    "B3": "No Virus Control",
}

# ----- Date parsing helpers -----
_RE_6DIG = re.compile(r"(\d{6})$")   # YYMMDD
_RE_8DIG = re.compile(r"(\d{8})$")   # could be MMDDYYYY or YYYYMMDD

def _parse_metadata_date_to_iso(val):
    """Return YYYY-MM-DD from Metadata_Date.

    Supports:
      - YYMMDD (e.g., 251114 -> 2025-11-14)
      - 8 digits (tries MMDDYYYY then YYYYMMDD)
    Accepts ints/floats/strings (e.g., 251114.0).
    """
    if pd.isna(val):
        return None
    s = str(val).strip().replace(".0", "")

    m6 = _RE_6DIG.search(s)
    if m6:
        yyMMdd = m6.group(1)
        try:
            return datetime.strptime(yyMMdd, "%y%m%d").strftime("%Y-%m-%d")
        except Exception:
            return None

    m8 = _RE_8DIG.search(s)
    if m8:
        d8 = m8.group(1)
        for fmt in ("%m%d%Y", "%Y%m%d"):
            try:
                return datetime.strptime(d8, fmt).strftime("%Y-%m-%d")
            except Exception:
                pass
        return None

    return None

def summarize_combined_csv(input_csv: Path) -> pd.DataFrame:
    """Read ONE big CSV and summarize by Date and well.

    If you have multiple images per well/day, we summarize *across images* using:
      - mean count per image (per Date×Well)
      - SEM across images

    Output columns (per Date×Well):
      - Date
      - Metadata_Well
      - Count_GFP_positive (mean across images)
      - SEM_GFP_positive
      - N_GFP_images
      - Count_mCherry_positive (mean across images)
      - SEM_mCherry_positive
      - N_mCherry_images
    """
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Cannot read {input_csv}: {e}")
        return pd.DataFrame(columns=[
            "Date", "Metadata_Well",
            "Count_GFP_positive", "SEM_GFP_positive", "N_GFP_images",
            "Count_mCherry_positive", "SEM_mCherry_positive", "N_mCherry_images",
        ])
    import os
    import re

    def extract_well_from_url(url):
        if pd.isna(url):
            return None
        fname = os.path.basename(str(url))
        m = re.search(r'_([A-H])(\d{2})f\d{2}d\d', fname)
        if not m:
            return None
        row = m.group(1)
        col = int(m.group(2))
        return f"{row}{col}"

    df["Metadata_Well"] = df["URL_GFP"].apply(extract_well_from_url)
    required = {"Metadata_Date", "Metadata_Well", "Count_GFP_positive", "Count_mCherry_positive"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Parse Metadata_Date -> Date (YYYY-MM-DD)
    df["Date"] = df["Metadata_Date"].apply(_parse_metadata_date_to_iso)
    bad_dates = df["Date"].isna().sum()
    if bad_dates:
        print(f"Warning: {bad_dates} rows have unparseable Metadata_Date and will be dropped.")
    df = df.dropna(subset=["Date"])

    # Numeric counts (per-image values)
    df["Count_GFP_positive"] = pd.to_numeric(df["Count_GFP_positive"], errors="coerce")
    df["Count_mCherry_positive"] = pd.to_numeric(df["Count_mCherry_positive"], errors="coerce")

    g = df.groupby(["Date", "Metadata_Well"], as_index=False)

    grouped = g.agg(
        Count_GFP_positive=("Count_GFP_positive", "mean"),
        SD_GFP_positive=("Count_GFP_positive", "std"),
        N_GFP_images=("Count_GFP_positive", "count"),
        Count_mCherry_positive=("Count_mCherry_positive", "mean"),
        SD_mCherry_positive=("Count_mCherry_positive", "std"),
        N_mCherry_images=("Count_mCherry_positive", "count"),
    )

    grouped["SEM_GFP_positive"] = grouped["SD_GFP_positive"] / np.sqrt(grouped["N_GFP_images"].replace(0, np.nan))
    grouped["SEM_mCherry_positive"] = grouped["SD_mCherry_positive"] / np.sqrt(grouped["N_mCherry_images"].replace(0, np.nan))

    grouped = grouped.drop(columns=["SD_GFP_positive", "SD_mCherry_positive"])
    grouped = grouped.dropna(subset=["Count_GFP_positive", "Count_mCherry_positive"], how="all")

    grouped = grouped.sort_values(["Date", "Metadata_Well"]).reset_index(drop=True)
    return grouped

def add_ratio_column(df: pd.DataFrame) -> pd.DataFrame:
    """Compute GFP / mCherry (safe divide; mCherry==0 -> NaN) and propagate SEM if available."""
    d = df.copy()
    for c in ["Count_GFP_positive", "Count_mCherry_positive", "SEM_GFP_positive", "SEM_mCherry_positive"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    G = d["Count_GFP_positive"]
    M = d["Count_mCherry_positive"].replace(0, np.nan)

    d["Ratio_GFP_to_mCherry"] = G / M

    # Propagate SEM for ratio if SEM columns exist:
    # sigma_R = R * sqrt( (sigma_G/G)^2 + (sigma_M/M)^2 )
    if "SEM_GFP_positive" in d.columns and "SEM_mCherry_positive" in d.columns:
        sigma_G = d["SEM_GFP_positive"]
        sigma_M = d["SEM_mCherry_positive"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.sqrt((sigma_G / G) ** 2 + (sigma_M / M) ** 2)
            d["SEM_Ratio_GFP_to_mCherry"] = d["Ratio_GFP_to_mCherry"] * rel
    return d

def normalize_ratio_per_well(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each well’s ratio to its first valid value (chronologically) via subtraction.

    SEM handling:
      Norm = R - R0
      SEM_Norm = sqrt( SEM_R^2 + SEM_R0^2 )  (assumes independence; conservative)
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values(["Metadata_Well", "Date"])

    def _norm(group):
        group = group.sort_values("Date").copy()
        valid_mask = group["Ratio_GFP_to_mCherry"].notna()
        if not valid_mask.any():
            group["Norm_Ratio_GFP_to_mCherry"] = np.nan
            if "SEM_Ratio_GFP_to_mCherry" in group.columns:
                group["SEM_Norm_Ratio_GFP_to_mCherry"] = np.nan
            return group

        first_idx = group.index[valid_mask][0]
        base_val = group.loc[first_idx, "Ratio_GFP_to_mCherry"]
        group["Norm_Ratio_GFP_to_mCherry"] = group["Ratio_GFP_to_mCherry"] - base_val

        if "SEM_Ratio_GFP_to_mCherry" in group.columns:
            base_sem = group.loc[first_idx, "SEM_Ratio_GFP_to_mCherry"]
            group["SEM_Norm_Ratio_GFP_to_mCherry"] = np.sqrt(
                group["SEM_Ratio_GFP_to_mCherry"] ** 2 + (base_sem ** 2)
            )
        return group

    return d.groupby("Metadata_Well", group_keys=False).apply(_norm)

def normalize_counts_per_well(df: pd.DataFrame, count_col: str, sem_col: str,
                              out_col: str, out_sem_col: str) -> pd.DataFrame:
    """For each well, subtract the first non-NaN value (chronologically).

    SEM handling:
      Norm = X - X0
      SEM_Norm = sqrt( SEM_X^2 + SEM_X0^2 )  (assumes independence; conservative)
    """
    d = df.copy()
    d[out_col] = np.nan
    d[out_sem_col] = np.nan

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values(["Metadata_Well", "Date"])
    d[count_col] = pd.to_numeric(d[count_col], errors="coerce")
    if sem_col in d.columns:
        d[sem_col] = pd.to_numeric(d[sem_col], errors="coerce")

    if d.empty:
        return d

    def _norm(group):
        group = group.sort_values("Date").copy()
        base_series = group[count_col].dropna()
        if base_series.empty:
            group[out_col] = np.nan
            group[out_sem_col] = np.nan
            return group

        first_idx = base_series.index[0]
        base = group.loc[first_idx, count_col]
        group[out_col] = group[count_col] - base

        if sem_col in group.columns:
            base_sem = group.loc[first_idx, sem_col]
            group[out_sem_col] = np.sqrt(group[sem_col] ** 2 + (base_sem ** 2))
        else:
            group[out_sem_col] = np.nan
        return group

    return d.groupby("Metadata_Well", group_keys=False).apply(_norm)

def plot_all_wells(df: pd.DataFrame, column: str, ylabel: str, title: str,
                   outpath: Path, normalized: bool = False, exclude_wells=None,
                   err_col: Optional[str] = None, err_alpha: float = 0.22):
    """Generic plotting function for any ratio/count column with optional SEM bands."""
    if df.empty:
        print("No data to plot.")
        return
    if column not in df.columns:
        print(f"Column '{column}' not found; skipping plot {outpath}.")
        return

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date", column])

    if err_col is not None and err_col in d.columns:
        d[err_col] = pd.to_numeric(d[err_col], errors="coerce")

    if exclude_wells:
        d = d[~d["Metadata_Well"].isin(exclude_wells)]
    if d.empty:
        print(f"No valid {column} data to plot after exclusions.")
        return

    wells = sorted(d["Metadata_Well"].unique())

    colors = ['steelblue', 'tomato', 'gold', 'lightskyblue', 'firebrick', 'gray']
    if len(wells) > len(colors):
        cmap = cm.get_cmap("RdBu_r", len(wells))
        colors = [cmap(i) for i in range(len(wells))]

    plt.figure(figsize=(9, 5.5))
    for i, well in enumerate(wells):
        sub = d[d["Metadata_Well"] == well].sort_values("Date")
        label = WELL_LABELS.get(well, well)

        x = sub["Date"]
        y = sub[column].astype(float)

        plt.plot(
            x, y,
            marker="o", label=label, linewidth=3, markersize=10,
            color=colors[i], markeredgecolor="black", markeredgewidth=1
        )

        if err_col is not None and err_col in sub.columns:
            e = sub[err_col].astype(float)
            mask = np.isfinite(y) & np.isfinite(e)
            if mask.any():
                plt.fill_between(
                    x[mask],
                    (y - e)[mask],
                    (y + e)[mask],
                    color=colors[i],
                    alpha=err_alpha,
                    linewidth=0
                )

    plt.xlabel("Day", fontsize=18, fontname="Arial")
    plt.ylabel(ylabel, fontsize=18, fontname="Arial")
    plt.title(title, fontsize=20, fontname="Arial", fontweight="bold", pad=20)
    plt.grid(True)

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.margins(x=0.1, y=0.2)

    if normalized:
        plt.axhline(0.0, linestyle="--", color="gray")

    plt.legend(title="guide RNA", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved plot: {outpath}")

def normalize_to_specific_date(df: pd.DataFrame, column: str, sem_col: Optional[str],
                               out_col: str, out_sem_col: str,
                               target_date_str: str) -> pd.DataFrame:
    """Normalize each well’s values to the value on a specific date via subtraction.

    SEM handling:
      Norm = X - Xref
      SEM_Norm = sqrt( SEM_X^2 + SEM_Xref^2 )  (assumes independence; conservative)
    """
    d = df.copy()
    d[out_col] = np.nan
    d[out_sem_col] = np.nan

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d[column] = pd.to_numeric(d[column], errors="coerce")
    if sem_col is not None and sem_col in d.columns:
        d[sem_col] = pd.to_numeric(d[sem_col], errors="coerce")

    target_date = pd.to_datetime(target_date_str)

    def _norm(group):
        group = group.sort_values("Date").copy()
        ref_row = group.loc[group["Date"] == target_date]
        if ref_row.empty or pd.isna(ref_row.iloc[0][column]):
            group[out_col] = np.nan
            group[out_sem_col] = np.nan
            return group

        ref_val = ref_row.iloc[0][column]
        group[out_col] = group[column] - ref_val

        if sem_col is not None and sem_col in group.columns:
            ref_sem = ref_row.iloc[0][sem_col]
            group[out_sem_col] = np.sqrt(group[sem_col] ** 2 + (ref_sem ** 2))
        else:
            group[out_sem_col] = np.nan
        return group

    return d.groupby("Metadata_Well", group_keys=False).apply(_norm)

if __name__ == "__main__":
    print(f"Run directory: {RUN_DIR}")
    print(f"Input CSV:     {INPUT_CSV}")
    print(f"Output dir:    {OUT_DIR}")

    # 1) Per-date/per-well means + SEMs (from one combined CSV)
    per_date_per_well = summarize_combined_csv(INPUT_CSV)
    per_date_per_well.to_csv(OUTPUT_PER_DATE_PER_WELL, index=False)

    # 2) Per-date totals (sum across wells) + conservative SEM combine across wells
    if not per_date_per_well.empty:
        per_date_totals = (
            per_date_per_well.groupby("Date", as_index=False).agg(
                Count_GFP_positive=("Count_GFP_positive", "sum"),
                Count_mCherry_positive=("Count_mCherry_positive", "sum"),
                SEM_GFP_positive=("SEM_GFP_positive", lambda x: np.sqrt(np.nansum(np.square(x)))),
                SEM_mCherry_positive=("SEM_mCherry_positive", lambda x: np.sqrt(np.nansum(np.square(x)))),
            ).sort_values("Date")
        )
    else:
        per_date_totals = pd.DataFrame(columns=[
            "Date", "Count_GFP_positive", "Count_mCherry_positive",
            "SEM_GFP_positive", "SEM_mCherry_positive"
        ])
    per_date_totals.to_csv(OUTPUT_PER_DATE_TOTALS, index=False)

    # 3) Ratios (GFP/mCherry) + SEM propagation
    ratio_df = add_ratio_column(per_date_per_well)
    ratio_df["Date"] = pd.to_datetime(ratio_df["Date"], errors="coerce")
    ratio_df.to_csv(OUTPUT_RATIO_CSV, index=False)

    # 4) Normalized ratios (baseline = first valid per well)
    norm_ratio_df = normalize_ratio_per_well(ratio_df)
    norm_ratio_df.to_csv(OUTPUT_NORMALIZED_CSV, index=False)

    # 5) Normalized counts (baseline = first valid per well)
    gfp_norm_df = normalize_counts_per_well(
        per_date_per_well,
        count_col="Count_GFP_positive",
        sem_col="SEM_GFP_positive",
        out_col="Norm_GFP_Count",
        out_sem_col="SEM_Norm_GFP_Count"
    )
    gfp_norm_df.to_csv(OUTPUT_GFP_NORM_CSV, index=False)

    mch_norm_df = normalize_counts_per_well(
        per_date_per_well,
        count_col="Count_mCherry_positive",
        sem_col="SEM_mCherry_positive",
        out_col="Norm_mCherry_Count",
        out_sem_col="SEM_Norm_mCherry_Count"
    )
    mch_norm_df.to_csv(OUTPUT_MCH_NORM_CSV, index=False)

    # 6) Plots (with SEM bands)
    plot_all_wells(
        ratio_df, "Ratio_GFP_to_mCherry",
        ylabel="GFP / mCherry",
        title="GFP/mCherry Ratio Over Time (Raw)",
        outpath=PLOT_RATIO_PATH,
        normalized=False,
        err_col="SEM_Ratio_GFP_to_mCherry"
    )

    plot_all_wells(
        norm_ratio_df, "Norm_Ratio_GFP_to_mCherry",
        ylabel="Normalized GFP/mCherry (day=0)",
        title="Normalized GFP/mCherry Ratio Over Time (day=0)",
        outpath=PLOT_NORMALIZED_PATH,
        normalized=True,
        err_col="SEM_Norm_Ratio_GFP_to_mCherry"
    )

    # Remove No Virus Control from GFP-positive plot only (B3, not B03)
    plot_all_wells(
        gfp_norm_df, "Norm_GFP_Count",
        ylabel="Normalized GFP-positive count",
        title="Normalized GFP-positive Counts Over Time",
        outpath=PLOT_GFP_NORM_PATH,
        normalized=True,
        exclude_wells=["B3"],
        err_col="SEM_Norm_GFP_Count"
    )

    plot_all_wells(
        mch_norm_df, "Norm_mCherry_Count",
        ylabel="Normalized mCherry-positive count",
        title="Normalized mCherry-positive Counts Over Time",
        outpath=PLOT_MCH_NORM_PATH,
        normalized=True,
        err_col="SEM_Norm_mCherry_Count"
    )

    print("\nOutputs written:")
    print(f"  Output directory:               {OUT_DIR}")
    print(f"  Counts per date/well:           {OUTPUT_PER_DATE_PER_WELL}")
    print(f"  Totals per date:                {OUTPUT_PER_DATE_TOTALS}")
    print(f"  Raw ratio data:                 {OUTPUT_RATIO_CSV}")
    print(f"  Normalized ratio data:          {OUTPUT_NORMALIZED_CSV}")
    print(f"  Normalized GFP count data:      {OUTPUT_GFP_NORM_CSV}")
    print(f"  Normalized mCherry count data:  {OUTPUT_MCH_NORM_CSV}")
    print(f"  Raw ratio plot:                 {PLOT_RATIO_PATH}")
    print(f"  Normalized ratio plot:          {PLOT_NORMALIZED_PATH}")
    print(f"  Normalized GFP count plot:      {PLOT_GFP_NORM_PATH}")
    print(f"  Normalized mCherry count plot:  {PLOT_MCH_NORM_PATH}")

    # --- Normalize ratios to the value on a specific day-of-month (kept your existing ref-day=8 behavior) ---
    dates_day = ratio_df["Date"].dropna().dt.day
    if (dates_day == 18).any():
        date_ref = ratio_df.loc[ratio_df["Date"].dt.day == 18, "Date"].iloc[0]
        date_ref_str = date_ref.strftime("%Y-%m-%d")

        norm_to_ref_df = normalize_to_specific_date(
            ratio_df,
            column="Ratio_GFP_to_mCherry",
            sem_col="SEM_Ratio_GFP_to_mCherry",
            out_col="Norm_Ratio_to_8th",
            out_sem_col="SEM_Norm_Ratio_to_8th",
            target_date_str=date_ref_str
        )
        norm_to_ref_df.to_csv(OUTPUT_NORMALIZED_TO_8TH_CSV, index=False)

        plot_all_wells(
            norm_to_ref_df,
            "Norm_Ratio_to_8th",
            ylabel="Normalized GFP/mCherry (ref day=8)",
            title="Normalized GFP/mCherry Ratio Over Time (ref day=8)",
            outpath=PLOT_NORMALIZED_TO_8TH_PATH,
            normalized=True,
            err_col="SEM_Norm_Ratio_to_8th"
        )

        print(f"  Normalized-to-ref ratio data:   {OUTPUT_NORMALIZED_TO_8TH_CSV}")
        print(f"  Normalized-to-ref plot:         {PLOT_NORMALIZED_TO_8TH_PATH}")
    else:
        print("No measurement on the chosen reference day-of-month — skipping reference normalization plot.")
