#!/usr/bin/env python3
"""
analyze_seedling_data.py

Full analysis pipeline for the wide-format seedling dataset.

Outputs:
 - cleaned CSV: data/processed/wide_seedling_data_cleaned.csv
 - summary tables: data/processed/*.csv
 - plots: data/processed/analysis_outputs/*.png

Usage:
  python analyze_seedling_data.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# CONFIG / PATHS
# -----------------------
# The script will try these paths in order; update if the file lives elsewhere.
CANDIDATE_PATHS = [
    Path("data/processed/wide_seedling_data.xlsx"),
    Path("data/wide_seedling_data.xlsx"),
    Path("wide_seedling_data.xlsx"),
    Path("/mnt/data/wide_seedling_data.xlsx"),  # environment path used earlier
]

OUT_DIR = Path("data/processed/analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# UTIL: find input file
# -----------------------
def find_input_file():
    for p in CANDIDATE_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find wide_seedling_data.xlsx. Checked:\n" +
        "\n".join(str(p) for p in CANDIDATE_PATHS)
    )

# -----------------------
# Step 1: Load dataset
# -----------------------
input_path = find_input_file()
print("Loading data from:", input_path)
df = pd.read_excel(input_path)
print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# Quick copy so we don't accidentally overwrite original
df = df.copy()

# -----------------------
# Step 2: Normalize column names (help with small naming differences)
# -----------------------
# We'll work with canonical lowercase column keys for code clarity, but preserve originals.
col_map = {}
for c in df.columns:
    lc = c.strip()
    # common variants
    if lc.lower().replace(" ", "") in ("height(cm)", "heightcm", "height"):
        col_map[c] = "Height(cm)"
    elif "diam" in lc.lower() or "dbh" in lc.lower():
        col_map[c] = "Diameter(cm)"
    elif "canopy" in lc.lower() or "canop" in lc.lower() or "light" in lc.lower():
        col_map[c] = "CanopyScore"
    elif lc.lower() in ("species", "species_code", "sp"):
        col_map[c] = "Species"
    elif lc.lower() in ("location", "site"):
        col_map[c] = "Location"
    elif lc.lower() in ("browsed", "browse"):
        col_map[c] = "Browsed"
    elif lc.lower() in ("notes",):
        col_map[c] = "Notes"
    elif lc.lower() in ("coordinates", "coord", "transect_x", "transect_y"):
        col_map[c] = "Coordinates"
    elif lc.lower().replace(" ", "") == "rowid":
        col_map[c] = "RowID"
    else:
        # preserve as-is if no mapping
        col_map[c] = c

df = df.rename(columns=col_map)
print("Renamed columns:", df.columns.tolist())

# -----------------------
# Step 3: Basic cleaning
# -----------------------
# Standardize species and location strings, Browsed labels
df["Species"] = df.get("Species", "").astype(str).str.strip()
df["Location"] = df.get("Location", "").astype(str).str.strip()
# Normalize Browsed: expect values like Yes/No/B/ N / Unknown etc.
if "Browsed" in df.columns:
    df["Browsed_raw"] = df["Browsed"].astype(str).str.strip()
    df["Browsed_norm"] = df["Browsed_raw"].str.lower().str[0].map({
        "y": "Browsed",
        "b": "Browsed",  # if 'B' used
        "n": "Not browsed",
        "u": "Unknown",
        "": "Unknown"
    }).fillna(df["Browsed_raw"])  # keep original if weird
else:
    df["Browsed_norm"] = "Unknown"

# Numeric measurements
df["height_cm"] = pd.to_numeric(df.get("Height(cm)", pd.Series(np.nan)), errors="coerce")
df["diam_cm"] = pd.to_numeric(df.get("Diameter(cm)", pd.Series(np.nan)), errors="coerce")
df["canopy_score"] = pd.to_numeric(df.get("CanopyScore", pd.Series(np.nan)), errors="coerce")

# -----------------------
# Step 4: Life stage classification
# - Seedling: height < 200 cm (2 m)
# - Sapling: height >= 200 cm and diameter < 18 cm
# - Adult: diameter >= 18 cm
# -----------------------
def classify_lifestage(row):
    h = row["height_cm"]
    d = row["diam_cm"]
    # Adult if diameter >= 18
    if pd.notna(d) and d >= 18:
        return "Adult"
    # Sapling if height >= 200 and diameter < 18 (or diameter missing)
    if pd.notna(h) and h >= 200 and (pd.isna(d) or d < 18):
        return "Sapling"
    # Seedling if height < 200
    if pd.notna(h) and h < 200:
        return "Seedling"
    return "Unknown"

df["LifeStage"] = df.apply(classify_lifestage, axis=1)

# -----------------------
# Step 5: Species classification (red / white / other oak / non-oak)
# -----------------------
# Normalize codes to uppercase no-whitespace
df["Species_code"] = df["Species"].astype(str).str.strip().str.upper()

# Hard-coded maps from list (update if needed)
red_oak_codes = {"QURU", "QUVE", "R", "QUR", "QRU", "QURU"}  # expand if necessary
white_oak_codes = {"QUAL", "QUMU", "W", "QU MU".replace(" ","")}
add_red = {"QURU"} 
red_oak_codes |= add_red

def classify_oak_code(code):
    if not isinstance(code, str) or code.strip() == "" or code.strip().upper() == "NAN":
        return "non_oak"
    s = code.strip().upper()
    if s in red_oak_codes:
        return "red_oak"
    if s in white_oak_codes:
        return "white_oak"
    if s.startswith("Q"):
        return "other_oak"
    return "non_oak"

df["oak_group"] = df["Species_code"].apply(classify_oak_code)

# -----------------------
# Step 6: Site classification (Fern Station vs Nature Park)
# - Any location labeled 'fern station' (case-insensitive) -> Fern Station
# - All others (including blanks/unknown) -> Nature Park
# -----------------------
def classify_site(loc):
    if not isinstance(loc, str):
        return "Nature Park"
    if "fern station" in loc.strip().lower():
        return "Fern Station"
    return "Nature Park"

df["Site"] = df["Location"].apply(classify_site)

# -----------------------
# Step 7: Save a cleaned CSV for future reproducibility
# -----------------------
clean_out = Path("data/processed/wide_seedling_data_cleaned.csv")
df.to_csv(clean_out, index=False)
print("Saved cleaned CSV:", clean_out)

# -----------------------
# Helper util: safe groupby count table -> CSV
# -----------------------
def save_group_counts(df_, by, name):
    out = df_.groupby(by).size().reset_index(name="count")
    path = OUT_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    print("Saved:", path)
    return out

# -----------------------
# 1) Seedling abundance: counts by Site x oak_group (seedlings only)
# -----------------------
seedlings = df[df["LifeStage"] == "Seedling"].copy()
save_group_counts(seedlings, ["Site", "oak_group"], "seedling_counts_site_oak")

# Simple pivot table for quick view
seedling_pivot = seedlings.pivot_table(index="Site", columns="oak_group", values="height_cm", aggfunc="count", fill_value=0)
print("\nSeedling counts by Site x Oak group:\n", seedling_pivot)

# Plot seedling counts
plt.figure(figsize=(8,5))
seedling_pivot.plot(kind="bar", stacked=False)
plt.title("Seedling counts by Site and Oak Group")
plt.tight_layout()
plt.savefig(OUT_DIR / "seedling_counts_by_site_oak.png")
plt.close()

# -----------------------
# 2) Browsing pressure analysis
# - compute proportion browsed by Site and oak_group
# -----------------------
# Normalize browsed labels to two categories: 'Browsed' and 'Not browsed' (treat unknown as NaN)
df["Browsed_clean"] = df["Browsed_norm"].map({
    "Browsed": "Browsed",
    "Not browsed": "Not browsed",
    "Unknown": np.nan
}).fillna(np.nan)

browsing = df.groupby(["Site", "oak_group"])["Browsed_clean"].apply(
    lambda s: pd.Series({
        "n_total": s.size,
        "n_browsed": (s == "Browsed").sum(),
        "prop_browsed": (s == "Browsed").mean() if s.size>0 else np.nan
    })
).reset_index()
# Fix multiindex flattened
if isinstance(browsing.columns, pd.MultiIndex):
    browsing.columns = browsing.columns.map(str)
browsing_out = OUT_DIR / "browsing_by_site_oak.csv"
browsing.to_csv(browsing_out, index=False)
print("Saved browsing summary:", browsing_out)
print(browsing.head(20))

# Plot proportion browsed for red oak seedlings specifically
red_seed = df[(df["oak_group"] == "red_oak") & (df["LifeStage"] == "Seedling")].copy()
if len(red_seed) > 0:
    red_seed_group = red_seed.groupby("Site")["Browsed_clean"].apply(lambda s: (s=="Browsed").mean())
    plt.figure(figsize=(5,4))
    red_seed_group.plot(kind="bar")
    plt.ylabel("Proportion browsed")
    plt.title("Proportion of red-oak seedlings browsed (by Site)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "red_seedling_prop_browsed_by_site.png")
    plt.close()
    print("Saved red seedling browsing plot.")
else:
    print("No red seedling records for browsing plot.")

# -----------------------
# 3) Growth metrics (height for seedlings; diameter for saplings/adults)
# -----------------------
# Summary stats for seedling heights by Site and oak_group
height_summary = seedlings.groupby(["Site", "oak_group"])["height_cm"].agg(['count','median','mean','std']).reset_index()
height_summary.to_csv(OUT_DIR / "seedling_height_summary.csv", index=False)
print("Saved seedling height summary:", OUT_DIR / "seedling_height_summary.csv")

# Boxplot: red oak seedling height by Browsed_clean (across sites)
plt.figure(figsize=(7,5))
sns.boxplot(data=red_seed, x="Browsed_clean", y="height_cm")
plt.title("Red oak seedling height by Browsed status")
plt.xlabel("")
plt.ylabel("Height (cm)")
plt.tight_layout()
plt.savefig(OUT_DIR / "red_seedling_height_box_by_browsed.png")
plt.close()
print("Saved red seedling height boxplot.")

# For saplings/adults: diameter comparisons by Site and oak_group
saplings_adults = df[df["LifeStage"].isin(["Sapling","Adult"])].copy()
if not saplings_adults.empty:
    diam_summary = saplings_adults.groupby(["Site","oak_group"])["diam_cm"].agg(['count','median','mean','std']).reset_index()
    diam_summary.to_csv(OUT_DIR / "diameter_summary_sapling_adult.csv", index=False)
    print("Saved diameter summary:", OUT_DIR / "diameter_summary_sapling_adult.csv")

# -----------------------
# 4) Transition failure: counts of Seedling->Sapling->Adult by Site and OakGroup
# -----------------------
def stage_counts_by_site_oak(df_):
    return df_.groupby(["Site","oak_group","LifeStage"]).size().unstack(fill_value=0)

transition_table = stage_counts_by_site_oak(df)
# Save
transition_table.to_csv(OUT_DIR / "transition_table_site_oak.csv")
print("Saved transition table:", OUT_DIR / "transition_table_site_oak.csv")
print(transition_table.head(20))

# -----------------------
# 5) Canopy effects
# - correlation: canopy_score vs height (seedlings)
# - scatter + optional simple linear regression
# -----------------------
seedlings_with_canopy = seedlings.dropna(subset=["canopy_score","height_cm"]).copy()
if not seedlings_with_canopy.empty:
    corr = seedlings_with_canopy["canopy_score"].corr(seedlings_with_canopy["height_cm"])
    print("Correlation (canopy_score, height_cm) for seedlings:", corr)
    # scatter
    plt.figure(figsize=(6,4))
    sns.regplot(data=seedlings_with_canopy, x="canopy_score", y="height_cm", scatter_kws={'s':10})
    plt.title("Seedling height vs CanopyScore (regression line)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "seedling_height_vs_canopy_regplot.png")
    plt.close()
    print("Saved canopy vs height plot.")
else:
    print("Not enough canopy + height data to analyze canopy effects.")

# -----------------------
# 6) Optional statistical tests (neutral framing)
# - Example: test difference in red-oak seedling heights browsed vs not browsed
# - We print test results but do not 'prove' anything; interpretation is up to you.
# -----------------------
def run_ttest(groupA, groupB, labelA="A", labelB="B"):
    groupA = pd.Series(groupA).dropna()
    groupB = pd.Series(groupB).dropna()
    if len(groupA) < 2 or len(groupB) < 2:
        print(f"Not enough data to run t-test between {labelA} and {labelB} (nA={len(groupA)}, nB={len(groupB)})")
        return None
    res = stats.ttest_ind(groupA, groupB, equal_var=False, nan_policy="omit")
    out = {
        "nA": len(groupA),
        "nB": len(groupB),
        "meanA": groupA.mean(),
        "meanB": groupB.mean(),
        "t_stat": float(res.statistic),
        "pvalue": float(res.pvalue)
    }
    print(f"T-test {labelA} vs {labelB}:", out)
    return out

# run test for red-oak seedling height browsed vs not browsed (pooled across sites)
if len(red_seed) > 0:
    browsed_vals = red_seed[red_seed["Browsed_clean"]=="Browsed"]["height_cm"]
    not_browsed_vals = red_seed[red_seed["Browsed_clean"]=="Not browsed"]["height_cm"]
    run_ttest(browsed_vals, not_browsed_vals, "Red_Browsed", "Red_NotBrowsed")

# -----------------------
# 7) Save a few summary CSVs for inspection
# -----------------------
height_summary.to_csv(OUT_DIR / "seedling_height_summary.csv", index=False)
transition_table.to_csv(OUT_DIR / "transition_table_site_oak.csv")
pd.DataFrame(df["Species"].value_counts()).reset_index().to_csv(OUT_DIR / "species_counts.csv", index=False)

print("Saved final summaries and plots to:", OUT_DIR)
print("Analysis complete. Review CSVs and PNGs in the analysis_outputs folder and interpret the results carefully.")