#!/usr/bin/env python3
"""
full_analysis.py
Comprehensive analysis pipeline for seedling data.

Inputs:
 - data/processed/wide_seedling_data_cleaned.csv

Outputs (data/processed/analysis_outputs/*):
 - CSV summaries
 - PNG plots
 - Simple models saved as text
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# Optional spatial libs (only used if coordinates parse ok)
try:
    import geopandas as gpd
    from shapely.geometry import Point
    SPATIAL_AVAILABLE = True
except Exception:
    SPATIAL_AVAILABLE = False

# -----------------------
# CONFIG
# -----------------------
INPUT = Path("data/processed/wide_seedling_data_cleaned.csv")
OUT_DIR = Path("data/processed/analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# HELPERS
# -----------------------
def safe_read(path):
    print("Reading:", path)
    return pd.read_csv(path)

def save_fig(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved", p)

# -----------------------
# LOAD
# -----------------------
df = safe_read(INPUT)
print("Loaded shape:", df.shape)

# Ensure numeric columns exist and normalized names
if "height_cm" not in df.columns:
    df["height_cm"] = pd.to_numeric(df.get("Height(cm)", df.get("Height", np.nan)), errors="coerce")
if "diam_cm" not in df.columns:
    df["diam_cm"] = pd.to_numeric(df.get("Diameter(cm)", df.get("Diameter", np.nan)), errors="coerce")
if "canopy_score" not in df.columns:
    df["canopy_score"] = pd.to_numeric(df.get("CanopyScore", np.nan), errors="coerce")

# Normalize browsed column to 'Browsed'/'Not browsed'/NaN
if "Browsed_clean" in df.columns:
    df["browsed_flag"] = df["Browsed_clean"].map({"Browsed":1, "Not browsed":0})
else:
    df["browsed_flag"] = df.get("Browsed", "").astype(str).str.lower().str.startswith("y").astype(int)

# Normalize oak_group and site (assume present)
df["oak_group"] = df["oak_group"].fillna("non_oak")
df["Site"] = df["Site"].fillna("Nature Park")

# Quick filter copies
seedlings = df[df["LifeStage"].str.lower().eq("seedling")]
saplings = df[df["LifeStage"].str.lower().eq("sapling")]
adults = df[df["LifeStage"].str.lower().eq("adult")]

# -----------------------
# 1) Species composition & basic summaries
# -----------------------
species_counts = df["Species"].value_counts().reset_index()
species_counts.columns = ["Species","count"]
species_counts["prop"] = species_counts["count"] / len(df)
species_counts.to_csv(OUT_DIR / "species_counts_full.csv", index=False)
print("Species counts saved.")

# Top species plot
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(data=species_counts.head(15), x="Species", y="count", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_title("Top 15 species by record count")
save_fig(fig, "species_top15.png")

# -----------------------
# 2) Seedling abundance (counts and densities)
# -----------------------
seedling_counts = seedlings.groupby(["Site","oak_group"]).size().reset_index(name="count")
seedling_counts.to_csv(OUT_DIR / "seedling_counts_site_oak.csv", index=False)

# Plot stacked by site
pivot = seedling_counts.pivot(index="Site", columns="oak_group", values="count").fillna(0)
fig = pivot.plot(kind="bar", stacked=False, figsize=(8,5)).get_figure()
save_fig(fig, "seedling_counts_by_site_oak.png")

# -----------------------
# 3) Browsing pressure analysis
# -----------------------
# Proportion browsed by Site and oak_group
browsing = df.groupby(["Site","oak_group"])["browsed_flag"].agg(['count','sum']).reset_index()
browsing["prop_browsed"] = browsing["sum"]/browsing["count"]
browsing.to_csv(OUT_DIR / "browsing_summary.csv", index=False)
print("Browsing summary saved.")

# Plot proportion for red oak seedlings
red_seed = seedlings[seedlings["oak_group"]=="red_oak"]
if not red_seed.empty:
    prop = red_seed.groupby("Site")["browsed_flag"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.barplot(data=prop, x="Site", y="browsed_flag", ax=ax)
    ax.set_ylabel("Proportion browsed")
    ax.set_title("Proportion of red oak seedlings browsed by Site")
    save_fig(fig, "red_seedling_prop_browsed_by_site.png")

# -----------------------
# 4) Growth metrics
# -----------------------
# Seedling heights summary
height_summary = seedlings.groupby(["Site","oak_group"])["height_cm"].agg(['count','median','mean','std']).reset_index()
height_summary.to_csv(OUT_DIR / "seedling_height_summary.csv", index=False)

# Sapling/adult diameter summary
diam_summary = df[df["LifeStage"].isin(["Sapling","Adult"])].groupby(["Site","oak_group"])["diam_cm"].agg(['count','median','mean','std']).reset_index()
diam_summary.to_csv(OUT_DIR / "diameter_summary_by_stage.csv", index=False)

# Boxplot: red oak seedlings height by browsed
if not red_seed.empty:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=red_seed, x="browsed_flag", y="height_cm", ax=ax)
    ax.set_xticklabels(["Not browsed","Browsed"])
    ax.set_title("Red oak seedling height: browsed vs not browsed")
    save_fig(fig, "red_seedling_height_box_browsed.png")

# -----------------------
# 5) Transition (recruitment funnel)
# -----------------------
# counts per stage per site per oak group
transition = df.groupby(["Site","oak_group","LifeStage"]).size().unstack(fill_value=0)
transition.to_csv(OUT_DIR / "transition_table_site_oak.csv")
print("Transition table saved.")

# Visualize funnel for red oak
red_trans = transition.loc[:,["red_oak"]] if "red_oak" in transition.columns else None
# (we can create a small bar plot of Seedling/Sapling/Adult counts per site)
for site in df["Site"].unique():
    subset = df[(df["Site"]==site) & (df["oak_group"]=="red_oak")]
    counts = subset["LifeStage"].value_counts()
    fig, ax = plt.subplots(figsize=(5,4))
    counts.reindex(["Seedling","Sapling","Adult","Unknown"]).fillna(0).plot.bar(ax=ax)
    ax.set_title(f"Red oak life-stage counts: {site}")
    save_fig(fig, f"red_oak_funnel_{site.replace(' ','_')}.png")

# -----------------------
# 6) Canopy effects
# -----------------------
# We will use seedlings with both canopy and height present
canopy_df = seedlings.dropna(subset=["canopy_score","height_cm"])
if len(canopy_df) >= 10:
    corr = canopy_df["canopy_score"].corr(canopy_df["height_cm"])
    print("Canopy-height corr (seedlings):", corr)
    # regression
    X = sm.add_constant(canopy_df["canopy_score"])
    model = sm.OLS(canopy_df["height_cm"], X, missing='drop').fit()
    with open(OUT_DIR / "canopy_height_regression.txt","w") as f:
        f.write(model.summary().as_text())
    # plot
    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(data=canopy_df, x="canopy_score", y="height_cm", scatter_kws={'s':8})
    ax.set_title("Seedling height vs canopy score (regression)")
    save_fig(fig, "seedling_height_vs_canopy.png")
else:
    print("Not enough canopy data to analyze (need >=10 complete rows)")

# -----------------------
# 7) Browsing model (logistic): predict browsed (1/0) from height, site, canopy, species
# -----------------------
# Prepare data
model_df = seedlings.copy()
# keep rows where we have browsed_flag (0/1)
model_df = model_df[model_df["browsed_flag"].notna()]
# features
model_df["is_red"] = (model_df["oak_group"]=="red_oak").astype(int)
model_df["site_is_fern"] = (model_df["Site"]=="Fern Station").astype(int)
X = model_df[["height_cm","canopy_score","is_red","site_is_fern"]].fillna(0)
y = model_df["browsed_flag"].astype(int)
if len(y.unique())>1 and len(model_df) >= 30:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = LogisticRegression(max_iter=200).fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    with open(OUT_DIR / "browsing_model_report.txt","w") as f:
        f.write("Logistic Regression for browsing (predicting browsed=1)\n")
        f.write("AUC: %.3f\n" % auc)
        f.write("Coefficients:\n")
        for k,v in zip(X.columns, clf.coef_[0]):
            f.write(f"{k}: {v}\n")
    print("Saved browsing model report. AUC:", auc)
else:
    print("Not enough data or only one class to build browsing model.")

# -----------------------
# 8) Growth trajectories (simple): height quantiles by date
# -----------------------
# Parse date if possible
try:
    df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
except Exception:
    df["Date_parsed"] = pd.NaT

time_df = seedlings.dropna(subset=["Date_parsed","height_cm"]).copy()
if not time_df.empty:
    q = time_df.groupby(pd.Grouper(key="Date_parsed", freq="M"))["height_cm"].quantile([0.25,0.5,0.75]).unstack()
    q.to_csv(OUT_DIR / "height_quantiles_by_month.csv")
    # plot median over time
    med = time_df.groupby(pd.Grouper(key="Date_parsed", freq="M"))["height_cm"].median()
    fig, ax = plt.subplots(figsize=(8,4))
    med.plot(ax=ax)
    ax.set_ylabel("Median seedling height (cm)")
    ax.set_title("Median seedling height over time")
    save_fig(fig, "median_seedling_height_over_time.png")
else:
    print("Not enough date+height data for growth trajectories.")

# -----------------------
# 9) Spatial clustering (if coordinates parse)
# -----------------------
# Attempt to parse Coordinates stored as "x,y" or "(x, y)"
def parse_coords_str(s):
    if pd.isna(s): return (np.nan,np.nan)
    try:
        s2 = str(s).strip().replace("(", "").replace(")","")
        parts = [p.strip() for p in s2.split(",")]
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return (np.nan, np.nan)

df[["lat","lon"]] = df["Coordinates"].apply(lambda s: pd.Series(parse_coords_str(s)))
if SPATIAL_AVAILABLE and df[["lat","lon"]].dropna().shape[0] > 10:
    gdf = gpd.GeoDataFrame(df.dropna(subset=["lat","lon"]), geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])])
    gdf.to_file(OUT_DIR / "seedlings_points.geojson", driver="GeoJSON")
    print("Saved GeoJSON for mapping.")
else:
    print("Spatial libs not available or not enough coordinate data; saved lat/lon CSV")
    df[["lat","lon"]].to_csv(OUT_DIR / "lat_lon.csv", index=False)

# -----------------------
# FINISH
# -----------------------
print("Full analysis finished. Outputs in:", OUT_DIR)
