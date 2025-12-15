import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import os

# -----------------------
# Utility helpers
# -----------------------
OUT_DIR = Path("data/processed/analysis_outputs") if 'Path' in globals() else None
if OUT_DIR is None:
    from pathlib import Path
    OUT_DIR = Path("data/processed/analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, fname):
    p = OUT_DIR / fname
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved:", p)

def safe_save_df(df_obj, fname):
    p = OUT_DIR / fname
    df_obj.to_csv(p, index=False)
    print("Saved:", p)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
INPUT = "data/processed/wide_seedling_data_cleaned.csv"
print("Loading:", INPUT)
df = pd.read_csv(INPUT)
print(f"Loaded dataset: {df.shape}")

# ------------------------------------------------------------
# 2. BASIC CLEANING
# ------------------------------------------------------------
# Trim column names
df.columns = [c.strip() for c in df.columns]

# Ensure numeric conversions for main measurement columns
df["height_cm"] = pd.to_numeric(df.get("height_cm", df.get("Height(cm)", np.nan)), errors="coerce")
df["diam_cm"] = pd.to_numeric(df.get("diam_cm", df.get("Diameter(cm)", np.nan)), errors="coerce")
df["canopy_score"] = pd.to_numeric(df.get("canopy_score", df.get("CanopyScore", np.nan)), errors="coerce")

# -------------------------
# Browsing normalization fix (robust)
# -------------------------
# If Browsed_norm exists, coerce it to numeric (many earlier scripts produced mixed types)
if "Browsed_norm" in df.columns:
    df["Browsed_norm"] = pd.to_numeric(df["Browsed_norm"], errors="coerce")
else:
    # Attempt to infer from other columns if available
    if "Browsed_clean" in df.columns:
        # map textual categories to numeric: Browsed -> 1, Not browsed -> 0
        df["Browsed_norm"] = df["Browsed_clean"].map({"Browsed": 1.0, "Not browsed": 0.0})
    else:
        # Fallback: try raw Browsed column heuristics
        if "Browsed" in df.columns:
            df["Browsed_norm"] = df["Browsed"].astype(str).str.strip().str.lower().map({
                "yes": 1.0, "y": 1.0, "b": 1.0, "browsed": 1.0,
                "no": 0.0, "n": 0.0, "not browsed": 0.0, "not": 0.0
            })
        else:
            df["Browsed_norm"] = np.nan

# Create a browsed binary flag safely (treat missing/unknown as NaN)
# Thresholding at >0 (anything positive means browsed), but keep NaN if original was NaN
df["browsed_bin"] = df["Browsed_norm"].apply(lambda v: 1 if pd.notna(v) and float(v) > 0 else (0 if pd.notna(v) and float(v) == 0 else np.nan))

# ------------------------------------------------------------
# 3. DATE PARSING (safe)
# ------------------------------------------------------------
if "Date" in df.columns:
    df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    print("Parsed Date -> Date_parsed (NA where invalid).")
else:
    df["Date_parsed"] = pd.NaT
    print("Warning: 'Date' column not found; Date_parsed set to NaT.")

# ------------------------------------------------------------
# 4. HEIGHT CLASSIFICATION (use numeric height_cm)
# ------------------------------------------------------------
def height_class(h):
    try:
        if pd.isna(h):
            return np.nan
        h = float(h)
    except Exception:
        return np.nan
    if h < 10:
        return "<10 cm"
    if h < 30:
        return "10–29 cm"
    if h < 100:
        return "30–99 cm"
    return "≥100 cm"

df["height_class"] = df["height_cm"].apply(height_class)

# ------------------------------------------------------------
# 5. OAK / SPECIES GROUPING (robust)
# ------------------------------------------------------------
# Use Species_code if exists else Species
if "Species_code" in df.columns:
    sp_series = df["Species_code"].astype(str).str.strip().str.upper()
else:
    sp_series = df["Species"].astype(str).str.strip().str.upper()

def classify_oak(code):
    if pd.isna(code) or code == "NAN" or code == "":
        return "non-oak"
    s = str(code).strip().upper()
    # explicit known mappings
    red_codes = {"QURU", "R", "QUVE", "QURU"}  # add more if needed
    white_codes = {"QUAL", "QUMU", "W"}
    if s in red_codes:
        return "red_oak"
    if s in white_codes:
        return "white_oak"
    if s.startswith("Q"):
        return "other_oak"
    return "non-oak"

df["oak_group"] = sp_series.apply(classify_oak)

# ------------------------------------------------------------
# 6. SITE CLASSIFICATION
# ------------------------------------------------------------
def classify_site(loc):
    if pd.isna(loc):
        return "Nature Park"
    s = str(loc).strip().lower()
    if "fern station" in s or s == "fern station":
        return "Fern Station"
    # treat blanks/unknowns as Nature Park per your earlier instruction
    if s in ("", "unknown", "unk", "na", "n/a"):
        return "Nature Park"
    return "Nature Park"  # all others -> Nature Park

df["Site"] = df["Location"].apply(classify_site)

# ------------------------------------------------------------
# 7. SAVE CLEAN SNAPSHOT
# ------------------------------------------------------------
safe_save_df(df, "analysis_ready_seedlings.csv")

# ------------------------------------------------------------
# 8. HEIGHT DISTRIBUTIONS & PLOTS
# ------------------------------------------------------------
# histogram
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df["height_cm"].dropna(), bins=40, kde=True, ax=ax)
ax.set_title("Height distribution (cm)")
ax.set_xlabel("Height (cm)")
save_fig(fig, "height_distribution.png")

# height classes
fig, ax = plt.subplots(figsize=(7,4))
order = ["<10 cm", "10–29 cm", "30–99 cm", "≥100 cm"]
sns.countplot(data=df, x="height_class", order=order, ax=ax)
ax.set_title("Height classes")
save_fig(fig, "height_classes.png")

# ------------------------------------------------------------
# 9. OAK GROUP COMPOSITION
# ------------------------------------------------------------
oak_counts = df["oak_group"].value_counts().reset_index()
oak_counts.columns = ["oak_group", "count"]
safe_save_df(oak_counts, "oak_group_counts.csv")

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=oak_counts, x="oak_group", y="count", ax=ax)
ax.set_title("Oak group counts")
save_fig(fig, "oak_group_counts.png")

# ------------------------------------------------------------
# 10. BROWSING PRESSURE
# ------------------------------------------------------------
browsing_summary = df.groupby(["Site", "oak_group"])["browsed_bin"].agg(['count', 'sum']).reset_index()
browsing_summary["prop_browsed"] = browsing_summary["sum"] / browsing_summary["count"]
safe_save_df(browsing_summary, "browsing_summary_site_oak.csv")

# Plot browsing proportion by oak_group (aggregated across sites)
agg = df.groupby("oak_group")["browsed_bin"].agg(['count','mean']).reset_index().rename(columns={"mean":"prop_browsed"})
safe_save_df(agg, "browsing_prop_by_oak_group.csv")
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=agg, x="oak_group", y="prop_browsed", ax=ax)
ax.set_ylabel("Proportion browsed (observed records)")
ax.set_title("Browsing proportion by oak group")
save_fig(fig, "browsing_prop_by_oak_group.png")

# ------------------------------------------------------------
# 11. CANOPY EFFECTS
# ------------------------------------------------------------
canopy_df = df.dropna(subset=["canopy_score", "height_cm"])
if len(canopy_df) >= 6:
    corr = canopy_df["canopy_score"].corr(canopy_df["height_cm"])
    with open(OUT_DIR / "canopy_height_corr.txt", "w") as f:
        f.write(f"Canopy vs height correlation: r = {corr:.4f}\n")
    # scatter + regplot
    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(data=canopy_df, x="canopy_score", y="height_cm", scatter_kws={"s":8}, ax=ax)
    ax.set_title("Seedling height vs Canopy Score")
    save_fig(fig, "seedling_height_vs_canopy.png")
else:
    print("Not enough canopy+height data to run canopy analysis (need >=6 complete rows).")

# ------------------------------------------------------------
# 12. RED OAK SEEDLING EXPLORATION (browsed vs not browsed)
# ------------------------------------------------------------
red_seed = df[(df["oak_group"] == "red_oak") & (df["height_cm"].notna())]
if len(red_seed) > 0:
    # basic summary
    red_summary = red_seed.groupby("Site")["browsed_bin"].agg(["count","mean"]).reset_index().rename(columns={"mean":"prop_browsed"})
    safe_save_df(red_summary, "red_seedling_browsing_by_site.csv")
    # boxplot heights by browsed status (if enough data)
    if red_seed["browsed_bin"].nunique() > 1 and red_seed.shape[0] >= 10:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(data=red_seed, x="browsed_bin", y="height_cm", ax=ax)
        ax.set_xticklabels(["Not browsed","Browsed"])
        ax.set_title("Red oak seedling height by browsed status")
        save_fig(fig, "red_seedling_height_by_browsed.png")
else:
    print("No red oak seedlings with height data found for deeper analysis.")

# ------------------------------------------------------------
# 13. TRANSITION / RECRUITMENT TABLE
# ------------------------------------------------------------
transition = df.groupby(["Site","oak_group","LifeStage"]).size().unstack(fill_value=0)
transition = transition.reset_index()
safe_save_df(transition, "transition_table_site_oak.csv")

# Plot simple life-stage funnels (counts) per site for red/white
for site in df["Site"].unique():
    subset = df[df["Site"] == site]
    fig, ax = plt.subplots(figsize=(7,4))
    for group in ["red_oak","white_oak","other_oak","non-oak"]:
        s = subset[subset["oak_group"]==group]["LifeStage"].value_counts().reindex(["Seedling","Sapling","Adult","Unknown"]).fillna(0)
        ax.plot(["Seedling","Sapling","Adult","Unknown"], s.values, marker="o", label=group)
    ax.set_title(f"Life-stage counts by oak group - {site}")
    ax.set_ylabel("Count")
    ax.legend()
    save_fig(fig, f"life_stage_funnel_{site.replace(' ','_')}.png")

# ------------------------------------------------------------
# 14. CANONICAL STATISTICAL TESTS (neutral framing)
# ------------------------------------------------------------
# Chi-square: seedling oak_group vs Site (seedlings only)
seedlings = df[df["LifeStage"] == "Seedling"].copy()
pivot = seedlings.pivot_table(index="oak_group", columns="Site", values="height_cm", aggfunc="count", fill_value=0)
if pivot.shape[0] > 1 and pivot.shape[1] > 1:
    try:
        chi2, p, dof, expected = stats.chi2_contingency(pivot.values)
        with open(OUT_DIR / "chi2_seedling_site_oak.txt", "w") as f:
            f.write(f"Chi2={chi2:.4f}, p={p:.6g}, dof={dof}\n")
            f.write("Expected:\n")
            f.write(np.array2string(expected))
        print("Saved chi-square result for seedling oak_group vs Site.")
    except Exception as e:
        print("Chi-square failed:", e)
else:
    print("Not enough categories for chi-square (seedlings).")

# T-test example: red-oak browsed vs not (if both groups exist)
red_seed_valid = red_seed.dropna(subset=["height_cm","browsed_bin"])
if red_seed_valid["browsed_bin"].nunique() > 1:
    a = red_seed_valid[red_seed_valid["browsed_bin"]==1]["height_cm"].dropna()
    b = red_seed_valid[red_seed_valid["browsed_bin"]==0]["height_cm"].dropna()
    if len(a) >= 2 and len(b) >= 2:
        tstat, pval = stats.ttest_ind(a,b, equal_var=False, nan_policy="omit")
        with open(OUT_DIR / "ttest_red_browsed_vs_not.txt","w") as f:
            f.write(f"T-statistic={tstat:.4f}, p={pval:.6g}\n")
        print("Saved t-test red browsed vs not.")
    else:
        print("Insufficient samples for T-test red browsed vs not.")

# ------------------------------------------------------------
# 15. BROWSING MODEL (logistic) — predict browsed_flag from height/canopy/site/species
# ------------------------------------------------------------
model_df = seedlings.dropna(subset=["browsed_bin"])
if model_df.shape[0] >= 40 and model_df["browsed_bin"].nunique() > 1:
    model_df["is_red"] = (model_df["oak_group"] == "red_oak").astype(int)
    model_df["site_fern"] = (model_df["Site"] == "Fern Station").astype(int)
    X = model_df[["height_cm","canopy_score","is_red","site_fern"]].fillna(0)
    y = model_df["browsed_bin"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = LogisticRegression(max_iter=400).fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    with open(OUT_DIR / "browsing_logistic_model.txt", "w") as f:
        f.write("Logistic model for browse probability\n")
        f.write(f"AUC: {auc:.4f}\nCoefficients:\n")
        for k,v in zip(X.columns, clf.coef_[0]):
            f.write(f"{k}: {v:.6f}\n")
    print("Saved logistic browsing model. AUC:", auc)
else:
    print("Not enough data to build a robust logistic browsing model (need >=40 seedlings with browse labels).")

# ------------------------------------------------------------
# 16. TIME SERIES / GROWTH TRAJECTORY
# ------------------------------------------------------------
if df["Date_parsed"].notna().sum() > 10:
    time_df = df.dropna(subset=["Date_parsed","height_cm"]).copy()
    monthly = time_df.groupby(pd.Grouper(key="Date_parsed", freq="M"))["height_cm"].median().reset_index()
    safe_save_df(monthly, "median_height_by_month.csv")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(monthly["Date_parsed"], monthly["height_cm"], marker="o")
    ax.set_title("Median seedling height by month")
    ax.set_ylabel("Median height (cm)")
    save_fig(fig, "median_height_by_month.png")
else:
    print("Not enough date+height data for time series analysis.")

# ------------------------------------------------------------
# 17. SPATIAL PREP (lat/lon parsing)
# ------------------------------------------------------------
def parse_coords(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    try:
        s2 = str(s).replace("(", "").replace(")", "").strip()
        parts = [p.strip() for p in s2.split(",")]
        if len(parts) >= 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return (lat, lon)
    except Exception:
        pass
    return (np.nan, np.nan)

df[["coord_lat","coord_lon"]] = df["Coordinates"].apply(lambda x: pd.Series(parse_coords(x)))
safe_save_df(df[["RowID","coord_lat","coord_lon"]], "lat_lon_prepped.csv")

# ------------------------------------------------------------
# 18. SNAPSHOT / CLEANED DATA SAVE
# ------------------------------------------------------------
safe_save_df(df, "annotated_dataset_snapshot.csv")

print("All steps complete. Check outputs in:", OUT_DIR)