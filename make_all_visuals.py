#!/usr/bin/env python3
"""
make_all_visualizations.py

Generates a suite of publication-style visualizations for the oak seedling project.
Outputs saved to data/processed/analysis_outputs/visualizations/

Visualizations produced:
1) Browsing probability by Site (Fern Station vs Nature Park) with 95% CIs
2) Red oak height distributions by Site (boxplot + violin + KDE)
3) Browsing probability vs Height for seedlings (site-colored logistic curves)
4) Species composition comparison (stacked bar / top species)
5) Oak recruitment funnel (Seedling -> Sapling -> Adult) per Site
6) Sapling/juvenile height by browsed status (boxplot) — for larger plants
Plus simple statistics saved to CSV / text.

Requires: pandas, numpy, matplotlib, seaborn, scipy, sklearn, statsmodels
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
CLEANED_CSV = Path("data/processed/wide_seedling_data_cleaned.csv")
OUT_DIR = Path("data/processed/analysis_outputs/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid", context="notebook", rc={"figure.dpi": 150})

# ---------- LOAD & PREP ----------
if not CLEANED_CSV.exists():
    raise FileNotFoundError(f"Cleaned CSV not found: {CLEANED_CSV}")

df = pd.read_csv(CLEANED_CSV)
print("Loaded:", CLEANED_CSV, "shape:", df.shape)

# Ensure clean numeric columns
df["height_cm"] = pd.to_numeric(df.get("height_cm", df.get("Height(cm)", np.nan)), errors="coerce")
df["diam_cm"] = pd.to_numeric(df.get("diam_cm", df.get("Diameter(cm)", np.nan)), errors="coerce")
df["canopy_score"] = pd.to_numeric(df.get("canopy_score", df.get("CanopyScore", np.nan)), errors="coerce")

# ---------- Robust browse parsing ----------
# Find candidate columns
browse_candidates = [c for c in df.columns if ("brows" in c.lower()) or ("browse" in c.lower())]

browse_col = None
preferred = ["browsed_bin", "browsed", "Browsed", "browse", "Browse", "browse_status", "Browsed_clean", "Browsed_norm"]
for p in preferred:
    if p in df.columns:
        browse_col = p
        break

if browse_col is None and browse_candidates:
    browse_col = browse_candidates[0]

df["browsed_bin"] = np.nan

if browse_col is not None:
    s = df[browse_col].astype(str).str.strip().str.lower()

    # normalize common “empty/unknown” tokens
    s = s.replace({
        "": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan,
        "unknown": np.nan, "unk": np.nan, "na": np.nan, "n/a": np.nan
    })

    browse_map = {
        "y": 1, "yes": 1, "true": 1, "t": 1, "1": 1, "browsed": 1, "browse": 1,
        "n": 0, "no": 0, "false": 0, "f": 0, "0": 0, "not browsed": 0,
        "unbrowsed": 0, "not_browsed": 0
    }

    df["browsed_bin"] = s.map(browse_map).astype(float)

    # fallback: startswith logic for messy entries
    still_nan = df["browsed_bin"].isna() & s.notna()
    df.loc[still_nan & s.str.startswith(("y", "b")), "browsed_bin"] = 1.0
    df.loc[still_nan & s.str.startswith(("n", "u")), "browsed_bin"] = 0.0

    print(f"\nUsing browse column: {browse_col}")
    print("Browse raw value counts (top 20):")
    print(s.value_counts(dropna=False).head(20))
    print("\nParsed browsed_bin counts:")
    print(df["browsed_bin"].value_counts(dropna=False))
else:
    print("\nNo browse column detected; browsed_bin will remain NaN.")

# Oak group and Site should exist from preprocessing; fallback safe columns
if "oak_group" not in df.columns:
    df["oak_group"] = df.get("Species_code", df.get("Species", "")).astype(str).apply(
        lambda s: "red_oak" if str(s).upper() in ("QURU", "R", "QUVE")
        else ("white_oak" if str(s).upper().startswith("QU") and "W" in str(s).upper()
              else "non_oak")
    )

if "Site" not in df.columns:
    if "Location" in df.columns:
        df["Site"] = df["Location"].apply(
            lambda s: "Fern Station" if isinstance(s, str) and "fern" in s.lower() else "Nature Park"
        )
    else:
        df["Site"] = "Unknown"

# Subsets
if "LifeStage" in df.columns:
    seedlings = df[df["LifeStage"].astype(str).str.strip().str.lower() == "seedling"].copy()
else:
    seedlings = df[df["height_cm"].notna()].copy()

red_seedlings = seedlings[seedlings["oak_group"] == "red_oak"].copy()

# Helper: save caption text
def save_caption(name, text):
    (OUT_DIR / f"{name}_caption.txt").write_text(text)

# ---------- 1) Browsing probability by site ----------
def plot_browsing_by_site():
    d = df.dropna(subset=["browsed_bin", "Site"]).copy()
    if d.shape[0] < 5:
        print("Not enough browse-labelled records to plot browsing proportion by site.")
        (OUT_DIR / "browsing_proportion_by_site.csv").write_text("Site,prop,count,sum,ci_lo,ci_hi\n")
        return

    summary = d.groupby("Site")["browsed_bin"].agg(["count", "sum"]).reset_index()
    summary["prop"] = summary["sum"] / summary["count"]

    # Wilson CI
    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        phat = k / n
        denom = 1 + z * z / n
        centre = phat + z * z / (2 * n)
        margin = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
        lo = (centre - margin) / denom
        hi = (centre + margin) / denom
        return max(0, lo), min(1, hi)

    cis = summary.apply(lambda r: wilson_ci(r["sum"], r["count"]), axis=1)
    summary["ci_lo"] = [c[0] for c in cis]
    summary["ci_hi"] = [c[1] for c in cis]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(
        x=summary["Site"],
        y=summary["prop"],
        yerr=[summary["prop"] - summary["ci_lo"], summary["ci_hi"] - summary["prop"]],
        fmt="o",
        capsize=5
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion browsed")
    ax.set_title("Browsing proportion by Site (95% CI)")
    plt.tight_layout()

    figpath = OUT_DIR / "browsing_proportion_by_site.png"
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary.to_csv(OUT_DIR / "browsing_proportion_by_site.csv", index=False)
    save_caption(
        "browsing_by_site",
        "Proportion of observations recorded as browsed (binary) at Fern Station and Nature Park. "
        "Wilson 95% confidence intervals shown."
    )

plot_browsing_by_site()

# ---------- 2) Red oak height distribution by site ----------
def plot_red_oak_height_by_site():
    rs = red_seedlings.dropna(subset=["height_cm", "Site"])
    if rs.shape[0] < 5:
        print("Not enough red oak seedling height data to plot distribution by site.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=rs, x="Site", y="height_cm", ax=axes[0])
    axes[0].set_title("Red oak seedling height by Site (boxplot)")
    axes[0].set_ylabel("Height (cm)")

    sns.kdeplot(data=rs, x="height_cm", hue="Site", common_norm=False, fill=True, ax=axes[1])
    axes[1].set_title("Red oak height density by Site")

    plt.tight_layout()
    figpath = OUT_DIR / "red_oak_height_by_site.png"
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    rs.groupby("Site")["height_cm"].describe().to_csv(OUT_DIR / "red_oak_height_stats_by_site.csv")
    save_caption(
        "red_oak_height_by_site",
        "Red oak seedling height distributions at Fern Station vs Nature Park. "
        "Boxplot and kernel density estimate shown."
    )

plot_red_oak_height_by_site()

# ---------- 3) Browsing probability vs height (logistic curves) ----------
def plot_browsing_vs_height():
    rs = red_seedlings.dropna(subset=["height_cm", "browsed_bin", "Site"])
    if rs.shape[0] < 30:
        print("Not enough red oak records for reliable logistic curves.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    heights_grid = np.linspace(rs["height_cm"].min(), rs["height_cm"].max(), 200)

    for site, group in rs.groupby("Site"):
        X = group[["height_cm"]].values.reshape(-1, 1)
        y = group["browsed_bin"].astype(int).values
        if len(np.unique(y)) < 2:
            print(f"Site {site} has only one browsing class; skipping logistic fit.")
            continue
        clf = LogisticRegression().fit(X, y)
        probs = clf.predict_proba(heights_grid.reshape(-1, 1))[:, 1]
        auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
        ax.plot(heights_grid, probs, label=f"{site} (AUC={auc:.2f})")

        # binned empirical proportions
        bins = np.unique(np.percentile(group["height_cm"], np.linspace(0, 100, 8)))
        if len(bins) > 2:
            bin_idx = np.digitize(group["height_cm"], bins)
            bin_props = (
                pd.DataFrame({"bin": bin_idx, "b": group["browsed_bin"], "h": group["height_cm"]})
                .groupby("bin")
                .agg(prop=("b", "mean"), h=("h", "median"))
                .dropna()
            )
            ax.scatter(bin_props["h"], bin_props["prop"], s=40, alpha=0.6)

    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Predicted probability browsed")
    ax.set_ylim(0, 1)
    ax.set_title("Predicted browsing probability vs height (red oak) by Site")
    ax.legend()
    plt.tight_layout()

    figpath = OUT_DIR / "red_browse_prob_vs_height_by_site.png"
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    save_caption(
        "browse_vs_height",
        "Predicted probability that a red oak seedling is recorded as browsed, plotted against seedling height. "
        "Logistic regression fit separately by site; binned empirical proportions overplotted."
    )

plot_browsing_vs_height()

# ---------- 4) Species composition comparison ----------
def plot_species_composition():
    if "Species" not in df.columns or "Site" not in df.columns:
        print("Missing Species or Site columns; skipping species composition plot.")
        return

    topN = 12
    counts = df.groupby(["Species", "Site"]).size().reset_index(name="count")
    total_counts = df["Species"].value_counts().head(topN).index.tolist()
    counts = counts[counts["Species"].isin(total_counts)]
    pivot = counts.pivot(index="Species", columns="Site", values="count").fillna(0)
    pivot = pivot.loc[total_counts]

    fig = pivot.plot(kind="bar", figsize=(10, 5)).get_figure()
    plt.title(f"Top {topN} species counts by Site")
    plt.tight_layout()

    figpath = OUT_DIR / f"species_top{topN}_by_site.png"
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pivot.to_csv(OUT_DIR / f"species_top{topN}_by_site.csv")
    save_caption(
        "species_composition",
        f"Counts of the top {topN} species split by Site."
    )

plot_species_composition()

# ---------- 5) Oak recruitment funnel ----------
def plot_recruitment_funnel():
    if "LifeStage" not in df.columns:
        def derive_stage(r):
            h = r.get("height_cm", np.nan)
            d = r.get("diam_cm", np.nan)
            if pd.notna(d) and d >= 18:
                return "Adult"
            if pd.notna(h) and h >= 200:
                return "Sapling"
            if pd.notna(h) and h < 200:
                return "Seedling"
            return "Unknown"
        df["LifeStage"] = df.apply(derive_stage, axis=1)

    for site in df["Site"].dropna().unique():
        subset = df[df["Site"] == site]
        oak_groups = ["red_oak", "white_oak", "other_oak", "non_oak"]

        fig, ax = plt.subplots(figsize=(7, 4))
        for g in oak_groups:
            s = subset[subset["oak_group"] == g]["LifeStage"].astype(str).str.strip().str.capitalize().value_counts()
            vals = [s.get("Seedling", 0), s.get("Sapling", 0), s.get("Adult", 0), s.get("Unknown", 0)]
            ax.plot(["Seedling", "Sapling", "Adult", "Unknown"], vals, marker="o", label=g)

        ax.set_ylabel("Count")
        ax.set_title(f"Recruitment funnel by oak group — {site}")
        ax.legend()
        plt.tight_layout()

        figpath = OUT_DIR / f"recruitment_funnel_{site.replace(' ', '_')}.png"
        fig.savefig(figpath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    trans = df.groupby(["Site", "oak_group", "LifeStage"]).size().unstack(fill_value=0)
    trans.to_csv(OUT_DIR / "recruitment_table_site_oak.csv")
    save_caption(
        "recruitment_funnel",
        "Counts of individuals in each life stage (Seedling, Sapling, Adult) for each oak group and site."
    )

plot_recruitment_funnel()

# ---------- 6) Sapling/juvenile height by browsed status ----------
def plot_sapling_height_vs_browsed():
    if "LifeStage" in df.columns:
        bigger = df[df["LifeStage"].astype(str).isin(["Sapling", "Adult", "Juvenile"])].dropna(subset=["height_cm", "browsed_bin"])
    else:
        bigger = df[df["height_cm"] > 50].dropna(subset=["height_cm", "browsed_bin"])

    if bigger.shape[0] < 6:
        bigger = df[df["height_cm"] > 50].dropna(subset=["height_cm", "browsed_bin"])

    if bigger.shape[0] < 6:
        print("Not enough larger plants with browse labels for sapling height plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bigger["browsed_label"] = bigger["browsed_bin"].map({0.0: "Not browsed", 1.0: "Browsed"})
    sns.boxplot(data=bigger, x="browsed_label", y="height_cm", ax=ax)

    ax.set_title("Height of larger individuals by browsed status")
    ax.set_ylabel("Height (cm)")
    plt.tight_layout()

    figpath = OUT_DIR / "sapling_height_by_browsed_box.png"
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    bigger.groupby("browsed_label")["height_cm"].describe().to_csv(OUT_DIR / "sapling_height_by_browsed_stats.csv")
    save_caption(
        "sapling_height_browsed",
        "Comparison of heights for larger individuals split by whether they were recorded as browsed."
    )

plot_sapling_height_vs_browsed()

# ---------- SIMPLE SUMMARY LOG ----------
summary_lines = []
summary_lines.append(f"Total records: {len(df)}")

if "LifeStage" in df.columns:
    summary_lines.append(
        f"Red oak seedlings: {len(df[(df['oak_group']=='red_oak') & (df['LifeStage'].astype(str).str.lower()=='seedling')])}"
    )
else:
    summary_lines.append("Red oak seedlings: (LifeStage missing; not computed)")

summary_lines.append(f"Records with browse label: {df['browsed_bin'].notna().sum()}")
(OUT_DIR / "visualization_summary.txt").write_text("\n".join(summary_lines))

print("Finished all visualizations. Outputs in:", OUT_DIR)