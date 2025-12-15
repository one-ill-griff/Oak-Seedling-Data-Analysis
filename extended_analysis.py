# extended_analysis.py
# Additional seedling analysis:
# - species composition
# - height distributions
# - browsing vs height class
# - optional spatial prep

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------
FILE = "data/processed/wide_seedling_data_cleaned.csv"
df = pd.read_csv(FILE)

# Ensure directories exist
OUT_DIR = "data/processed/analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loaded:", df.shape)


# ---------------------------------------------------
# Re-construct oak group classification
# (same mapping used previously)
# ---------------------------------------------------
oak_map = {
    "QURU": "red_oak",
    "R": "red_oak",
    "QUVE": "red_oak",
    "QUAL": "white_oak",
    "QUMU": "white_oak",
    "W": "white_oak",
}

def classify_oak(sp):
    # Handle missing species values
    if pd.isna(sp):
        return "unknown"

    sp = str(sp).strip().upper()

    # Direct map
    if sp in oak_map:
        return oak_map[sp]

    # Any other oak abbreviations starting with QU
    if sp.startswith("QU"):
        # Red oak group indicators
        if any(x in sp for x in ["RU", "VE", "PA", "NI", "FA"]):
            return "red_oak"
        else:
            return "white_oak"

    # Everything else is non-oak
    return "non_oak"

df["oak_group"] = df["Species"].apply(classify_oak)


# ---------------------------------------------------
# Life-stage determination
# ---------------------------------------------------
def get_lifestage(row):
    h = row.get("height_cm", None)
    d = row.get("diam_cm", None)

    # ensure numeric
    try:
        h = float(h) if h not in [None, ""] else None
    except:
        h = None

    try:
        d = float(d) if d not in [None, ""] else None
    except:
        d = None

    # classification rules
    if h is None and d is None:
        return "Unknown"

    # Seedlings: <30 cm height and no measurable diameter
    if h is not None and h < 30:
        return "Seedling"

    # Saplings: 30–200 cm height OR tiny diameter (<2 cm)
    if (h is not None and 30 <= h <= 200) or (d is not None and d < 2):
        return "Sapling"

    # Poles / Juvenile trees: height >200 cm OR DBH 2–8 cm
    if (h is not None and h > 200) or (d is not None and 2 <= d <= 8):
        return "Juvenile"

    # Mature trees: DBH >8 cm
    if d is not None and d > 8:
        return "Mature"

    return "Unknown"

df["LifeStage"] = df.apply(get_lifestage, axis=1)


# ---------------------------------------------------
# Species composition (counts + proportions)
# ---------------------------------------------------
species_counts = df["Species"].value_counts().reset_index()
species_counts.columns = ["Species", "Count"]
species_counts["Proportion"] = species_counts["Count"] / len(df)

species_counts.to_csv(f"{OUT_DIR}/species_composition.csv", index=False)
print("Saved species composition.")

# Bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=species_counts.head(15), x="Species", y="Count")
plt.xticks(rotation=45)
plt.title("Top 15 Species by Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/species_composition_top15.png")
plt.close()


# ---------------------------------------------------
# Height distribution analysis by oak group
# ---------------------------------------------------
height_df = df[df["Height(cm)"].notna()].copy()

plt.figure(figsize=(10, 6))
sns.boxplot(data=height_df, x="oak_group", y="Height(cm)")
plt.title("Height Distribution by Oak Group")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/height_distribution_oak_group.png")
plt.close()

# Histograms for each group
for group in height_df["oak_group"].unique():
    sub = height_df[height_df["oak_group"] == group]
    plt.figure(figsize=(8, 5))
    plt.hist(sub["Height(cm)"], bins=25)
    plt.title(f"Height Distribution – {group}")
    plt.xlabel("Height (cm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/height_hist_{group}.png")
    plt.close()

print("Saved height distribution plots.")


# ---------------------------------------------------
# Browsing vs height class (does deer select height ranges?)
# ---------------------------------------------------
def height_class(h):
    # clean/convert to numeric
    try:
        h = float(h)
    except:
        return np.nan

    if h < 10:
        return "<10"
    elif h < 30:
        return "10–30"
    elif h < 60:
        return "30–60"
    elif h < 100:
        return "60–100"
    elif h < 200:
        return "100–200"
    else:
        return "200+"


df["height_class"] = df["Height(cm)"].apply(lambda x: height_class(x) if pd.notna(x) else np.nan)

browse_height = df[df["Browsed"] != "Unknown"].copy()
browse_height["Browsed_bin"] = browse_height["Browsed"].str.lower().isin(["yes", "y", "browsed"])

height_browse_summary = browse_height.groupby(["oak_group", "height_class"])["Browsed_bin"] \
                                     .mean().reset_index()
height_browse_summary.to_csv(f"{OUT_DIR}/browse_by_height_class.csv", index=False)

# Heatmap
pivot = height_browse_summary.pivot(index="height_class", columns="oak_group", values="Browsed_bin")

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
plt.title("Browsing Probability by Height Class and Oak Group")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/browse_height_heatmap.png")
plt.close()

print("Saved browsing vs height class analysis.")


# ---------------------------------------------------
# Optional: Spatial prep (if coordinates become usable)
# ---------------------------------------------------
# For now, extract numeric coords if present.
# Format assumed: "(lat, lon)" or "lat,lon"
def parse_coords(c):
    if pd.isna(c) or c == "Unknown":
        return pd.Series([np.nan, np.nan])
    try:
        c = c.replace("(", "").replace(")", "")
        lat, lon = c.split(",")
        return pd.Series([float(lat), float(lon)])
    except:
        return pd.Series([np.nan, np.nan])

df[["Lat", "Lon"]] = df["Coordinates"].apply(parse_coords)

df.to_csv(f"{OUT_DIR}/spatial_prepped.csv", index=False)
print("Saved spatial-prepped dataset.")

print("\nExtended analysis complete.")
print("Check outputs in:", OUT_DIR)