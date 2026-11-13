# spatio_temporal_hotspot_analysis.py
# MSc Dissertation - Crime Hotspot Analysis
# Zainuddin Lambate
# University of Greenwich | Supervisor: Jia Wang
#
# Run:  python spatio_temporal_hotspot_analysis.py
# Placed MPS crime CSVs in data/raw/ and LSOA shapefile in data/boundaries/

import logging
import subprocess
import sys
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.getLogger("statsmodels").setLevel(logging.ERROR)

from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
from sklearn.cluster import DBSCAN
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



RAW_DIR      = Path("data/raw")
BOUNDARY_DIR = Path("data/boundaries")
PROC_DIR     = Path("data/processed")
MAPS_DIR     = Path("outputs/maps")

PROC_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)



START_DATE = "2023-03-01"
END_DATE   = "2026-02-01"


TRAIN_END  = pd.Timestamp("2025-02-01")
TEST_START = pd.Timestamp("2025-03-01")

SPARSE_THRESHOLD = 3
FDR_ALPHA        = 0.05

DBSCAN_EPS         = 1500
DBSCAN_MIN_SAMPLES = 5

PERSISTENT_MONTHS   = 24
EMERGING_LATE_MIN   = 8
EMERGING_EARLY_MAX  = 3
DECLINING_EARLY_MIN = 8

# LISA permutations - 999 for final run, 499 for quick test
LISA_PERMUTATIONS = 999

MA_WINDOW = 3
RF_LAGS   = 12
RF_TREES  = 200
RF_SEED   = 42

# Set to e.g. 200 for a quick test run, None for full analysis
SAMPLE_LSOAS = None

# -- checkpoint flags - set True to skip already-completed stages 
SKIP_IF_EXISTS = True   # skip a stage if its output CSV already exists


def stage_done(path):
    """Return True if checkpoint skipping is on and the file exists."""
    return SKIP_IF_EXISTS and Path(path).exists()


print("=" * 60)
print("Crime Hotspot Pipeline")
print("=" * 60)


# -- Stage 1: Load raw crime CSVs 
print("\nStage 1: Loading raw crime data")
csv_files = list(RAW_DIR.rglob("*.csv"))
if not csv_files:
    raise FileNotFoundError(
        f"No CSVs found in {RAW_DIR}/ - download MPS data from "
        "data.police.uk and place files there."
    )
print(f"  {len(csv_files)} files found")

frames = []
for f in tqdm(csv_files, desc="  Reading"):
    frames.append(pd.read_csv(f, low_memory=False))
crime_raw = pd.concat(frames, ignore_index=True)

crime_raw.columns = (
    crime_raw.columns.str.strip().str.lower().str.replace(" ", "_")
)
print(f"  Loaded {len(crime_raw):,} rows")
print(f"  Columns: {list(crime_raw.columns)}")


# -- Stage 2: Clean and filter
print("\nStage 2: Cleaning and filtering")

crime = crime_raw.dropna(subset=["lsoa_code", "month", "crime_type"]).copy()
crime = crime[crime["lsoa_code"].str.startswith("E01", na=False)]
crime["year_month"] = pd.to_datetime(crime["month"], format="%Y-%m")
crime = crime[
    (crime["year_month"] >= START_DATE) &
    (crime["year_month"] <= END_DATE)
]

# Use falls_within column for borough - avoids broken string-split
# on multi-word LSOA names (e.g. "Tower Hamlets 001A")
if "falls_within" in crime.columns:
    crime["borough"] = crime["falls_within"].str.strip()
else:
    # fallback: strip the trailing code from lsoa_name
    crime["borough"] = (crime["lsoa_name"]
                        .str.replace(r"\s+\d{3}[A-Z]$", "", regex=True)
                        .str.strip())

n_before = len(crime)
crime = crime[
    ~crime["crime_type"].str.lower().str.contains(
        "sexual offence|sexual offense", na=False
    )
]
print(f"  Removed {n_before - len(crime):,} sexual offence rows")

crime["year"]      = crime["year_month"].dt.year
crime["month_num"] = crime["year_month"].dt.month

print(f"  Rows kept:     {len(crime):,}")
print(f"  Unique LSOAs:  {crime['lsoa_code'].nunique():,}")
print(f"  Months:        {crime['year_month'].nunique()} (expecting 36)")
print(f"  Range:         "
      f"{crime['year_month'].min().strftime('%Y-%m')} "
      f"to {crime['year_month'].max().strftime('%Y-%m')}")
print("\n  Crime types:")
print(crime["crime_type"].value_counts().to_string())


# -- Stage 3: Build the LSOA × month panel 
print("\nStage 3: Building the LSOA × month panel")

if stage_done(PROC_DIR / "lsoa_monthly_crime_panel.csv"):
    print("  Checkpoint found - loading existing panel")
    panel = pd.read_csv(PROC_DIR / "lsoa_monthly_crime_panel.csv",
                        parse_dates=["year_month"])
    borough_monthly = pd.read_csv(PROC_DIR / "borough_monthly_totals.csv",
                                  parse_dates=["year_month"])
else:
    monthly_counts = (
        crime
        .groupby(["lsoa_code", "lsoa_name", "year_month"])
        .size()
        .reset_index(name="crime_count")
    )

    all_lsoas  = crime["lsoa_code"].unique()
    all_months = pd.date_range(START_DATE, END_DATE, freq="MS")

    idx = pd.MultiIndex.from_product(
        [all_lsoas, all_months], names=["lsoa_code", "year_month"]
    )
    panel = (
        pd.DataFrame(index=idx)
        .reset_index()
        .merge(monthly_counts, on=["lsoa_code", "year_month"], how="left")
    )
    panel["crime_count"] = panel["crime_count"].fillna(0).astype(int)
    panel["year"]        = panel["year_month"].dt.year
    panel["month_num"]   = panel["year_month"].dt.month

    lsoa_names = crime[["lsoa_code", "lsoa_name"]].drop_duplicates()
    panel = (
        panel
        .drop(columns="lsoa_name", errors="ignore")
        .merge(lsoa_names, on="lsoa_code", how="left")
        .sort_values(["lsoa_code", "year_month"])
        .reset_index(drop=True)
    )

    # Borough - use falls_within-derived column from raw crime data
    borough_map = (crime[["lsoa_code", "borough"]]
                   .drop_duplicates("lsoa_code")
                   .set_index("lsoa_code")["borough"])
    panel["borough"] = panel["lsoa_code"].map(borough_map)

    # Also build a crime type breakdown panel for dashboard use
    crime_type_panel = (
        crime
        .groupby(["lsoa_code", "year_month", "crime_type"])
        .size()
        .reset_index(name="count")
    )
    crime_type_panel.to_csv(PROC_DIR / "lsoa_crime_type_panel.csv", index=False)
    print("  Saved: lsoa_crime_type_panel.csv")

    borough_monthly = (
        panel.groupby(["borough", "year_month"])["crime_count"]
        .sum().reset_index()
    )

    panel.to_csv(PROC_DIR / "lsoa_monthly_crime_panel.csv", index=False)
    borough_monthly.to_csv(PROC_DIR / "borough_monthly_totals.csv", index=False)
    print("  Saved: lsoa_monthly_crime_panel.csv")
    print("  Saved: borough_monthly_totals.csv")

all_lsoas  = panel["lsoa_code"].unique()
all_months = pd.date_range(START_DATE, END_DATE, freq="MS")

print(f"  Panel: {panel.shape[0]:,} rows  (expecting ~{len(all_lsoas) * 36:,})")
print("\n  Top 5 boroughs by mean monthly count:")
print(
    borough_monthly.groupby("borough")["crime_count"]
    .mean().nlargest(5).round(1).to_string()
)


# -- Stage 4: Sparsity check 
print(f"\nStage 4: Sparsity check  (threshold: median < {SPARSE_THRESHOLD})")

lsoa_medians = (
    panel.groupby("lsoa_code")["crime_count"]
    .median().reset_index()
    .rename(columns={"crime_count": "median_monthly"})
)
sparse_lsoas = lsoa_medians.loc[
    lsoa_medians["median_monthly"] < SPARSE_THRESHOLD, "lsoa_code"
].tolist()
dense_lsoas = lsoa_medians.loc[
    lsoa_medians["median_monthly"] >= SPARSE_THRESHOLD, "lsoa_code"
].tolist()

print(f"  Dense:  {len(dense_lsoas):,} LSOAs  (used for SARIMA and RF)")
print(f"  Sparse: {len(sparse_lsoas):,} LSOAs  (rolled up to MSOA level)")

panel_dense = panel[panel["lsoa_code"].isin(dense_lsoas)].copy()

lookup_path = BOUNDARY_DIR / "lsoa_msoa_lookup.csv"
if lookup_path.exists():
    lookup = pd.read_csv(lookup_path)
    lookup.columns = lookup.columns.str.lower()
    msoa_col = next(c for c in lookup.columns if "msoa" in c and "cd" in c)
    lsoa_col = next(c for c in lookup.columns if "lsoa" in c and "cd" in c)
    lookup   = lookup[[lsoa_col, msoa_col]].rename(
        columns={lsoa_col: "lsoa_code", msoa_col: "msoa_code"}
    )
    panel_sparse_msoa = (
        panel[panel["lsoa_code"].isin(sparse_lsoas)]
        .merge(lookup, on="lsoa_code", how="left")
        .groupby(["msoa_code", "year_month", "year", "month_num"])
        ["crime_count"].sum().reset_index()
    )
    panel_sparse_msoa.to_csv(
        PROC_DIR / "msoa_monthly_sparse_panel.csv", index=False
    )
    print(f"  Saved MSOA panel: {len(panel_sparse_msoa):,} rows")
else:
    print(f"  Warning: lookup not found at {lookup_path}")
    print("  Download from geoportal.statistics.gov.uk and save as "
          "data/boundaries/lsoa_msoa_lookup.csv")
    print("  Continuing with dense LSOAs only.")

lsoa_medians["sparse"] = lsoa_medians["lsoa_code"].isin(sparse_lsoas)
lsoa_medians.to_csv(PROC_DIR / "lsoa_sparsity_report.csv", index=False)
print("  Saved: lsoa_sparsity_report.csv")


# -- Stage 5: Load LSOA boundaries 
print("\nStage 5: Loading LSOA boundaries")

shp_files = [f for f in BOUNDARY_DIR.rglob("*.shp")
             if "lsoa" in f.name.lower()]
if not shp_files:
    raise FileNotFoundError(
        "No LSOA shapefile in data/boundaries/ - download from "
        "geoportal.statistics.gov.uk (LSOA December 2021 Boundaries EW BGC)."
    )

lsoa_gdf = gpd.read_file(shp_files[0])
lsoa_gdf.columns = lsoa_gdf.columns.str.lower()
print(f"  Using: {shp_files[0].name}")

code_col = next(c for c in lsoa_gdf.columns if "lsoa" in c and "cd" in c)
lsoa_gdf = lsoa_gdf.rename(columns={code_col: "lsoa_code"})

lsoa_london = (
    lsoa_gdf[lsoa_gdf["lsoa_code"].isin(all_lsoas)]
    .copy()
    .to_crs(epsg=27700)
)
print(f"  Matched: {len(lsoa_london):,} London LSOAs")

# Pre-compute centroids for DBSCAN - done once, reused every month
lsoa_london["cx"] = lsoa_london.geometry.centroid.x
lsoa_london["cy"] = lsoa_london.geometry.centroid.y


# -- Stage 6: Monthly Local Moran's I with BH-FDR 
print("\nStage 6: Monthly Local Moran's I with BH-FDR correction")
print(f"  {len(lsoa_london):,} tests per month  -  FDR alpha = {FDR_ALPHA}")
print(f"  Without FDR: ~{int(len(all_lsoas) * 0.05)} false positives expected")
print(f"  Permutations per month: {LISA_PERMUTATIONS}")

if stage_done(PROC_DIR / "lsoa_monthly_lisa.csv"):
    print("  Checkpoint found - loading existing LISA results")
    lisa_df = pd.read_csv(PROC_DIR / "lsoa_monthly_lisa.csv",
                          parse_dates=["year_month"])
else:
    print("  Building Queen contiguity weights matrix...")
    w = Queen.from_dataframe(lsoa_london, silence_warnings=True)
    w.transform = "r"
    avg_n = np.mean([len(v) for v in w.neighbors.values()])
    print(f"  {w.n} units, {avg_n:.1f} neighbours on average")

    lsoa_order = lsoa_london["lsoa_code"].tolist()

    # Pre-build index dict for O(1) LSOA to position lookups
    lsoa_idx = {l: i for i, l in enumerate(lsoa_order)}

    quad_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    results = []

    # Global Moran's I summary across months
    global_moran_rows = []

    for ym in tqdm(all_months, desc="  Running LISA"):
        counts = (
            panel[panel["year_month"] == ym]
            .groupby("lsoa_code")["crime_count"].sum()
            .reindex(lsoa_order).fillna(0).values
        )

        # Global Moran's I - single statistic for this month
        gm = Moran(counts, w, permutations=LISA_PERMUTATIONS)
        global_moran_rows.append({
            "year_month": ym,
            "global_i":  round(gm.I, 6),
            "p_sim":     round(gm.p_sim, 6),
        })

        # Local Moran's I
        lisa = Moran_Local(counts, w,
                           permutations=LISA_PERMUTATIONS, seed=42)
        p = lisa.p_sim
        n = len(p)

        # Benjamini-Hochberg step
        sorted_idx = np.argsort(p)
        bh_crits   = ((np.arange(1, n + 1)) / n) * FDR_ALPHA
        passing    = p[sorted_idx] <= bh_crits
        thresh = (np.max(np.where(passing)[0]) + 1) / n * FDR_ALPHA \
                 if passing.any() else 0.0
        sig = p <= thresh

        for i, lsoa in enumerate(lsoa_order):
            results.append({
                "lsoa_code":    lsoa,
                "year_month":   ym,
                "local_i":      lisa.Is[i],
                "p_sim":        p[i],
                "bh_threshold": thresh,
                "significant":  bool(sig[i]),
                "quadrant":     lisa.q[i],
                "lisa_type":    quad_labels.get(lisa.q[i], "NS")
                                if sig[i] else "NS",
            })

    lisa_df = pd.DataFrame(results)
    lisa_df.to_csv(PROC_DIR / "lsoa_monthly_lisa.csv", index=False)

    global_moran_df = pd.DataFrame(global_moran_rows)
    global_moran_df.to_csv(PROC_DIR / "global_moran_monthly.csv", index=False)
    print("  Saved: lsoa_monthly_lisa.csv")
    print("  Saved: global_moran_monthly.csv")

hh_monthly = (
    lisa_df[lisa_df["lisa_type"] == "HH"]
    .groupby("year_month").size().reset_index(name="hh_count")
)
print(f"  HH clusters/month - "
      f"mean: {hh_monthly['hh_count'].mean():.0f}, "
      f"range: {hh_monthly['hh_count'].min()}-"
      f"{hh_monthly['hh_count'].max()}")


# -- Stage 7: DBSCAN clustering 
print("\nStage 7: DBSCAN clustering")
print(f"  eps = {DBSCAN_EPS}m, min_samples = {DBSCAN_MIN_SAMPLES}")

if stage_done(PROC_DIR / "lsoa_monthly_dbscan.csv"):
    print("  Checkpoint found - loading existing DBSCAN results")
    dbscan_df = pd.read_csv(PROC_DIR / "lsoa_monthly_dbscan.csv",
                            parse_dates=["year_month"])
else:
    coords = lsoa_london[["cx", "cy"]].values

    # Pre-allocate features array - update third column in-place each month
    features = np.empty((len(lsoa_london), 3))
    features[:, :2] = coords

    dbscan_results = []
    lsoa_list = lsoa_london["lsoa_code"].tolist()

    for ym in tqdm(all_months, desc="  Running DBSCAN"):
        counts = (
            panel[panel["year_month"] == ym]
            .groupby("lsoa_code")["crime_count"].sum()
            .reindex(lsoa_london["lsoa_code"]).fillna(0).values
        )
        z = (counts - counts.mean()) / (counts.std() + 1e-9)
        features[:, 2] = z * DBSCAN_EPS * 0.5

        labels = DBSCAN(
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES
        ).fit(features).labels_

        for i, lsoa in enumerate(lsoa_list):
            dbscan_results.append({
                "lsoa_code":      lsoa,
                "year_month":     ym,
                "dbscan_cluster": int(labels[i]),
                "in_cluster":     bool(labels[i] != -1),
            })

    dbscan_df = pd.DataFrame(dbscan_results)
    dbscan_df.to_csv(PROC_DIR / "lsoa_monthly_dbscan.csv", index=False)
    print("  Saved: lsoa_monthly_dbscan.csv")


# -- Stage 8: Hotspot persistence classification 
print("\nStage 8: Classifying hotspot persistence across 36 months")

if stage_done(PROC_DIR / "lsoa_hotspot_classification.csv"):
    print("  Checkpoint found - loading existing classification")
    persistence = pd.read_csv(PROC_DIR / "lsoa_hotspot_classification.csv")
else:
    hh_pivot = (
        lisa_df
        .assign(is_hh=lisa_df["lisa_type"] == "HH")
        .pivot(index="lsoa_code", columns="year_month", values="is_hh")
        .fillna(False)
    )

    early = [m for m in all_months if m.year == 2023]
    late  = [m for m in all_months if m.year == 2025]

    def classify_hotspot(row):
        total   = row.sum()
        early_n = row[early].sum()
        late_n  = row[late].sum()
        if total >= PERSISTENT_MONTHS:
            return "Persistent hotspot"
        elif late_n >= EMERGING_LATE_MIN and early_n <= EMERGING_EARLY_MAX:
            return "Emerging hotspot"
        elif early_n >= DECLINING_EARLY_MIN and late_n <= EMERGING_EARLY_MAX:
            return "Declining hotspot"
        elif total >= 6:
            return "Intermittent hotspot"
        else:
            return "Non-hotspot"

    persistence = hh_pivot.apply(classify_hotspot, axis=1).reset_index()
    persistence.columns = ["lsoa_code", "hotspot_class"]
    persistence["hh_months"] = hh_pivot.sum(axis=1).values
    persistence = persistence.merge(
        lsoa_medians[["lsoa_code", "median_monthly", "sparse"]],
        on="lsoa_code", how="left"
    )
    persistence.to_csv(PROC_DIR / "lsoa_hotspot_classification.csv", index=False)
    print("  Saved: lsoa_hotspot_classification.csv")

print("\n  Results:")
print(persistence["hotspot_class"].value_counts().to_string())
n_warn = ((persistence["hotspot_class"] == "Persistent hotspot") &
          persistence["sparse"]).sum()
if n_warn:
    print(f"\n  Note: {n_warn} persistent hotspots are in sparse LSOAs")


# -- Stage 9: Maps and charts 
print("\nStage 9: Producing maps and charts")

LISA_COLOURS = {
    "HH": "#e63946",
    "LL": "#457b9d",
    "HL": "#f4a261",
    "LH": "#a8dadc",
    "NS": "#2d2d3a",
}
PERSIST_COLOURS = {
    "Persistent hotspot":   "#e63946",
    "Emerging hotspot":     "#f4a261",
    "Declining hotspot":    "#457b9d",
    "Intermittent hotspot": "#ffd166",
    "Non-hotspot":          "#2d2d3a",
}
INNER_LONDON = {
    "minx": 515000, "maxx": 545000,
    "miny": 175000, "maxy": 195000
}


def add_legend(ax, colours, title, bg="#1a1a2e", present=None):
    # Only show legend entries for classes that actually appear in the data
    filtered = {l: c for l, c in colours.items()
                if present is None or l in present}
    patches = [Patch(facecolor=c, label=l, edgecolor="#555")
               for l, c in filtered.items()]
    leg = ax.legend(handles=patches, title=title, loc="lower left",
                    fontsize=7, title_fontsize=8,
                    framealpha=0.9, facecolor=bg, labelcolor="white")
    leg.get_title().set_color("white")


def plot_london(ax, gdf, col_col, bg_col, lw=0, edge="#555"):
    bg = gdf[gdf[col_col] == bg_col]
    fg = gdf[gdf[col_col] != bg_col]
    if not bg.empty:
        bg.plot(color=bg_col, linewidth=0, ax=ax)
    if not fg.empty:
        fg.plot(color=fg[col_col], linewidth=lw,
                edgecolor=edge if lw > 0 else "none", ax=ax)


def add_inset_rect(ax):
    ax.add_patch(Rectangle(
        (INNER_LONDON["minx"], INNER_LONDON["miny"]),
        INNER_LONDON["maxx"] - INNER_LONDON["minx"],
        INNER_LONDON["maxy"] - INNER_LONDON["miny"],
        linewidth=0.8, edgecolor="white", facecolor="none",
        linestyle="--", zorder=5
    ))


# Map 1 - LISA snapshots
snap_months = [
    pd.Timestamp("2023-03-01"),
    pd.Timestamp("2024-03-01"),
    pd.Timestamp("2025-03-01"),
]
fig = plt.figure(figsize=(24, 9), facecolor="#1a1a2e")
fig.suptitle(
    "Local Moran's I Crime Hotspots - Greater London\n"
    "BH-FDR corrected (\u03b1=0.05)",
    fontsize=13, fontweight="bold", color="white", y=1.0
)
axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
for ax, ym in zip(axes, snap_months):
    snap = lisa_df[lisa_df["year_month"] == ym][["lsoa_code", "lisa_type"]]
    gdf  = lsoa_london.merge(snap, on="lsoa_code", how="left")
    gdf["col"] = gdf["lisa_type"].map(LISA_COLOURS).fillna(LISA_COLOURS["NS"])
    ax.set_facecolor("#1a1a2e")
    plot_london(ax, gdf, "col", LISA_COLOURS["NS"])
    ax.set_title(ym.strftime("%B %Y"), fontsize=12,
                 fontweight="bold", color="white", pad=6)
    ax.axis("off")
    axins = ax.inset_axes([0.55, 0.0, 0.45, 0.45])
    axins.set_facecolor("#1a1a2e")
    plot_london(axins, gdf, "col", LISA_COLOURS["NS"], lw=0.3, edge="#444")
    axins.set_xlim(INNER_LONDON["minx"], INNER_LONDON["maxx"])
    axins.set_ylim(INNER_LONDON["miny"], INNER_LONDON["maxy"])
    axins.axis("off")
    axins.set_title("Inner London", fontsize=7, color="white", pad=2)
    add_inset_rect(ax)
    add_legend(ax, LISA_COLOURS, "Cluster type")
plt.tight_layout()
plt.savefig(MAPS_DIR / "01_lisa_snapshots.png", dpi=200,
            bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  Saved: 01_lisa_snapshots.png")

# Map 2 - Persistence map
gdf_p = lsoa_london.merge(persistence, on="lsoa_code", how="left")
gdf_p["col"] = (gdf_p["hotspot_class"]
                .map(PERSIST_COLOURS)
                .fillna(PERSIST_COLOURS["Non-hotspot"]))
fig, axes_p = plt.subplots(1, 2, figsize=(18, 9),
                            gridspec_kw={"width_ratios": [2, 1]},
                            facecolor="#1a1a2e")
ax_full = axes_p[0]
ax_full.set_facecolor("#1a1a2e")
plot_london(ax_full, gdf_p, "col", PERSIST_COLOURS["Non-hotspot"])
ax_full.set_title("Greater London", fontsize=11, color="white", pad=6)
ax_full.axis("off")
add_inset_rect(ax_full)
add_legend(ax_full, PERSIST_COLOURS, "Hotspot class",
           present=set(gdf_p["hotspot_class"].dropna().unique()))
ax_zoom = axes_p[1]
ax_zoom.set_facecolor("#1a1a2e")
plot_london(ax_zoom, gdf_p, "col", PERSIST_COLOURS["Non-hotspot"],
            lw=0.5, edge="#555")
ax_zoom.set_xlim(INNER_LONDON["minx"], INNER_LONDON["maxx"])
ax_zoom.set_ylim(INNER_LONDON["miny"], INNER_LONDON["maxy"])
ax_zoom.set_title("Inner London (zoomed)", fontsize=11, color="white", pad=6)
ax_zoom.axis("off")
fig.suptitle(
    "Hotspot Persistence 2023\u20132026 \u2014 Greater London\n"
    "Month-by-month Local Moran\u2019s I across 36 months",
    fontsize=13, fontweight="bold", color="white", y=1.01
)
fig.text(0.5, 0.01,
         f"Persistent: HH \u226524 of 36 months  |  "
         f"Persistent: {(gdf_p['hotspot_class']=='Persistent hotspot').sum()}  |  "
         f"Intermittent: {(gdf_p['hotspot_class']=='Intermittent hotspot').sum()}  |  "
         f"Non-hotspot: {(gdf_p['hotspot_class']=='Non-hotspot').sum()}",
         ha="center", fontsize=8, color="#aaaaaa")
plt.tight_layout()
plt.savefig(MAPS_DIR / "02_hotspot_persistence.png", dpi=200,
            bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  Saved: 02_hotspot_persistence.png")

# Chart 3 - Monthly crime trend
monthly_total = panel.groupby("year_month")["crime_count"].sum().reset_index()
fig, ax = plt.subplots(figsize=(13, 4))
ax.fill_between(monthly_total["year_month"], monthly_total["crime_count"],
                alpha=0.08, color="#2E75B6")
ax.plot(monthly_total["year_month"], monthly_total["crime_count"],
        color="#2E75B6", linewidth=2, marker="o", markersize=4)
ax.axvline(TRAIN_END, color="#888888", linestyle="--",
           linewidth=1.2, label="Train / test split (Mar 2025)")
ax.set_title("Total Monthly Recorded Crime \u2014 Greater London 2023\u20132026",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Total crimes")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_facecolor("#fafafa")
plt.tight_layout()
plt.savefig(MAPS_DIR / "03_monthly_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_monthly_trend.png")

# Chart 4 - HH count over time
fig, ax = plt.subplots(figsize=(13, 4))
ax.fill_between(hh_monthly["year_month"], hh_monthly["hh_count"],
                alpha=0.25, color="#e63946")
ax.plot(hh_monthly["year_month"], hh_monthly["hh_count"],
        color="#e63946", linewidth=2.5, marker="o", markersize=3)
ax.axvline(TRAIN_END, color="#888888", linestyle="--",
           linewidth=1.2, label="Train / test split")
ax.set_title("Number of HH Hotspot LSOAs per Month (after BH-FDR correction)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("HH LSOA count")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_facecolor("#fafafa")
plt.tight_layout()
plt.savefig(MAPS_DIR / "04_hh_count_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_hh_count_over_time.png")


# -- Stage 10: Forecasting 
print("\nStage 10: Forecasting")
print(f"  Train: Mar 2023 \u2013 Feb 2025   |   Test: Mar 2025 \u2013 Feb 2026")
print(f"  {len(dense_lsoas):,} dense LSOAs in scope")

if stage_done(PROC_DIR / "lsoa_forecast_predictions.csv"):
    print("  Checkpoint found - loading existing predictions")
    preds_df = pd.read_csv(PROC_DIR / "lsoa_forecast_predictions.csv",
                           parse_dates=["year_month"])

    # Generate feature importances if missing (common when checkpoint was used)
    fi_path = PROC_DIR / "rf_feature_importances.csv"
    if not fi_path.exists():
        print("  rf_feature_importances.csv missing - regenerating from predictions...")
        panel_dense = panel_dense.merge(
            persistence[["lsoa_code", "hotspot_class"]],
            on="lsoa_code", how="left"
        )
        panel_dense["hotspot_class"] = panel_dense["hotspot_class"].fillna("Non-hotspot")
        panel_lags = panel_dense.copy()
        panel_lags["month_num"] = panel_lags["year_month"].dt.month
        panel_lags["month_sin"] = np.sin(2 * np.pi * panel_lags["month_num"] / 12)
        panel_lags["month_cos"] = np.cos(2 * np.pi * panel_lags["month_num"] / 12)
        for lag in range(1, RF_LAGS + 1):
            panel_lags[f"lag_{lag}"] = (panel_lags.groupby("lsoa_code")["crime_count"]
                                        .shift(lag))
        # Add spatial lag (simplified: use 0 if weights not available)
        if "spatial_lag" not in panel_lags.columns:
            panel_lags["spatial_lag"] = 0.0
        lag_cols = [f"lag_{i}" for i in range(1, RF_LAGS + 1)] + ["month_sin", "month_cos", "spatial_lag"]
        panel_lags = panel_lags.dropna(subset=lag_cols)
        panel_lags_grouped = {k: v for k, v in panel_lags.groupby("lsoa_code")}

        rf_fi = RandomForestRegressor(n_estimators=RF_TREES, random_state=RF_SEED, n_jobs=-1)
        fi_accum = np.zeros(len(lag_cols))
        fi_count = 0
        lsoas_sample = dense_lsoas[:500] if len(dense_lsoas) > 500 else dense_lsoas
        for lsoa in tqdm(lsoas_sample, desc="  Refitting RF for feature importances"):
            grp = panel_lags_grouped.get(lsoa)
            if grp is None:
                continue
            grp = grp.sort_values("year_month")
            tr = grp[grp["year_month"] <= TRAIN_END]
            X_tr = tr[lag_cols].values
            y_tr = tr["crime_count"].values
            if len(X_tr) > 5:
                try:
                    rf_fi.fit(X_tr, y_tr)
                    fi_accum += rf_fi.feature_importances_
                    fi_count += 1
                except Exception:
                    pass
        if fi_count > 0:
            fi_df = pd.DataFrame({
                "feature": lag_cols,
                "importance": (fi_accum / fi_count).round(6),
            }).sort_values("importance", ascending=False)
            fi_df.to_csv(fi_path, index=False)
            print(f"  Saved: rf_feature_importances.csv (from {fi_count} LSOAs)")
            print(fi_df.head().to_string(index=False))
else:
    panel_dense = panel_dense.merge(
        persistence[["lsoa_code", "hotspot_class"]],
        on="lsoa_code", how="left"
    )
    panel_dense["hotspot_class"] = (panel_dense["hotspot_class"]
                                    .fillna("Non-hotspot"))

    # -- vectorised lag features 
    def build_lag_features(df, n_lags):
        df = df.copy().sort_values(["lsoa_code", "year_month"])
        for i in range(1, n_lags + 1):
            df[f"lag_{i}"] = (df.groupby("lsoa_code")["crime_count"].shift(i))
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
        return df.dropna()

    print("  Building vectorised lag features...")
    panel_lags = build_lag_features(panel_dense, RF_LAGS)

    # -- spatial lag feature 
    # Build LSOA index dict if not already built (checkpoint path skips Stage 6)
    if "lsoa_order" not in dir():
        lsoa_order = lsoa_london["lsoa_code"].tolist()
    if "lsoa_idx" not in dir():
        lsoa_idx = {l: i for i, l in enumerate(lsoa_order)}
    if "w" not in dir():
        print("  Rebuilding weights matrix for spatial lag...")
        w = Queen.from_dataframe(lsoa_london, silence_warnings=True)
        w.transform = "r"

    print("  Building spatial lag features...")
    count_lookup = (
        panel_dense.groupby(["lsoa_code", "year_month"])["crime_count"]
        .sum()
        .to_dict()
    )

    lsoa_order_dense = [l for l in lsoa_order if l in set(dense_lsoas)]
    spatial_lags = {}

    for lsoa in lsoa_order_dense:
        idx_pos = lsoa_idx.get(lsoa, -1)
        if idx_pos == -1:
            continue
        neighbour_codes = [lsoa_order[n] for n in w.neighbors.get(idx_pos, [])
                           if n < len(lsoa_order)]
        if not neighbour_codes:
            continue
        for ym in all_months:
            prev_ym = ym - pd.DateOffset(months=1)
            vals = [count_lookup.get((nc, prev_ym), 0)
                    for nc in neighbour_codes]
            spatial_lags[(lsoa, ym)] = float(np.mean(vals)) if vals else 0.0

    spatial_lag_series = (pd.Series(spatial_lags, name="spatial_lag")
                          .rename_axis(["lsoa_code", "year_month"])
                          .reset_index())
    panel_lags = panel_lags.merge(
        spatial_lag_series, on=["lsoa_code", "year_month"], how="left"
    )
    panel_lags["spatial_lag"] = panel_lags["spatial_lag"].fillna(0)

    lag_feature_cols = (
        [f"lag_{i}" for i in range(1, RF_LAGS + 1)]
        + ["month_sin", "month_cos", "spatial_lag"]
    )
    print(f"  Feature set: {len(lag_feature_cols)} features "
          f"({RF_LAGS} lags + sin/cos + spatial lag)")

    def compute_metrics(actual, predicted):
        mae  = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return mae, rmse

    # Pre-group panel_dense by LSOA for O(1) lookup inside loop
    panel_dense_grouped = {k: v for k, v in panel_dense.groupby("lsoa_code")}
    panel_lags_grouped  = {k: v for k, v in panel_lags.groupby("lsoa_code")}

    # Create RF regressor once - reused for every LSOA (avoids repeated
    # process pool creation from n_jobs=-1)
    rf_model = RandomForestRegressor(
        n_estimators=RF_TREES, random_state=RF_SEED, n_jobs=-1
    )

    # Feature importance accumulator
    feat_importances = np.zeros(len(lag_feature_cols))
    feat_count = 0

    all_preds = []
    lsoas_to_run = dense_lsoas[:SAMPLE_LSOAS] if SAMPLE_LSOAS else dense_lsoas

    for lsoa in tqdm(lsoas_to_run, desc="  Fitting per LSOA"):
        grp = panel_dense_grouped.get(lsoa)
        if grp is None:
            continue

        series = (grp.sort_values("year_month")
                  .set_index("year_month")["crime_count"])
        train  = series[series.index <= TRAIN_END]
        test   = series[series.index >= TEST_START]

        if len(test) < 6:
            continue

        actual = test.values
        hclass = grp["hotspot_class"].iloc[0]

        pred_persist = np.full(len(test), train.iloc[-1])
        pred_ma      = np.full(len(test), train.iloc[-MA_WINDOW:].mean())

        # SARIMA
        try:
            train_s = train.copy()
            train_s.index = pd.DatetimeIndex(train_s.index, freq="MS")
            fit = SARIMAX(
                train_s,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 0, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            raw = fit.forecast(steps=len(test)).values
            cap = max(train_s.max() * 5, 1)
            pred_sarima = np.clip(raw, 0, cap)
        except Exception:
            pred_sarima = pred_ma

        # Random Forest
        try:
            lsoa_lags = panel_lags_grouped.get(lsoa)
            if lsoa_lags is not None:
                lsoa_lags = lsoa_lags.sort_values("year_month")
                tr_lags   = lsoa_lags[lsoa_lags["year_month"] <= TRAIN_END]
                te_lags   = lsoa_lags[lsoa_lags["year_month"] >= TEST_START]
                X_tr = tr_lags[lag_feature_cols].values
                y_tr = tr_lags["crime_count"].values
                X_te = te_lags[lag_feature_cols].values

                if len(X_tr) > 0 and len(X_te) == len(test):
                    rf_model.fit(X_tr, y_tr)
                    pred_rf = np.maximum(rf_model.predict(X_te), 0)
                    # accumulate feature importances
                    feat_importances += rf_model.feature_importances_
                    feat_count += 1
                else:
                    pred_rf = pred_ma
            else:
                pred_rf = pred_ma
        except Exception:
            pred_rf = pred_ma

        for i, ym in enumerate(test.index):
            all_preds.append({
                "lsoa_code":     lsoa,
                "hotspot_class": hclass,
                "year_month":    ym,
                "actual":        actual[i],
                "persist":       pred_persist[i],
                "moving_avg":    pred_ma[i],
                "sarima":        pred_sarima[i],
                "random_forest": pred_rf[i],
            })

    preds_df = pd.DataFrame(all_preds)
    preds_df.to_csv(PROC_DIR / "lsoa_forecast_predictions.csv", index=False)
    print("  Saved: lsoa_forecast_predictions.csv")

    # Save average feature importances
    if feat_count > 0:
        fi_df = pd.DataFrame({
            "feature":    lag_feature_cols,
            "importance": (feat_importances / feat_count).round(6),
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(PROC_DIR / "rf_feature_importances.csv", index=False)
        print("  Saved: rf_feature_importances.csv")
        print("\n  Top 5 RF features by mean importance:")
        print(fi_df.head().to_string(index=False))


# -- Stage 11: Evaluation 
print("\nStage 11: Evaluation by hotspot type  (Research Question 4)")

models = {
    "Persistence":   "persist",
    "Moving avg":    "moving_avg",
    "SARIMA":        "sarima",
    "Random Forest": "random_forest",
}

summary_rows = []
print("\n  Overall:")
for name, col in models.items():
    mae = mean_absolute_error(preds_df["actual"], preds_df[col])
    rmse = np.sqrt(mean_squared_error(preds_df["actual"], preds_df[col]))
    print(f"    {name:<18}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    summary_rows.append({"model": name, "hotspot_class": "ALL",
                         "mae": round(mae, 4), "rmse": round(rmse, 4)})

print("\n  By hotspot class:")
for hclass in sorted(preds_df["hotspot_class"].unique()):
    sub = preds_df[preds_df["hotspot_class"] == hclass]
    n   = sub["lsoa_code"].nunique()
    print(f"\n  {hclass}  ({n} LSOAs)")
    for name, col in models.items():
        mae  = mean_absolute_error(sub["actual"], sub[col])
        rmse = np.sqrt(mean_squared_error(sub["actual"], sub[col]))
        print(f"    {name:<18}  MAE={mae:.3f}  RMSE={rmse:.3f}")
        summary_rows.append({"model": name, "hotspot_class": hclass,
                              "mae": round(mae, 4), "rmse": round(rmse, 4)})

eval_df = pd.DataFrame(summary_rows)
eval_df.to_csv(PROC_DIR / "model_evaluation_by_hotspot_type.csv", index=False)
print("\n  Saved: model_evaluation_by_hotspot_type.csv")

# Evaluation chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, metric in zip(axes, ["mae", "rmse"]):
    pivot = (
        eval_df[eval_df["hotspot_class"] != "ALL"]
        .pivot(index="hotspot_class", columns="model", values=metric)
    )
    pivot.plot(kind="bar", ax=ax, colormap="tab10",
               edgecolor="white", linewidth=0.5)
    ax.set_title(f"{metric.upper()} by Hotspot Class",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, title="Model")
    ax.grid(axis="y", alpha=0.3)
plt.suptitle(
    "Model Evaluation by Hotspot Type \u2014 Research Question 4\n"
    "Train: Mar 2023 \u2013 Feb 2025  |  Test: Mar 2025 \u2013 Feb 2026",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(MAPS_DIR / "05_evaluation_by_hotspot_type.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_evaluation_by_hotspot_type.png")


print("\nDone. Output files in data/processed/ and outputs/maps/")

# Launch dashboard
dashboard_path = Path(__file__).parent / "dashboard.py"
if dashboard_path.exists():
    print("\nAll outputs saved. Launching dashboard...")
    print("Press Ctrl+C to stop the dashboard.\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(dashboard_path)], check=False)
else:
    print("\nTo launch:  streamlit run dashboard.py")
