"""Publication figures for the Geographies manuscript, drawn from the pipeline CSVs.

Run with:  python3 make_figures.py <crime_project_root> <output_dir>
All figures are saved as 600 dpi PNG at the width they are printed (17.5 cm full width).
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

matplotlib.use("Agg")
ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
PROC = ROOT / "data" / "processed"
DPI = 600
W = 6.9  # inches, MDPI full text width

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

CLASS_COL = {"Persistent hotspot": "#b2182b", "Intermittent hotspot": "#f5a623", "Non-hotspot": "#e4e4e4"}
LISA_COL = {"HH": "#b2182b", "LL": "#4575b4", "LH": "#a6d3e8", "HL": "#f5a623", "NS": "#e4e4e4"}
MODEL_COL = {"Persistence": "#9a9a9a", "Moving avg": "#6b8fb5", "SARIMA": "#e08a2c", "Random Forest": "#1b7f6b"}

# ---------------------------------------------------------------- data
cls = pd.read_csv(PROC / "lsoa_hotspot_classification.csv")
lisa = pd.read_csv(PROC / "lsoa_monthly_lisa.csv")
ev = pd.read_csv(PROC / "model_evaluation_by_hotspot_type.csv")

shp = next((ROOT / "data" / "boundaries").glob("*.shp"))
gdf = gpd.read_file(shp)
code_col = next(c for c in gdf.columns if c.upper().startswith("LSOA") and c.upper().endswith("CD"))
gdf = gdf.rename(columns={code_col: "lsoa_code"})
gdf = gdf[gdf["lsoa_code"].isin(cls["lsoa_code"])].to_crs(27700)
gdf = gdf.merge(cls[["lsoa_code", "hotspot_class"]], on="lsoa_code", how="left")

core = gdf[gdf.hotspot_class.isin(["Persistent hotspot", "Intermittent hotspot"])]
minx, miny, maxx, maxy = core.total_bounds
pad = 2500
ZOOM = (minx - pad, maxx + pad, miny - pad, maxy + pad)
LON = gdf.total_bounds  # whole study area


def style_map(ax, extent=None):
    ax.set_aspect("equal")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color("#666")
    if extent:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])


def scalebar(ax, extent, length_m, label):
    x0 = extent[0] + 0.05 * (extent[1] - extent[0]); y0 = extent[2] + 0.05 * (extent[3] - extent[2])
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", lw=1.4, solid_capstyle="butt")
    ax.text(x0 + length_m / 2, y0 + 0.012 * (extent[3] - extent[2]), label, ha="center", va="bottom", fontsize=6.5)


# ---------------------------------------------------------------- Figure 1: persistence map
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
base = gdf.assign(c=gdf.hotspot_class.map(CLASS_COL))
base.plot(ax=a, color=base.c, linewidth=0.0)
core.assign(c=core.hotspot_class.map(CLASS_COL)).plot(ax=a, color="none", edgecolor="#333", linewidth=0.35)
style_map(a, (LON[0], LON[2], LON[1], LON[3]))
a.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2], fill=False, ec="black", lw=0.8, ls="--"))
a.set_title("(a) Greater London", loc="left")
scalebar(a, (LON[0], LON[2], LON[1], LON[3]), 10000, "10 km")

for k, col in CLASS_COL.items():
    gdf[gdf.hotspot_class == k].plot(ax=b, color=col, edgecolor="#8a8a8a" if k == "Non-hotspot" else "#222",
                                     linewidth=0.15 if k == "Non-hotspot" else 0.35)
style_map(b, ZOOM)
b.set_title("(b) Central London", loc="left")
scalebar(b, ZOOM, 2000, "2 km")
n = cls.hotspot_class.value_counts()
handles = [Patch(fc=CLASS_COL[k], ec="#555", lw=0.4,
                 label=f"{k.replace(' hotspot', '')} (n = {n[k]:,})") for k in CLASS_COL]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.1, wspace=0.04)
fig.savefig(OUT / "Figure3_persistence_map.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: LISA snapshots
snaps = ["2023-03-01", "2024-03-01", "2025-03-01"]
labels = ["March 2023", "March 2024", "March 2025"]
fig, axes = plt.subplots(2, 3, figsize=(W, 4.2), gridspec_kw={"height_ratios": [1.15, 1]})
for j, (s, lab) in enumerate(zip(snaps, labels)):
    m = lisa[lisa.year_month == s][["lsoa_code", "lisa_type"]]
    g = gdf.merge(m, on="lsoa_code", how="left")
    g["lisa_type"] = g["lisa_type"].fillna("NS")
    nHH = int((g.lisa_type == "HH").sum())
    for r, ext in enumerate([(LON[0], LON[2], LON[1], LON[3]), ZOOM]):
        ax = axes[r, j]
        g.assign(c=g.lisa_type.map(LISA_COL)).plot(ax=ax, color=g.lisa_type.map(LISA_COL), linewidth=0.0 if r == 0 else 0.1,
                                                   edgecolor="#9a9a9a")
        hh = g[g.lisa_type == "HH"]
        if len(hh):
            hh.plot(ax=ax, color=LISA_COL["HH"], edgecolor="#222", linewidth=0.2 if r == 1 else 0.0)
        style_map(ax, ext)
        if r == 0:
            ax.set_title(f"{chr(97 + j)}) {lab} (HH = {nHH})", loc="left")
            if j == 0:
                scalebar(ax, ext, 10000, "10 km")
        elif j == 0:
            scalebar(ax, ext, 2000, "2 km")
    axes[0, j].add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2], fill=False, ec="black", lw=0.6, ls="--"))
names = {"HH": "High-High", "LL": "Low-Low", "LH": "Low-High", "HL": "High-Low", "NS": "Not significant"}
fig.legend(handles=[Patch(fc=LISA_COL[k], ec="#555", lw=0.4, label=names[k]) for k in LISA_COL], loc="lower center",
           ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.08, wspace=0.04, hspace=0.02)
fig.savefig(OUT / "Figure2_LISA_snapshots.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: monthly HH count
hh = lisa[lisa.lisa_type == "HH"].groupby("year_month").lsoa_code.nunique()
hh.index = pd.to_datetime(hh.index)
fig, ax = plt.subplots(figsize=(W, 2.5))
ax.bar(hh.index, hh.values, width=24, color="#b2182b", edgecolor="none")
ax.axvline(pd.Timestamp("2025-02-15"), color="black", ls="--", lw=0.8)
ax.text(pd.Timestamp("2025-02-05"), hh.max() * 1.0, "Test period\nstarts (Mar 2025)", fontsize=7, va="top", ha="right")
ax.axhline(hh.median(), color="#555", lw=0.8, ls=":")
ax.text(pd.Timestamp("2023-09-20"), hh.median() + 3, f"Median = {hh.median():.0f}", fontsize=7, color="#333", ha="center")
ax.set_ylabel("HH LSOAs per month (count)")
ax.set_xlabel("Month")
ax.set_ylim(0, hh.max() * 1.08)
ax.set_xlim(pd.Timestamp("2023-02-01"), pd.Timestamp("2026-03-20"))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
fig.tight_layout()
fig.savefig(OUT / "Figure1_monthly_HH_count.png", dpi=DPI)
plt.close(fig)
hh.to_csv(OUT / "monthly_HH_counts_used_in_Figure1.csv", header=["hh_lsoas"])

# ---------------------------------------------------------------- Figure 4: MAE by class and model (matched comparison)
mb = pd.read_csv(OUT / "matched_baseline_evaluation.csv")
order = ["All dense LSOAs", "Non-hotspot", "Intermittent hotspot", "Persistent hotspot"]
titles = {"All dense LSOAs": "(a) All LSOAs", "Non-hotspot": "(b) Non-hotspot", "Intermittent hotspot": "(c) Intermittent",
          "Persistent hotspot": "(d) Persistent"}
M4 = [("Persistence (fixed origin)", "#bdbdbd", ""), ("Moving average (fixed origin)", "#a6c1db", ""),
      ("SARIMA (fixed origin)", "#e08a2c", ""), ("Random Forest (one-step)", "#1b7f6b", ""),
      ("Persistence (one-step)", "#5a5a5a", "////"), ("Moving average (one-step)", "#2f5f8f", "////")]
fig, axes = plt.subplots(1, 4, figsize=(W, 2.9))
for ax, k in zip(axes, order):
    d = mb[mb.group == k].set_index("model").loc[[m[0] for m in M4]]
    n_l = int(d.n_lsoas.iloc[0])
    for i, (name, col, h) in enumerate(M4):
        ax.bar(i, d.loc[name, "mae"], color=col, hatch=h, edgecolor="white" if h else "none", linewidth=0.0, width=0.8)
        ax.text(i, d.loc[name, "mae"] + d.mae.max() * 0.015, f"{d.loc[name, 'mae']:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks([]); ax.set_title(f"{titles[k]}\n(n = {n_l:,})", loc="left", fontsize=7.5)
    ax.set_ylim(0, d.mae.max() * 1.15)
axes[0].set_ylabel("MAE (crimes per LSOA per month)")
fig.legend(handles=[Patch(fc=c, hatch=h, ec="white" if h else c, label=n) for n, c, h in M4], loc="lower center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.2, wspace=0.42)
fig.savefig(OUT / "Figure4_MAE_by_class.png", dpi=DPI)
plt.close(fig)
print("done", sorted(p.name for p in OUT.iterdir()))
