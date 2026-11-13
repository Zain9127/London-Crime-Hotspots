"""Matched one-step-ahead baselines for the forecasting comparison.

The pipeline's persistence and moving-average baselines are fixed-origin forecasts: one value
taken at the end of the training window and held flat for all 12 test months. The Random Forest
predicts each test month from the observed previous 12 months (one-step-ahead). This script
adds persistence and 3-month moving-average baselines on the same one-step-ahead footing and
evaluates all six forecasts on the same LSOAs.

LSOA codes that map to two LSOA names in the panel (boundary revisions, 110 codes) are excluded
because the panel holds duplicated rows for them.

Run:  python3 matched_baselines.py <crime_project_root> <output_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
proc = root / "data" / "processed"

pr = pd.read_csv(proc / "lsoa_forecast_predictions.csv", parse_dates=["year_month"])
panel = pd.read_csv(proc / "lsoa_monthly_crime_panel.csv", parse_dates=["year_month"])

n_names = panel.groupby("lsoa_code").lsoa_name.nunique()
dup = set(n_names[n_names > 1].index)
clean = panel[~panel.lsoa_code.isin(dup)]
wide = clean.pivot(index="year_month", columns="lsoa_code", values="crime_count").sort_index()

p1 = wide.shift(1).stack().rename("persistence_1step").reset_index()
ma3 = ((wide.shift(1) + wide.shift(2) + wide.shift(3)) / 3).stack().rename("movavg3_1step").reset_index()
df = (pr[~pr.lsoa_code.isin(dup)]
      .merge(p1, on=["lsoa_code", "year_month"]).merge(ma3, on=["lsoa_code", "year_month"]))

# sanity check: the panel and the prediction file agree on observed counts
chk = df.merge(clean[["lsoa_code", "year_month", "crime_count"]], on=["lsoa_code", "year_month"])
assert np.allclose(chk.actual, chk.crime_count), "actual counts differ between panel and predictions"

MODELS = [
    ("Persistence (fixed origin)", "persist"),
    ("Moving average (fixed origin)", "moving_avg"),
    ("SARIMA (fixed origin)", "sarima"),
    ("Random Forest (one-step)", "random_forest"),
    ("Persistence (one-step)", "persistence_1step"),
    ("Moving average (one-step)", "movavg3_1step"),
]
rows = []
groups = [("All dense LSOAs", df)] + [(k, g) for k, g in df.groupby("hotspot_class")]
for label, g in groups:
    for name, col in MODELS:
        e = g.actual - g[col]
        rows.append({"group": label, "n_lsoas": g.lsoa_code.nunique(), "mean_actual": round(g.actual.mean(), 2),
                     "model": name, "mae": round(e.abs().mean(), 3), "rmse": round(float(np.sqrt((e ** 2).mean())), 3)})
res = pd.DataFrame(rows)
res.to_csv(out / "matched_baseline_evaluation.csv", index=False)
print(res.to_string(index=False))
sar_fallback = float(np.isclose(pr.sarima, pr.moving_avg).mean())
print(f"\nShare of forecasts where SARIMA equals the moving-average fallback: {sar_fallback:.3f}")
print("LSOAs excluded for duplicate names:", len(dup))
