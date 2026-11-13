# London crime hotspots

Code and derived data for the manuscript *Spatio-Temporal Analysis and Short-Term Forecasting of Urban Crime Hotspots in Greater London Using Open Police Data* (Lambate and Wang, submitted to *Geographies*).

The pipeline aggregates Metropolitan Police Service (MPS) street-level crime records to LSOAs, finds monthly High-High clusters with Local Moran's I (Benjamini-Hochberg corrected), classifies LSOAs by how many of 36 months they were hotspots, and compares persistence, moving-average, SARIMA and Random Forest forecasts.

## Contents

| File or folder | Purpose |
|---|---|
| `spatio_temporal_hotspot_analysis.py` | Full pipeline: cleaning, LISA, DBSCAN, persistence classes, forecasting |
| `matched_baselines.py` | One-step-ahead persistence and moving-average baselines on matched LSOAs |
| `make_figures.py` | Publication figures (600 dpi) |
| `dashboard.py` | Streamlit dashboard |
| `data/processed/` | Derived files produced by the pipeline |
| `figures/` | Manuscript figures and the table behind Figure 4 |

## Data

The raw data are not stored here. Download them and place them as follows:

- MPS street-level files for March 2023 to February 2026 from <https://data.police.uk/data/> into `data/raw/YYYY-MM/`.
- LSOA (December 2021) Boundaries EW BGC V5 from the ONS Geoportal into `data/boundaries/`.

## Run

```
pip install -r requirements.txt
python spatio_temporal_hotspot_analysis.py
python matched_baselines.py . figures
python make_figures.py . figures
streamlit run dashboard.py
```

The pipeline skips any stage whose output already exists in `data/processed/`. Random seeds are fixed (42).

## Known issues

- `lsoa_monthly_crime_panel.csv` holds duplicate rows for 110 LSOA codes that map to two LSOA names (290,844 rows instead of 286,344). Counts for those LSOAs may be inflated. `matched_baselines.py` excludes them.
- `rf_feature_importances.csv` is not included. The saved version was built with the spatial-lag column set to zero and does not describe the fitted models.
- In the pipeline, persistence and moving-average forecasts are fixed-origin (12 steps from February 2025), while the Random Forest is one-step-ahead. Use `matched_baselines.py` for a like-for-like comparison.

## Licence and attribution

Code: MIT (see `LICENSE`). Derived data contain public sector information licensed under the Open Government Licence v3.0, including data from the Metropolitan Police Service published on data.police.uk and ONS boundary data (Crown copyright).

## Citation

See `CITATION.cff`. DOI: to be added after the Zenodo archive is created.
