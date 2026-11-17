# Urban Heat Mitigation in Chennai

This project estimates seasonal Landsat land-surface temperature (LST), maps spatial heat patterns, compares environmental drivers, and screens conditional mitigation scenarios for Greater Chennai Corporation (GCC).

## Study design

- City: Chennai, Tamil Nadu, India.
- Period: 1 January 2021 to 31 December 2025.
- Scene selection: every Landsat 8/9 acquisition in the configured hot/cool months and 2021–2025 window with scene cloud cover at or below 60%; per-pixel QA removes cloud, shadow and other flagged pixels. There is no fixed date cap. Overlapping WRS tile records are not independent acquisition dates.
- Seasons: hot pre-monsoon March–May and cool winter December–February.
- The city-specific AOI, ward boundaries, grid, and CRS are retained.
- Population: GHSL epoch 2020, a modeled population source resampled from 100 m to the analysis grid.
- Heat index: CPCB AT/RH where at least three paired stations are available; otherwise a matched ERA5 background estimate is used.

## Run order

Run commands from the project root. The original project workflow uses `phase1` as the ordered data-acquisition step: it runs Landsat first, then ERA5, OSM/urban-form processing, and CPCB matching. Do not run those four substeps again after `phase1` unless you intentionally want to rerun only one data source.

If this is a fresh setup and the city AOI, ward files, or analysis grid are missing, create them once before Phase 1:

```powershell
python main.py aoi
python main.py wards
python main.py prepare_grid
```

Then run the full analysis in order:

```powershell
python main.py phase1
python main.py phase2
python main.py phase3
python main.py phase4
python main.py phase5
python main.py phase6
```

If the AOI, ward, and grid files are already present, start directly with `python main.py phase1`. Phase 1 discovers all scenes matching the common date, season, WRS, and 60% cloud rules, then applies per-pixel QA and creates the manifest, scene rasters, seasonal composites, ERA5 matches, urban-form predictors, and CPCB matches. After the expanded manifest and composites are built, rerun Phases 2–6 to regenerate downstream analyses, maps, tables, and reports. Count unique acquisition dates separately from tile records.

You can launch `main.py` with Python 3.14. It automatically relaunches under this project's configured Python 3.12 environment for non-QGIS steps; QGIS-dependent scripts use the installed QGIS Python runtime.

## Interpretation limits

LST is daytime satellite surface temperature, not 2 m air temperature or human thermal comfort. Seasonal composites pool selected scenes across five years. ERA5 is coarse background context, GHSL is modeled 2020 population rather than a current census, and resampling does not create true 30 m population detail. OSM completeness varies. Cross-season prediction and mitigation scenarios remain uncertain. Phase 6 costs are relative units, not verified project costs.

The selected acquisition dates and item IDs are recorded in data/processed/Landsat/Landsat_scene_manifest.csv. ERA5, CPCB, data catalogue, and phase QA reports are written under data/ and outputs/.

## Phase 3 year-held-out model comparison

Phase 3 now builds an acquisition-level predictor table from the existing Landsat scene rasters and fits Random Forest, XGBoost, and CatBoost under spatial-block cross-validation and compares all three using leave-one-year-out validation, separately for hot and cool seasons. The table retains scene ID, acquisition date, year, season, and grid cell. Phase 1 and Phase 2 do not need to be rerun when their current scene rasters, urban-form data, grid, and seasonal LULC rasters are present. Install the pinned dependencies with python -m pip install -r requirements.txt, then run python main.py phase3. Results are written to outputs/tables/*phase3_year_holdout* and summarized in outputs/reports/*PHASE3_QA.md.
