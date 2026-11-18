# Urban Heat Mitigation in Ahmedabad

Phase 1 data engineering for Ahmedabad is implemented in a reproducible local
workflow: 30 m grid and ward lookup, dynamically selected Landsat 8/9 scenes
across multiple years of hot and cool seasons, aligned ERA5 overpass context, and OSM-derived urban-form
predictors. See the [Phase 1 QA record](data/processed/PHASE1_QA.md) and
[data catalogue](data/dataset_audit.csv) for methods, source details, and open
limitations.

## Study design

- City: Ahmedabad, Gujarat, India.
- Period: 2021-01-01 to 2025-12-31. Recurring hot season is March-May and cool
  season is December-February. Every Landsat 8/9 acquisition with scene cloud
  cover at or below 60% is selected; exact dates, cloud percentages, year and
  season are recorded in the manifest. Per-pixel QA masks cloudy/invalid pixels.
- Grid: 30 m, WGS 84 / UTM zone 43N (EPSG:32643), 492,386 clipped cells.
- AOI: 48 wards, 441.058 km2 by stored ward areas.
- There is no fixed acquisition-date cap. If an AOI crosses multiple WRS rows,
  tile records from the same acquisition date must be counted as one date.
  Seasonal composites pool the eligible dates; scene-level rasters and the
  manifest retain per-date/year observations. LST is surface temperature, not
  2 m air temperature or comfort.

## Run

Run `main.py` from the project root. QGIS 3.44 Python handles GDAL/vector
processing; the Phase 3 Random Forest/XGBoost/CatBoost/SHAP step uses the project
`.venv` when it exists, even if `main.py` was launched by a different Python.
The pinned requirements target Python 3.11. Create and populate the project
environment with `py -3.11 -m venv .venv`, then
`.venv\Scripts\python.exe -m pip install -r requirements.txt`. Set
`QGIS_PYTHON` if the QGIS launcher is not found automatically. Run one step or
the available pipeline from the project root:

```powershell
python main.py era5
python main.py landsat
python main.py osm
python main.py cpcb
python main.py phase1
python main.py phase2
python main.py phase3
python main.py phase4
python main.py phase5
python main.py phase6
```

Each `python main.py phaseN` run prints a phase-specific terminal summary from the generated outputs after that phase completes. The summaries use the project's actual model names and explicitly label approximate attributions, relative costs, and sensitivity estimates.

Network access is needed to retrieve source observations. Phase 1 stores raw
inputs under `data/raw/`, derived rasters/tables under `data/processed/`,
scripts under `src/`, and parameters in `config.yaml`. The pipeline leaves raw
inputs intact. `phase1` discovers and downloads Landsat scenes first, aligns
ERA5 to the scene manifest, and runs OSM and CPCB acquisition/matching;
`phase2` downloads GHSL population and generates hotspot scores, Gi* clusters,
persistence, maps, and rankings. `phase3` assembles seasonal predictor tables
and produces OLS/spatial-error/GWR analyses, scikit-learn Random Forest,
XGBoost, and CatBoost models tuned with 3 km spatial-block CV, cross-season transfer checks,
native TreeSHAP rankings/figures, response curves, and local driver maps.
Earlier custom CART and approximate-attribution outputs are retained as legacy
comparators; see the Phase 3 QA report for the very poor cross-season transfer.
`phase4` builds radiation/moisture/thermal proxy features, compares a pure CART
forest with a constrained physics-feature model, and writes spatial/temporal
metrics, plausibility perturbations, and CPCB context checks.

## Known limitations

CPCB hourly data are requested for nine Ahmedabad stations for each study year.
Local CPCB archives yielded 72 scene/station matches; CPCB repository 404s for
other station-years are recorded. Phase 2 falls back to matched ERA5 temperature and
humidity when fewer than three CPCB station pairs exist for a scene. Population
is modeled GHSL 2020, not a 2023 census. OSM is a 2026
snapshot, not a 2023-24 historic extract.
OSM height and sky-view variables are proxies with incomplete height tagging;
the rule-based LULC has not been independently validated. The AOI and four
low-overlap ward assignments still need visual sign-off. Do not claim
population-weighted exposure; CPCB overpass values are context and do not
independently validate satellite LST.

The rule-based LULC, OSM urban-form proxies, and feasible-GLS spatial error
model require validation. Phase 3 now uses scikit-learn RandomForestRegressor,
XGBoost, CatBoost, 10-fold GroupKFold tuning over 3 km spatial blocks, and native SHAP
TreeExplainer in the project `.venv`. Its cross-season transfer
checks are very poor, so these models should not be used for operational
prediction across seasons. Phase 3 uses pooled seasonal composites and does not
yet establish year-by-year model transfer. Phase 2’s Heat Index is a
shade/light-wind metric; its
population layer is a modeled 2020 100 m source resampled as density to 30 m.
See [the Phase 2 methods and QA report](outputs/reports/Ahmedabad_PHASE2_QA.md).
Phase 4 estimates longwave radiation and emissivity, and uses EF and thermal
inertia proxies because measured fluxes and emissivity rasters are unavailable.
Its CPCB comparison is between LST and air temperature and is contextual, not
a direct validation. The pure forest fails the albedo monotonicity check; the
physics-informed model passes its tested perturbations. See
[the Phase 4 model report](outputs/reports/Ahmedabad_PHASE4_QA.md).
Keep LST, air temperature, and human thermal comfort distinct.


Phase 5 runs five conditional cooling scenarios with the constrained Phase 4 model and exports a comparison table, interaction test, rasters, maps, and QA report. Heat Index changes are sensitivity estimates only. Cross-season transfer is weak, so scenarios are not validated forecasts or site plans. See [the Phase 5 report](outputs/reports/Ahmedabad_PHASE5_QA.md).

Run `python main.py phase5` first, then `python main.py phase6`. Phase 6 uses Phase 5 scenario rasters to run a greedy baseline and custom NSGA-II portfolio search, exporting a Pareto table, recommended cell and ward lists, a map, and uncertainty sensitivity. Costs are relative units because verified parcel cost data are unavailable. See [the Phase 6 report](outputs/reports/Ahmedabad_PHASE6_QA.md).
