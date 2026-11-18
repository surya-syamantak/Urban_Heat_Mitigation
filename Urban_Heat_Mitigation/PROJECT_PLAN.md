# Implementation plan and evidence gates

| Phase | Required output | Status | Gate |
|---|---|---|---|
| 1. Data engineering | Verified AOI/grid, season-spanning imagery, aligned weather, urban form, reproducible scripts and catalogue | **Implemented; visual sign-off remains open.** CPCB is retrieved/matched; see `data/processed/PHASE1_QA.md`. | Visually inspect AOI and four ambiguous ward assignments; record a historical urban-form alternative if 2023-24 comparability is needed. |
| 2. Hotspots | Seasonal anomaly, comfort/exposure index, local significance, persistence, ward ranking and maps | **Implemented:** hot/cool severity maps, 492,386-cell priority table, 48-ward rankings, persistence and Gi* clusters. Methods and limits in `outputs/reports/Ahmedabad_PHASE2_QA.md`. | Treat relative quintiles as city-relative; review the 2020 modeled population and station interpolation before policy use. |
| 3. Drivers | Predictor table, interpretable and spatial models, response curves | **Implemented:** OLS/spatial-error/GWR plus scikit-learn Random Forest and XGBoost, 4-fold 5 km GroupKFold tuning, cross-season transfer checks, and native TreeSHAP. Custom CART results remain as legacy comparators. | Spatial metrics are moderate and cross-season transfer is very poor. Use the transfer results as a warning; improve temporal coverage and predictor quality before operational use. |
| 4. Physics model | Baseline and physics-informed model with spatial/temporal validation | **Implemented with documented limits:** 35,000 scene-cell sample; 5-fold spatial blocks; latest-scene and cross-season holdouts; monotonic physics-feature model; perturbation and CPCB context checks. | Review failed cross-season transfer and pure-ML albedo monotonicity; gather measured emissivity/fluxes and neighborhood meteorology before operational use. |
| 5. Scenarios | Intervention eligibility, cooling and population benefit maps | **Implemented as conditional model scenarios:** five interventions, comfort sensitivity, non-additivity check, maps and comparison table; see `outputs/reports/Ahmedabad_PHASE5_QA.md`. | Scenario effects remain unvalidated because cross-season transfer is weak; water sites are heuristic and dispersed. Treat population benefits and comfort changes as sensitivity outputs. |
| 6. Optimization | Greedy baseline, constrained multi-objective plan and uncertainty | **Implemented:** screened grid-cell candidates; greedy and custom NSGA-II; Pareto front; budgeted map, ward/cell lists and error sensitivity. See `outputs/reports/Ahmedabad_PHASE6_QA.md`. | Costs are relative weights; candidates are not parcels; spatial CV uncertainty is broad and temporal transfer is poor. Treat as a coding demonstration and sensitivity plan, not an implementable investment recommendation. |
| 7. Communication | Technical report and presentation | Not ready | Prepare after the results and limitations are supportable. |

## Phase 1 evidence

Ahmedabad is set as the city; the 48-ward, EPSG:32643 30 m grid is canonical.
The period is 2023-04-01 through 2024-02-29, with three April-May hot scenes
and four December-February cool scenes. Seven Landsat C2 L2 scenes have QA
screening, surface temperature, reflectance indices, albedo, seasonal medians,
and provisional rule-based land-cover classes. ERA5 hourly regional data have
been aligned to the seven satellite overpasses. A current OSM snapshot has
been projected and summarized to all grid cells for building, road, and mapped
green/water metrics; tag-derived height and sky-view are proxies.

CPCB hourly weather records from nine Ahmedabad stations have been retrieved
and aligned to each of the seven Landsat overpasses. GHSL 2020 population data
have now been selected and used in Phase 2. The AOI and four low-overlap
ward assignments still need visual review. OSM is a 2026 snapshot, so use its
metrics with temporal caution for 2023-24 imagery. See the catalogue and QA
record for exact methods and limits.

Phase 2 uses GHSL 2020 population density, scene-relative LST anomalies,
CPCB-derived Heat Index surfaces, multiplicative hazard/exposure percentiles,
Gi* clusters with FDR adjustment, and clear-sky hottest-decile persistence. The
hotspot score and classes are relative to Ahmedabad; Heat Index and LST remain
distinct measures. See the Phase 2 QA report and tables/maps for the output.

Phase 3 assembles a full valid-cell table for both seasonal Landsat composites,
including spectral indices/albedo/rule-based LULC, OSM urban-form fields,
coordinates and ERA5 background context. It retains OLS, feasible spatial-error
GLS, custom CART comparators, adaptive GWR, and response curves. The default
Phase 3 run then fits scikit-learn RandomForestRegressor and XGBoost models in
the launching Python, tunes two configurations per model with 4-fold GroupKFold
on 5 km spatial blocks, checks hot-to-cool and cool-to-hot transfer, and
calculates native TreeSHAP. Cross-season transfer is very poor; the QA report
states the limitations and reproduction steps.

Phase 4 estimates net-radiation, EF, and thermal-inertia features and compares
a custom CART forest with a constrained physics-feature regression. The model
uses an ERA5-air-temperature radiation proxy to avoid using observed Landsat
LST in the predictors; an estimated Rn based on observed Ts is catalogued but
withheld from training. The spatial CV physics-feature model outperforms the
forest in this sample, while cross-season transfer is poor. The forest fails
the higher-albedo plausibility check; the constrained model passes both tested
perturbations. CPCB comparison is LST versus 2 m air temperature and is
contextual, not direct validation. Fluxes and emissivity remain estimated or
unobserved; see `outputs/reports/Ahmedabad_PHASE4_QA.md`.

## Interpretation checks

- ERA5 is a coarse regional background, not a neighborhood microclimate field.
- Landsat LST is radiometric surface temperature, not 2 m air temperature or
  personal thermal comfort.
- Current OSM is not contemporaneous with the Landsat study window; interpret
  urban-form fields as current proxies until a suitable historical source is
  added.
- Feature importance is predictive association, not causal effect.
