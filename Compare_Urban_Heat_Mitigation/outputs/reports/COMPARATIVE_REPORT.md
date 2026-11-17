# Urban Heat Mitigation: Three-city comparison

This report summarizes outputs already present in the three project folders. It does not rerun city analyses.

## Projects and input availability

| City | Project folder | Study period | Unique Landsat dates | Item/tile records | Grid cells | Wards | Cell size (m) | ERA5 hours | CPCB match rows | Phase 2 scenes with CPCB support |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ahmedabad | `Urban_Heat_Mitigation` | 2021-01-01 to 2025-12-31 | 88 | 88 | 492386 | 48 | 30 | 43824 | 369 | 41 |
| Chennai | `Urban_Heat_Mitigation_Chennai` | 2021-01-01 to 2025-12-31 | 80 | 80 | 483211 | 200 | 30 | 43824 | 0 | 0 |
| Nagpur | `Urban_Heat_Mitigation_Nagpur` | 2021-01-01 to 2025-12-31 | 83 | 160 | 234587 | 38 | 30 | 43824 | 0 | 0 |

The detailed artifact checklist is in `outputs/tables/input_availability.csv`.
The scene manifest supplies each acquisition's season; the sampling table distinguishes distinct dates from satellite tile/item records.

## Landsat sampling by year and season

| City | Year | Season | Distinct acquisition dates | Landsat item/tile records |
|---|---:|---|---:|---:|
| Ahmedabad | 2021 | cool | 5 | 5 |
| Ahmedabad | 2021 | hot | 3 | 3 |
| Ahmedabad | 2022 | cool | 11 | 11 |
| Ahmedabad | 2022 | hot | 9 | 9 |
| Ahmedabad | 2023 | cool | 11 | 11 |
| Ahmedabad | 2023 | hot | 8 | 8 |
| Ahmedabad | 2024 | cool | 11 | 11 |
| Ahmedabad | 2024 | hot | 11 | 11 |
| Ahmedabad | 2025 | cool | 9 | 9 |
| Ahmedabad | 2025 | hot | 10 | 10 |
| Chennai | 2021 | cool | 6 | 6 |
| Chennai | 2021 | hot | 5 | 5 |
| Chennai | 2022 | cool | 10 | 10 |
| Chennai | 2022 | hot | 8 | 8 |
| Chennai | 2023 | cool | 8 | 8 |
| Chennai | 2023 | hot | 8 | 8 |
| Chennai | 2024 | cool | 8 | 8 |
| Chennai | 2024 | hot | 9 | 9 |
| Chennai | 2025 | cool | 9 | 9 |
| Chennai | 2025 | hot | 9 | 9 |
| Nagpur | 2021 | cool | 7 | 12 |
| Nagpur | 2021 | hot | 5 | 10 |
| Nagpur | 2022 | cool | 9 | 17 |
| Nagpur | 2022 | hot | 7 | 14 |
| Nagpur | 2023 | cool | 10 | 19 |
| Nagpur | 2023 | hot | 8 | 16 |
| Nagpur | 2024 | cool | 9 | 17 |
| Nagpur | 2024 | hot | 10 | 19 |
| Nagpur | 2025 | cool | 9 | 18 |
| Nagpur | 2025 | hot | 9 | 18 |

## City rankings

Budget caps are relative units rather than currency. Equal values receive the same rank. Budget use is ranked separately using selected-plan cost.

### Budget limit

| Rank | City | Budget limit (relative units) |
|---:|---|---:|
| 1 | Ahmedabad | 10,000 |
| 1 | Chennai | 10,000 |
| 1 | Nagpur | 10,000 |

### Budget used

| Rank | City | Selected-plan cost (relative units) |
|---:|---|---:|
| 1 | Chennai | 9,999.9 |
| 2 | Ahmedabad | 3,158.0 |
| 3 | Nagpur | 1,425.6 |

### FDR-significant hot-hotspot area

Area is the summed grid-cell area for Gi* high-hotspot cells with FDR q ≤ 0.05.

| Rank | City | Area (km²) | Significant cells |
|---:|---|---:|---:|
| 1 | Ahmedabad | 117.32 | 130677 |
| 2 | Chennai | 110.38 | 122647 |
| 3 | Nagpur | 51.80 | 57555 |

### Best Phase 3 model by season

Selected by lowest spatial-CV RMSE within each city and season.

| City | Season | Best model | RMSE (°C) | R² |
|---|---|---|---:|---:|
| Ahmedabad | Hot | Random Forest (scikit-learn) | 1.403 | 0.675 |
| Ahmedabad | Cool | CatBoost | 0.953 | 0.593 |
| Chennai | Hot | Random Forest (scikit-learn) | 1.535 | 0.637 |
| Chennai | Cool | CatBoost | 1.297 | 0.555 |
| Nagpur | Hot | Random Forest (scikit-learn) | 1.285 | 0.699 |
| Nagpur | Cool | Random Forest (scikit-learn) | 1.036 | 0.630 |

## Recommended comparisons

- **Seasonal heat:** compare Phase 2 hot/cool city median and 90th-percentile LST summaries. Scene dates are averaged first so multiple Landsat path/row images from one acquisition date do not count as separate dates.
- **Model reliability and drivers:** compare Phase 3 spatial-CV RMSE/R², cross-season transfer metrics, and the top native-SHAP drivers by city, season, and model.
- **Mitigation scenarios:** compare Phase 5 treated area, cooling, and GHSL 2020 population benefit for matching scenario names.
- **Optimization and equity:** compare Phase 6 plans under the recorded relative budget and inspect how selected cells/benefits align with each city’s highest heat-exposure wards.

## Interpretation limits

The current configs use the 2021–2025 study window and 30 m analysis grids. The city projects now select all Landsat 8/9 scenes in configured hot/cool months that pass the shared scene-cloud threshold and per-pixel QA; date counts may differ by year, season and city. Phase 2 summarizes available dates rather than synchronized same-day observations. Tile/item records can exceed unique acquisition dates when an AOI spans multiple Landsat path/row tiles. CPCB coverage varies by city: Ahmedabad: 369 match rows, 41 Phase 2 scenes with CPCB support; Chennai: 0 match rows, 0 Phase 2 scenes with CPCB support; Nagpur: 0 match rows, 0 Phase 2 scenes with CPCB support. Phase 2 uses ERA5 background fallback when CPCB station support is insufficient. Cities differ in climate, AOI size, ward structure, and valid-cell coverage, so seasonal means and relative heat-stress scores are not direct measures of city-to-city causation. Heat-stress quintiles are city-relative; use absolute LST summaries and population measures for cross-city levels.

Phase 3 SHAP values describe model associations, not causal cooling effects. Cross-season transfer is a stress test, not independent multi-year validation. Phase 5 estimates depend on each project’s scenario inputs and valid coverage. Phase 6 costs are relative units rather than currency; compare plans only alongside budget, candidate-pool size, and the stated assumptions. Ward-level equity shares are calculated within each city because ward areas and populations are not equivalent across cities.

## Generated outputs

Comparison CSVs are in `outputs/tables/`; charts are in `outputs/figures/`. Missing source artifacts are recorded in the availability table and omitted from the relevant comparison rather than treated as zero.
