# Compare Urban Heat Mitigation projects

This folder contains a reusable script that compares existing outputs from:

- `Urban_Heat_Mitigation_Ahmedabad`
- `Urban_Heat_Mitigation_Chennai`
- `Urban_Heat_Mitigation_Nagpur`

The comparison reads project data and outputs without changing those projects. It writes summary tables, charts, and a report under this folder's `outputs/` directory.

## Run

Open PowerShell and run:

```powershell
cd C:\Users\SURYA\Desktop\Compare_Urban_Heat_Mitigation
python .\compare_projects.py
```

If the selected Python cannot load Matplotlib or PyYAML, the script automatically restarts with an available project Python 3.12 environment. You can also run `run_comparison.ps1`. If using a separate environment, install this folder's `requirements.txt` first. The three project folders must remain alongside this folder on the Desktop.

## Comparisons generated

- `project_context.csv`: study window, Landsat item and distinct-date counts, ERA5 record counts, CPCB match counts, grid and ward counts, resolution, population source, and heat-index context.
- `landsat_sampling_summary.csv`: distinct Landsat dates and item/tile records by city, year, and season. This makes Nagpur’s multiple WRS tiles per acquisition date explicit.
- `phase2_seasonal_heat_summary.csv`: hot/cool acquisition-date mean median and 90th-percentile LST summaries. Multiple tile records on the same date are averaged before dates are combined; season labels come from each project's Landsat manifest.
- `phase3_spatial_cv_metrics.csv` and `phase3_cross_season_transfer.csv`: native Random Forest and XGBoost validation metrics and seasonal transfer results.
- `phase3_top_native_shap_drivers.csv`: top ten drivers by mean absolute TreeSHAP for each city, season, and model.
- `phase4_validation_metrics.csv` and `phase4_physical_perturbations.csv`: spatial/temporal validation and model response to predictor changes.
- `phase5_scenario_comparison.csv`: cooling, treated area, and estimated population benefit for each scenario.
- `phase6_optimization_summary.csv`, `phase6_ward_allocations.csv`, and `phase6_ward_equity_summary.csv`: recommended portfolios and how benefits align with high-exposure wards.
- The terminal summary ranks cities by Phase 6 budget cap and selected-plan spend, ranks the FDR-significant Gi* hotspot area, and reports each city's best Phase 3 model for hot and cool seasons using spatial-CV RMSE (lower is better).
- `phase6_budget_limit_ranking.csv`, `phase6_budget_used_ranking.csv`, `phase2_significant_hotspot_area_ranking.csv`, and `phase3_best_models_by_city_season.csv`: the rankings and model winners printed in the terminal. Cities tied on a budget cap share a rank; budgets and costs are relative units, not currency.
- `ward_heat_investment_equity.csv`: ward-level heat exposure and selected intervention allocation within each city.
- `outputs/figures/`: comparison charts.
- `outputs/reports/COMPARATIVE_REPORT.md`: interpretation and limitations.

`input_availability.csv` shows which source and analysis outputs were found. It checks the current manifests, catalogue, ERA5 and CPCB matches, and Phase 2–6 artifacts. Missing artifacts are left blank/omitted rather than treated as zero.

## Comparability notes

The refreshed projects use a 2021–2025 study window and a 30 m analysis grid. Each city targets two hot-season and two cool-season acquisition dates per year. A date may have multiple Landsat item/tile records when the AOI spans WRS path/row tiles, so compare unique dates and tile records separately. Phase 2 values summarize each city's selected dates and are not synchronized same-day observations. `project_context.csv` distinguishes CPCB match rows from Phase 2 scenes with CPCB support; Phase 2 uses ERA5 fallback where station support is insufficient. City-relative heat-stress quintiles should not be interpreted as comparable absolute heat levels.

Phase 3 SHAP values describe fitted-model associations, not causal effects. Cross-season transfer is a stress test and not multi-year validation. Phase 5 outputs depend on each project's intervention assumptions and valid coverage. Phase 6 costs are relative units, not currency; inspect the recorded budget and candidate-pool sizes alongside the recommendations. Ward-equity shares are calculated within each city because wards are not equal in area or population across cities.
