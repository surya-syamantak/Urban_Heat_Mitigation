"""Run data collection and analysis phases 1 through 6."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def qgis_python() -> str:
    configured = os.environ.get("QGIS_PYTHON", "")
    if configured:
        return configured
    if os.name == "nt":
        candidates = [
            Path(r"C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat"),
            Path(r"C:\Program Files\QGIS 3.40.0\bin\python-qgis-ltr.bat"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    raise RuntimeError(
        "Set QGIS_PYTHON to your QGIS Python launcher. QGIS/GDAL is required "
        "for Landsat and vector-grid preprocessing."
    )


def project_python() -> Path | None:
    """Return the project's isolated interpreter when it exists."""
    if os.name == "nt":
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def ensure_project_runtime() -> None:
    """Relaunch the CLI in .venv so shell Python versions cannot bypass its packages."""
    interpreter = project_python()
    if interpreter is None:
        return
    if Path(sys.executable).resolve() == interpreter.resolve():
        return
    print(f"Using project environment: {interpreter}", flush=True)
    result = subprocess.run(
        [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=ROOT,
        check=False,
    )
    raise SystemExit(result.returncode)


def run(script: str, *, qgis: bool = False) -> None:
    # QGIS-dependent scripts need QGIS's bundled Python/GDAL bindings. All
    # other scripts inherit this CLI's .venv interpreter after startup.
    interpreter = qgis_python() if qgis else sys.executable
    command = [interpreter, str(ROOT / script)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)



def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_quality(path: Path) -> tuple[int, int, float, list[tuple[str, int]]]:
    """Return row count, data-cell count, blank count, and most-missing fields."""
    row_count = cell_count = blank_count = 0
    field_blanks: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        field_blanks = {name: 0 for name in fields}
        for row in reader:
            row_count += 1
            for name in fields:
                cell_count += 1
                if not (row.get(name) or "").strip():
                    blank_count += 1
                    field_blanks[name] += 1
    worst = sorted(((k, v) for k, v in field_blanks.items() if v), key=lambda p: -p[1])[:3]
    return row_count, cell_count, blank_count, worst


def grid_cell_count(gpkg: Path) -> int | None:
    if not gpkg.exists():
        return None
    try:
        with sqlite3.connect(gpkg) as conn:
            row = conn.execute(
                "SELECT table_name FROM gpkg_geometry_columns ORDER BY table_name LIMIT 1"
            ).fetchone()
            if not row:
                return None
            table = '"' + row[0].replace('"', '""') + '"'
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except (sqlite3.Error, OSError):
        return None


def print_phase_summary(phase: str) -> None:
    """Print run-specific summaries from completed project outputs."""
    root = ROOT
    try:
        if phase == "phase1":
            import yaml
            cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
            manifest = read_rows(root / "data/processed/Landsat/Landsat_scene_manifest.csv")
            period = cfg["study"]
            grid = cfg["spatial"]
            cells = grid_cell_count(root / grid["grid"])
            print("\n========== PHASE 1: DATA ENGINEERING ==========")
            print(f"City: {cfg['project']['city']}, {cfg['project']['state']}")
            print(f"Study period: {period['start_date']} to {period['end_date']}")
            print(f"Grid resolution: {grid['resolution_m']} m | CRS: {grid['crs']}")
            print(f"Landsat item scenes processed: {len(manifest)}")
            if manifest:
                year_season = {}
                seen_dates = set()
                for row in manifest:
                    day = row.get("date_utc", "")[:10]
                    season = row.get("season", "unknown")
                    year = row.get("year", day[:4])
                    if (day, season) not in seen_dates:
                        year_season[(year, season)] = year_season.get((year, season), 0) + 1
                        seen_dates.add((day, season))
                print("Distinct acquisition dates by year/season: " + ", ".join(
                    f"{year} {season}={count}" for (year, season), count in sorted(year_season.items())))
                print(f"Valid pixels per scene: {min(int(r['valid_pixels']) for r in manifest):,} to {max(int(r['valid_pixels']) for r in manifest):,}; mean {sum(int(r['valid_pixels']) for r in manifest)/len(manifest):,.0f}")
                print("Scene QA retention (valid / ST-populated): " + ", ".join(f"{r['date_utc'][:10]} {100*float(r['valid_fraction_of_populated']):.1f}%" for r in manifest))
                populated = sum(int(r['st_b10_populated_pixels']) for r in manifest)
                valid_total = sum(int(r['valid_pixels']) for r in manifest)
                excluded = populated - valid_total
                print(f"Landsat pixels excluded by QA/missing ST: {excluded:,}/{populated:,} ({100*excluded/populated:.2f}%) across scenes")
            print(f"Grid cells created: {cells:,}" if cells is not None else "Grid cells created: not available")
            print("Missing values by tabular dataset (blank data cells):")
            datasets = [
                ("ERA5 overpass matches", root / "data/processed/ERA5_Landsat_Overpass_Matched.csv"),
                ("CPCB overpass matches", root / "data/processed/CPCB_Landsat_Overpass_Matched.csv"),
                ("OSM urban-form grid", root / "data/processed/urban_form/Nagpur_urban_form_30m.csv"),
            ]
            for label, path in datasets:
                if not path.exists():
                    print(f"  {label}: unavailable")
                    continue
                nrows, ncells, blanks, worst = csv_quality(path)
                pct = 100 * blanks / ncells if ncells else 0.0
                detail = ", ".join(f"{name}={count:,}" for name, count in worst) or "none"
                print(f"  {label}: {blanks:,}/{ncells:,} ({pct:.2f}%); most missing: {detail}")
            print("Preprocessing status: completed" if cells is not None and len(manifest) and all((root / p).exists() for p in ["data/processed/ERA5_Landsat_Overpass_Matched.csv", "data/processed/CPCB_Landsat_Overpass_Matched.csv", "data/processed/urban_form/Nagpur_urban_form_30m.csv"]) else "Preprocessing status: outputs incomplete")

        elif phase == "phase2":
            scenes = read_rows(root / "outputs/tables/Nagpur_phase2_scene_metrics.csv")
            ranked = read_rows(root / "outputs/tables/Nagpur_grid_hotspot_priority_ranking.csv")
            manifest = read_rows(root / "data/processed/Landsat/Landsat_scene_manifest.csv")
            mean_scene_lst = sum(float(r['city_median_lst_c']) for r in scenes) / max(len(scenes), 1)
            max_scene_lst = max(float(r['lst_max_c']) for r in manifest)
            mean_anomaly = sum(float(r['mean_hot_LST_anomaly_C']) for r in ranked) / max(len(ranked), 1)
            indices = [float(r['hot_heat_stress_index_0_100']) for r in ranked]
            significant = sum(float(r['GiStar_FDR_q']) <= .05 and int(r['GiStar_hot_cluster_FDR05']) == 1 for r in ranked)
            persistent = sum(float(r['persistence_share']) >= .5 for r in ranked)
            print("\n========== PHASE 2: HOTSPOT IDENTIFICATION ==========")
            print(f"Mean city median LST across scenes: {mean_scene_lst:.2f} deg C | maximum scene-pixel LST: {max_scene_lst:.2f} deg C")
            print(f"Mean hot-season LST anomaly: {mean_anomaly:.2f} deg C")
            print(f"Heat-stress index (0 to 100): mean {sum(indices)/len(indices):.2f}, max {max(indices):.2f}")
            print(f"FDR-significant Gi* high-hotspot cells (q <= 0.05): {significant:,}")
            print(f"Persistent cells (hottest-decile share >= 0.50): {persistent:,}")
            print("Top 10 grid cells (rank | grid ID | ward | hot LST anomaly | stress | severity):")
            for r in ranked[:10]:
                print(f"  {int(r['priority_rank']):>2} | {r['grid_id']} | {r['ward']} | {float(r['mean_hot_LST_anomaly_C']):.2f} deg C | {float(r['hot_heat_stress_index_0_100']):.1f} | {r['hot_severity']}")
            print("Hotspot analysis completed.")

        elif phase == "phase3":
            print("\n========== PHASE 3: DRIVER ANALYSIS ==========")
            native_path = root / "outputs/tables/Nagpur_phase3_spatial_cv_metrics.json"
            if native_path.exists():
                report = json.loads(native_path.read_text(encoding="utf-8"))
                ranked = read_rows(root / "outputs/tables/Nagpur_phase3_native_ranked_drivers.csv")
                print(f"Spatial validation: {report['cv_splits']}-fold GroupKFold, {report['spatial_block_size_m']/1000:.0f} km blocks; tuning objective is pooled held-out RMSE.")
                for season, models in report['estimators'].items():
                    for model_name, result in models.items():
                        m = result['spatial_cv']
                        print(f"{season.title()} {model_name}: spatial-CV R2 {m['R2']:.3f}, RMSE {m['RMSE_C']:.2f} deg C, MAE {m['MAE_C']:.2f} deg C, bias {m['bias_C']:.2f} deg C")
                        top = sorted((r for r in ranked if r['season']==season and r['model']==model_name),key=lambda r:-float(r['mean_abs_shap_c']))[:5]
                        print("  Top native SHAP drivers: " + ", ".join(f"{r['driver']} ({float(r['mean_abs_shap_c']):.3f} deg C)" for r in top))
                print("Cross-season transfer checks:")
                for m in report['cross_season_transfer']:
                    print(f"  train {m['train_season']} / test {m['test_season']} | {m['model']}: R2 {m['R2']:.3f}, RMSE {m['RMSE_C']:.2f} deg C")
                print("Models: scikit-learn RandomForestRegressor, XGBoost XGBRegressor, and CatBoost CatBoostRegressor; explanations: native TreeExplainer SHAP.")
                print("Legacy custom CART results remain in Nagpur_phase3_model_metrics.json for comparison.")
            else:
                metrics = read_rows(root / "outputs/tables/Nagpur_phase3_ranked_drivers.csv")
                model_metrics = json.loads((root / "outputs/tables/Nagpur_phase3_model_metrics.json").read_text(encoding="utf-8"))
                print("Native spatial model results not found; showing legacy custom-CART random-split results.")
                for m in model_metrics:
                    season = m['season']; print(f"{season.title()}: OLS R2 {m['OLS_R2']:.3f}; custom forest R2 {m['RandomForest_R2']:.3f}; custom boosted CART R2 {m['GradientBoostedTrees_R2']:.3f}")
                    rows = sorted((r for r in metrics if r['season'] == season), key=lambda r: -float(r['forest_mean_abs_shap_c']))
                    print("  Top legacy approximate drivers: " + ", ".join(r['driver'] for r in rows[:5]))
            year_path = root / "outputs/tables/Nagpur_phase3_year_holdout_metrics.json"
            if year_path.exists():
                year_report = json.loads(year_path.read_text(encoding="utf-8"))
                print("Year-held-out comparison (entire acquisition years held out; pooled out-of-fold predictions):")
                for season, models in year_report["pooled_out_of_fold_metrics"].items():
                    for model_name, metric in models.items():
                        print(f"  {season.title()} {model_name}: R2 {metric['R2']:.3f}, RMSE {metric['RMSE_C']:.2f} deg C, MAE {metric['MAE_C']:.2f} deg C (n={metric['n']:,})")
                print("Year-held-out models: Random Forest, XGBoost, CatBoost; per-scene table retains date, year, season, and scene ID.")
            print("Native and legacy figures are saved under outputs/figures/.")
            print("Driver analysis completed.")

        elif phase == "phase4":
            metrics = json.loads((root / "outputs/tables/Nagpur_phase4_metrics.json").read_text(encoding="utf-8"))
            print("\n========== PHASE 4: PHYSICS-INFORMED HEAT MODEL ==========")
            sample_n = metrics['spatial_block_cv'][0].get('n', 'unknown') if metrics.get('spatial_block_cv') else 'unknown'
            print(f"Spatial-block cross-validation ({sample_n:,} sampled scene-cells):" if isinstance(sample_n, int) else f"Spatial-block cross-validation ({sample_n} sampled scene-cells):")
            for row in metrics['spatial_block_cv']:
                print(f"  {row['model']}: R2 {row['R2']:.3f}, RMSE {row['RMSE_C']:.2f} deg C, MAE {row['MAE_C']:.2f} deg C, bias {row['bias_C']:.2f} deg C, residual Moran's I {row['residual_Morans_I_queen_3x3']:.3f}")
            print("Temporal holdouts:")
            for row in metrics['temporal_holdout']:
                print(f"  {row['holdout']} | {row['model']}: R2 {row['R2']:.3f}, RMSE {row['RMSE_C']:.2f} deg C, MAE {row['MAE_C']:.2f} deg C")
            print("Physical plausibility perturbations:")
            for row in metrics['physical_perturbation']:
                for prefix, label in [('pure_ml', 'Pure ML'), ('physics_model', 'Physics-informed')]:
                    delta = row[f'{prefix}_mean_delta_c']
                    share = row[f'{prefix}_share_nonincreasing']
                    direction = "decreased" if delta < 0 else "increased"
                    passed = share >= .999
                    print(f"  {label}, +{row['amount']} {row['perturbation']}: mean LST {direction} by {abs(delta):.3f} deg C; monotonic in {100*share:.1f}% ({'PASS' if passed else 'FAIL/CHECK'})")
            print("Note: temporal transfer is weak; CPCB comparison is contextual, not direct LST validation.")
            print("Model validation outputs completed.")

        elif phase == "phase5":
            rows = read_rows(root / "outputs/tables/Nagpur_phase5_scenario_comparison.csv")
            print("\n========== PHASE 5: COOLING SCENARIOS ==========")
            print("Intervention | mean cooling on area >=0.1 deg C | area affected (km2) | GHSL population benefiting | max cooling (deg C)")
            for r in rows:
                print(f"{r['scenario']} | {float(r['mean_cooling_on_benefiting_area_C']):.2f} deg C | {float(r['area_cooling_at_least_0_1C_km2']):.2f} | {float(r['population_benefiting_at_least_0_1C_GHSL2020']):,.0f} | {float(r['maximum_cooling_C']):.2f}")
            print("Mean cooling above is computed over cells with predicted cooling >=0.1 deg C; comfort is a sensitivity proxy, not validated UTCI/WBGT.")
            print("Scenario simulation completed.")

        elif phase == "phase6":
            summary = json.loads((root / "outputs/tables/Nagpur_phase6_optimization_summary.json").read_text(encoding="utf-8"))
            alloc = read_rows(root / "outputs/tables/Nagpur_phase6_ward_allocation.csv")
            cells = read_rows(root / "outputs/tables/Nagpur_phase6_recommended_cells.csv")
            rec = summary['recommended_budget_max_cooling_plan']
            print("\n========== PHASE 6: OPTIMIZATION ==========")
            print(f"Optimization method: greedy marginal-benefit baseline + custom NSGA-II; Pareto portfolios: {summary['pareto_solutions']}")
            print(f"Budget: {summary['budget_relative_units']:.0f} relative cost units (not INR; unit costs are assumed)")
            greedy = summary['greedy_baseline']
            print(f"Greedy: {greedy['selected_cells']} cells, cost {greedy['cost_relative_units']:.1f}, exposure-weighted cooling {greedy['exposure_weighted_cooling_person_C']:,.0f} person-deg C")
            print(f"Selected Pareto plan: {rec['selected_cells']} cells, cost {rec['cost_relative_units']:.1f}, mean direct-cell cooling {rec['mean_direct_cooling_C']:.2f} deg C")
            print(f"Estimated population on treated cells (GHSL 2020): {rec['estimated_population_on_treated_cells_GHSL2020']:,.0f}; wards reached: {rec['wards_reached']}")
            counts = {name: 0 for name in ("urban_greening_tree_cover", "green_roofs", "cool_roofs_pavements", "water_bodies")}
            for row in cells:
                counts[row['intervention']] = counts.get(row['intervention'], 0) + 1
            print("Intervention counts: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
            print("Leading wards: " + ", ".join(f"{ward} ({n} cells)" for ward, n in sorted({k:sum(int(x['selected_cells']) for x in alloc if x['ward']==k) for k in {x['ward'] for x in alloc}}.items(), key=lambda kv: -kv[1])[:5]))
            print("First selected grid IDs: " + ", ".join(r['grid_id'] for r in cells[:10]))
            lo, hi = rec['cooling_sensitivity_range_mean_C']
            print(f"Mean cooling sensitivity range: {lo:.2f} to {hi:.2f} deg C (one spatial-CV RMSE, not a confidence interval)")
            print("Uncertainty: temporal validation is poor; costs are relative assumptions, not procurement estimates.")
            print("Pareto front and placement files saved under outputs/tables/ and outputs/maps/.")
    except (OSError, ValueError, KeyError, csv.Error, json.JSONDecodeError, TypeError) as exc:
        print(f"\n[Phase summary unavailable: {exc}]")

def main() -> None:
    ensure_project_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", choices=("aoi", "wards", "prepare_grid", "era5", "landsat", "osm", "cpcb", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6"),
        help="Run a data acquisition step, Phase 1 pipeline, or Phases 2 through 6 analysis.",
    )
    args = parser.parse_args()
    steps = {
        "aoi": lambda: run("src/data_collection/download_nagpur_aoi.py"),
        "wards": lambda: run("src/data_collection/download_nagpur_wards.py"),
        "prepare_grid": lambda: run("src/preprocessing/prepare_nagpur_grid.py", qgis=True),
        "era5": lambda: run("src/data_collection/download_era5.py"),
        "landsat": lambda: run("src/data_collection/collect_landsat_seasons.py", qgis=True),
        "osm": lambda: (
            run("src/data_collection/download_osm.py", qgis=True),
            run("src/preprocessing/build_urban_form.py", qgis=True),
        ),
        "cpcb": lambda: (
            run("src/data_collection/download_cpcb.py", qgis=True),
            run("src/preprocessing/match_cpcb_overpasses.py", qgis=True),
        ),
        "phase2": lambda: (
            run("src/data_collection/download_population.py", qgis=True),
            run("src/hotspot_analysis/phase2_hotspots.py", qgis=True),
        ),
        "phase3": lambda: (
            run("src/driver_analysis/phase3_drivers.py", qgis=True),
            run("src/driver_analysis/phase3_per_scene_predictors.py", qgis=True),
            run("src/driver_analysis/phase3_figures.py", qgis=True),
            run("src/driver_analysis/phase3_established_models.py"),
            run("src/driver_analysis/phase3_year_holdout_models.py"),
        ),
        "phase4": lambda: run("src/physics_model/phase4_physics_model.py", qgis=True),
        "phase5": lambda: run("src/scenario_simulation/phase5_cooling_scenarios.py", qgis=True),
        "phase6": lambda: run("src/optimization/phase6_optimize.py", qgis=True),
    }
    if args.step == "phase1":
        steps["landsat"]()
        steps["era5"]()
        steps["osm"]()
        steps["cpcb"]()
        print_phase_summary("phase1")
    else:
        steps[args.step]()
        if args.step.startswith("phase"):
            print_phase_summary(args.step)


if __name__ == "__main__":
    main()
