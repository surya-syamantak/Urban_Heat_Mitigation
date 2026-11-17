"""Compare existing heat-mitigation outputs from Ahmedabad, Chennai, and Nagpur.

Run from any working directory with:
    python compare_projects.py

Input projects are read only. Comparison tables, charts, and a markdown summary
are written under this folder's outputs/ directory.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def ensure_plotting_runtime() -> None:
    """Relaunch under a working project environment if Matplotlib is broken."""
    current = Path(sys.executable).resolve()
    try:
        import matplotlib as _matplotlib
        import yaml as _yaml  # noqa: F401

        _matplotlib.use("Agg")
        import matplotlib.pyplot as _pyplot  # noqa: F401
        return
    except Exception as exc:
        candidates = [
            ROOT / "Urban_Heat_Mitigation_Chennai" / ".venv312" / "Scripts" / "python.exe",
            ROOT / "Urban_Heat_Mitigation_Nagpur" / ".venv" / "Scripts" / "python.exe",
        ]
        for interpreter in candidates:
            if interpreter.is_file() and interpreter.resolve() != current:
                print(
                    f"Matplotlib is unavailable in Python {sys.version_info.major}.{sys.version_info.minor} "
                    f"({type(exc).__name__}: {exc}); restarting with {interpreter}",
                    flush=True,
                )
                result = subprocess.run(
                    [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]],
                    cwd=HERE,
                    check=False,
                )
                raise SystemExit(result.returncode)
        raise RuntimeError(
            "Matplotlib or PyYAML could not be imported, and no alternate project Python environment was found. "
            "Install this folder's requirements.txt into Python 3.12 or run run_comparison.ps1."
        ) from exc


ensure_plotting_runtime()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


OUTPUT = HERE / "outputs"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
REPORTS = OUTPUT / "reports"

PROJECTS = [
    {"city": "Ahmedabad", "folder": "Urban_Heat_Mitigation", "prefix": "Ahmedabad"},
    {"city": "Chennai", "folder": "Urban_Heat_Mitigation_Chennai", "prefix": "Chennai"},
    {"city": "Nagpur", "folder": "Urban_Heat_Mitigation_Nagpur", "prefix": "Nagpur"},
]
CITY_COLORS = {"Ahmedabad": "#d95f02", "Chennai": "#1b9e77", "Nagpur": "#3973ac"}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: Path) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = TABLES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return sum(valid) / len(valid) if valid else math.nan


def get_project_context(project: dict[str, str]) -> dict[str, Any]:
    root = ROOT / project["folder"]
    prefix = project["prefix"]
    config_path = root / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    p2 = load_json(root / "outputs" / "reports" / f"{prefix}_phase2_metadata.json") or {}
    p3 = load_json(root / "outputs" / "tables" / f"{prefix}_phase3_spatial_cv_metrics.json") or {}
    p6 = load_json(root / "outputs" / "tables" / f"{prefix}_phase6_optimization_summary.json") or {}
    p2_scene_rows = load_csv(root / "outputs" / "tables" / f"{prefix}_phase2_scene_metrics.csv") or []
    manifest = load_csv(root / "data" / "processed" / "Landsat" / "Landsat_scene_manifest.csv") or []
    era5_meta = load_json(root / "data" / "raw" / "ERA5" / f"{project['city']}_ERA5_request_metadata.json") or {}
    cpcb_matches = load_csv(root / "data" / "processed" / "CPCB_Landsat_Overpass_Matched.csv") or []
    unique_dates = {str(row.get("date_utc", ""))[:10] for row in manifest if row.get("date_utc")}
    context = {
        **project,
        "root": root,
        "config": config,
        "phase2": p2,
        "phase3": p3,
        "phase6": p6,
        "manifest_rows": manifest,
        "landsat_item_scenes": len(manifest),
        "landsat_unique_dates": len(unique_dates),
        "era5_record_count": era5_meta.get("record_count", ""),
        "cpcb_match_count": len(cpcb_matches),
        "phase2_scenes_with_cpcb_support": sum(number(row.get("paired_station_count"), 0.0) > 0 for row in p2_scene_rows),
        "cell_size_m": config.get("spatial", {}).get("resolution_m"),
        "study_period": " to ".join(p2.get("study_period", [])),
        "grid_cells": p2.get("grid_cells", ""),
        "wards": p2.get("wards", ""),
        "population_source": p2.get("population_source", ""),
        "population_epoch": p2.get("population_epoch", ""),
        "population_tile": p2.get("population_tile", ""),
        "heat_index_method": p2.get("heat_index", ""),
        "heat_index_sources": ("; ".join(p2.get("heat_index_sources", [])) if isinstance(p2.get("heat_index_sources", []), list) else str(p2.get("heat_index_sources", ""))) or "; ".join(sorted({row.get("heat_index_source") or row.get("interpolation", "") for row in p2_scene_rows if row.get("heat_index_source") or row.get("interpolation")})),
        "heat_index_interpretation": p2.get("heat_index_interpretation", ""),
    }
    context["seasons"] = config.get("study", {}).get("seasons", {})
    return context


def output_path(context: dict[str, Any], subdir: str, filename: str) -> Path:
    return context["root"] / "outputs" / subdir / f"{context['prefix']}_{filename}"


def availability(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = {
        "landsat_scene_manifest": ("data/processed/Landsat", "Landsat_scene_manifest.csv"),
        "dataset_catalogue": ("data", "dataset_audit.csv"),
        "era5_request_metadata": ("data/raw/ERA5", "ERA5_request_metadata.json"),
        "era5_overpass_matches": ("data/processed", "ERA5_Landsat_Overpass_Matched.csv"),
        "cpcb_overpass_matches": ("data/processed", "CPCB_Landsat_Overpass_Matched.csv"),
        "phase2_scene_metrics": ("tables", "phase2_scene_metrics.csv"),
        "phase2_metadata": ("reports", "phase2_metadata.json"),
        "phase2_grid_hotspot_ranking": ("tables", "grid_hotspot_priority_ranking.csv"),
        "phase3_native_metrics": ("tables", "phase3_spatial_cv_metrics.json"),
        "phase3_native_drivers": ("tables", "phase3_native_ranked_drivers.csv"),
        "phase4_metrics": ("tables", "phase4_metrics.json"),
        "phase5_scenarios": ("tables", "phase5_scenario_comparison.csv"),
        "phase6_summary": ("tables", "phase6_optimization_summary.json"),
        "phase6_ward_allocation": ("tables", "phase6_ward_allocation.csv"),
        "ward_heat_priority": ("tables", "ward_hotspot_priority_ranking.csv"),
    }
    rows = []
    for context in contexts:
        row: dict[str, Any] = {"city": context["city"], "project_folder": context["folder"]}
        for label, (subdir, suffix) in checks.items():
            if label == "era5_request_metadata":
                path = context["root"] / "data" / "raw" / "ERA5" / f"{context['city']}_ERA5_request_metadata.json"
            else:
                path = context["root"] / subdir / suffix if subdir.startswith("data") else output_path(context, subdir, suffix)
            row[label] = path.is_file()
        row["all_comparison_inputs_present"] = all(row[k] for k in checks)
        rows.append(row)
    return rows


def landsat_sampling_comparison(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for context in contexts:
        groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"dates": set(), "item_records": 0})
        for row in context["manifest_rows"]:
            raw_date = str(row.get("date_utc", ""))[:10]
            if not raw_date:
                continue
            key = (str(row.get("year") or raw_date[:4]), str(row.get("season") or "unclassified"))
            groups[key]["dates"].add(raw_date)
            groups[key]["item_records"] += 1
        for (year, season), values in sorted(groups.items()):
            result.append({
                "city": context["city"],
                "year": year,
                "season": season,
                "distinct_acquisition_dates": len(values["dates"]),
                "landsat_item_or_tile_records": values["item_records"],
            })
    return result


def phase2_comparison(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for context in contexts:
        rows = load_csv(output_path(context, "tables", "phase2_scene_metrics.csv")) or []
        by_season_date: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        manifest_season = {
            str(row.get("date_utc", ""))[:10]: row.get("season", "unclassified")
            for row in context["manifest_rows"] if row.get("date_utc")
        }
        for row in rows:
            raw_date = (row.get("scene_date_utc") or "")[:10]
            try:
                observed = date.fromisoformat(raw_date)
            except ValueError:
                continue
            season = manifest_season.get(raw_date, "")
            if not season:
                for name, bounds in context["seasons"].items():
                    months = bounds.get("months", []) if isinstance(bounds, dict) else []
                    if observed.month in months:
                        season = name
                        break
                    try:
                        start = date.fromisoformat(str(bounds["start_date"]))
                        end = date.fromisoformat(str(bounds["end_date"]))
                    except (KeyError, ValueError, TypeError):
                        continue
                    if start <= observed <= end:
                        season = name
                        break
            season = season or "unclassified"
            by_season_date[(season, raw_date)].append(row)

        season_dates: dict[str, list[dict[str, float]]] = defaultdict(list)
        for (season, observed_date), day_rows in by_season_date.items():
            season_dates[season].append({
                "city_median_lst_c": mean([number(r.get("city_median_lst_c")) for r in day_rows]),
                "city_p90_lst_c": mean([number(r.get("city_p90_lst_c")) for r in day_rows]),
                "valid_lst_cells_per_scene_mean": mean([number(r.get("valid_lst_cells")) for r in day_rows]),
                "hottest_decile_cells_per_scene_mean": mean([number(r.get("hottest_decile_cells")) for r in day_rows]),
                "scene_rows": float(len(day_rows)),
            })
        for season in ("hot", "cool", "unclassified"):
            dates = season_dates.get(season, [])
            if not dates:
                continue
            result.append({
                "city": context["city"],
                "season": season,
                "scene_dates": len(dates),
                "scene_rows": int(sum(row["scene_rows"] for row in dates)),
                "mean_date_city_median_lst_c": mean([row["city_median_lst_c"] for row in dates]),
                "mean_date_city_p90_lst_c": mean([row["city_p90_lst_c"] for row in dates]),
                "mean_scene_valid_lst_cells": mean([row["valid_lst_cells_per_scene_mean"] for row in dates]),
                "mean_scene_hottest_decile_cells": mean([row["hottest_decile_cells_per_scene_mean"] for row in dates]),
                "population_epoch": context["population_epoch"],
                "heat_index_method": context["heat_index_method"],
                "heat_index_sources": context["heat_index_sources"],
                "heat_index_context": context["heat_index_interpretation"],
            })
    return result


def phase3_comparison(contexts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validation: list[dict[str, Any]] = []
    transfer: list[dict[str, Any]] = []
    top_drivers: list[dict[str, Any]] = []
    for context in contexts:
        metrics = context["phase3"]
        for season, models in metrics.get("estimators", {}).items():
            for model, detail in models.items():
                score = detail.get("spatial_cv", {})
                validation.append({
                    "city": context["city"], "season": season, "model": model,
                    "n": score.get("n", ""), "R2": score.get("R2", ""),
                    "RMSE_C": score.get("RMSE_C", ""), "MAE_C": score.get("MAE_C", ""),
                    "bias_C": score.get("bias_C", ""),
                    "cv_folds": metrics.get("cv_splits", ""),
                    "block_size_m": metrics.get("spatial_block_size_m", ""),
                    "best_params": json.dumps(detail.get("best_params", {}), sort_keys=True),
                })
        for row in metrics.get("cross_season_transfer", []):
            transfer.append({"city": context["city"], **row})

        driver_rows = load_csv(output_path(context, "tables", "phase3_native_ranked_drivers.csv")) or []
        for season in ("hot", "cool"):
            for model in sorted({r.get("model", "") for r in driver_rows if r.get("season") == season}):
                chosen = [r for r in driver_rows if r.get("season") == season and r.get("model") == model]
                chosen.sort(key=lambda r: number(r.get("mean_abs_shap_c"), -math.inf), reverse=True)
                for rank, row in enumerate(chosen[:10], start=1):
                    top_drivers.append({
                        "city": context["city"], "season": season, "model": model,
                        "rank": rank, "driver": row.get("driver", ""),
                        "mean_abs_shap_c": number(row.get("mean_abs_shap_c")),
                        "mean_shap_c": number(row.get("mean_shap_c")),
                        "share_of_total_percent": number(row.get("share_of_total_percent")),
                    })
    return validation, transfer, top_drivers


def phase4_comparison(contexts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation: list[dict[str, Any]] = []
    perturbations: list[dict[str, Any]] = []
    for context in contexts:
        metrics = load_json(output_path(context, "tables", "phase4_metrics.json")) or {}
        for section in ("spatial_block_cv", "temporal_holdout"):
            for row in metrics.get(section, []):
                validation.append({"city": context["city"], "validation_type": section, **row})
        for row in metrics.get("physical_perturbation", []):
            perturbations.append({"city": context["city"], **row})
    return validation, perturbations


def phase5_comparison(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for context in contexts:
        for row in load_csv(output_path(context, "tables", "phase5_scenario_comparison.csv")) or []:
            converted: dict[str, Any] = {"city": context["city"]}
            for key, value in row.items():
                converted[key] = number(value) if key not in ("scenario", "eligibility_and_input_change") else value
            result.append(converted)
    return result


def phase6_comparison(contexts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    equity_summary: list[dict[str, Any]] = []
    ward_detail: list[dict[str, Any]] = []
    for context in contexts:
        summary = context["phase6"]
        plan = summary.get("recommended_budget_max_cooling_plan", {})
        greedy = summary.get("greedy_baseline", {})
        summary_rows.append({
            "city": context["city"],
            "budget_relative_units": summary.get("budget_relative_units", ""),
            "candidate_pool_size": summary.get("candidate_pool_size", ""),
            "selected_cells": plan.get("selected_cells", ""),
            "cost_relative_units": plan.get("cost_relative_units", ""),
            "exposure_weighted_cooling_person_C": plan.get("exposure_weighted_cooling_person_C", ""),
            "population_on_treated_cells_GHSL2020": plan.get("estimated_population_on_treated_cells_GHSL2020", ""),
            "mean_direct_cooling_C": plan.get("mean_direct_cooling_C", ""),
            "wards_reached": plan.get("wards_reached", ""),
            "greedy_selected_cells": greedy.get("selected_cells", ""),
            "greedy_exposure_weighted_cooling_person_C": greedy.get("exposure_weighted_cooling_person_C", ""),
            "selection_rule": plan.get("selection_rule", ""),
            "uncertainty_note": summary.get("uncertainty_note", ""),
        })
        allocation_rows = load_csv(output_path(context, "tables", "phase6_ward_allocation.csv")) or []
        for row in allocation_rows:
            allocations.append({"city": context["city"], **row})
        ranking = load_csv(output_path(context, "tables", "ward_hotspot_priority_ranking.csv")) or []
        rank_key = lambda r: (r.get("ward", "").strip().casefold())
        alloc_by_ward: dict[str, dict[str, float]] = defaultdict(lambda: {"selected_cells": 0.0, "cost_relative_units": 0.0, "exposure_weighted_benefit_person_C": 0.0})
        for row in allocation_rows:
            target = alloc_by_ward[rank_key(row)]
            for field in target:
                target[field] += number(row.get(field), 0.0)
        ward_values = []
        for row in ranking:
            heat_pop = number(row.get("estimated_population_in_high_or_very_high_cells_2020"), 0.0)
            ward_values.append((heat_pop, row))
        ward_values.sort(key=lambda pair: pair[0], reverse=True)
        top_count = max(1, math.ceil(len(ward_values) * 0.20)) if ward_values else 0
        priority_keys = {rank_key(row) for _, row in ward_values[:top_count]}
        total_heat_pop = sum(max(0.0, value) for value, _ in ward_values)
        top_heat_pop = sum(max(0.0, value) for value, row in ward_values[:top_count])
        total_alloc_benefit = sum(v["exposure_weighted_benefit_person_C"] for v in alloc_by_ward.values())
        top_alloc_benefit = sum(v["exposure_weighted_benefit_person_C"] for key, v in alloc_by_ward.items() if key in priority_keys)
        total_cells = sum(v["selected_cells"] for v in alloc_by_ward.values())
        top_cells = sum(v["selected_cells"] for key, v in alloc_by_ward.items() if key in priority_keys)
        equity_summary.append({
            "city": context["city"], "wards_in_heat_ranking": len(ward_values),
            "priority_group_definition": "Top 20% of wards by population in high/very-high heat cells",
            "priority_wards": top_count,
            "priority_group_share_of_high_heat_population": top_heat_pop / total_heat_pop if total_heat_pop else math.nan,
            "priority_group_share_of_optimized_benefit": top_alloc_benefit / total_alloc_benefit if total_alloc_benefit else math.nan,
            "priority_group_share_of_selected_cells": top_cells / total_cells if total_cells else math.nan,
            "optimized_benefit_total_person_C": total_alloc_benefit,
            "selected_cells_total": int(total_cells),
        })
        for _, row in ward_values:
            a = alloc_by_ward[rank_key(row)]
            equity_row = {"city": context["city"], **row, **a}
            equity_row["top_20_percent_heat_population_priority"] = rank_key(row) in priority_keys
            ward_detail.append(equity_row)
    return summary_rows, allocations, equity_summary, ward_detail


def rank_rows(rows: list[dict[str, Any]], value_key: str, *, descending: bool = True) -> list[dict[str, Any]]:
    """Rank numeric rows, assigning equal values equal competition ranks."""
    valid = [dict(row) for row in rows if math.isfinite(number(row.get(value_key)))]
    valid.sort(key=lambda row: number(row[value_key]), reverse=descending)
    ranked: list[dict[str, Any]] = []
    previous = math.nan
    current_rank = 0
    for position, row in enumerate(valid, start=1):
        value = number(row[value_key])
        tolerance = 1e-9 * max(1.0, abs(value), abs(previous) if math.isfinite(previous) else 0.0)
        if not math.isfinite(previous) or abs(value - previous) > tolerance:
            current_rank = position
        row["rank"] = current_rank
        ranked.append(row)
        previous = value
    return ranked


def hotspot_area_comparison(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sum FDR-significant Gi* high-hotspot cells' areas for each city."""
    rows: list[dict[str, Any]] = []
    for context in contexts:
        source = load_csv(output_path(context, "tables", "grid_hotspot_priority_ranking.csv")) or []
        significant = [
            row for row in source
            if number(row.get("GiStar_FDR_q")) <= 0.05
            and number(row.get("GiStar_hot_cluster_FDR05"), 0.0) == 1
        ]
        area = sum(max(0.0, number(row.get("cell_area_km2"), 0.0)) for row in significant)
        rows.append({
            "city": context["city"],
            "significant_hotspot_area_km2": area if source else math.nan,
            "significant_hotspot_cells": len(significant) if source else "",
            "definition": "Sum of grid-cell area where Gi* hot-cluster flag=1 and FDR q<=0.05",
        })
    return rank_rows(rows, "significant_hotspot_area_km2")


def best_phase3_models(validation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose the lowest spatial-CV RMSE model separately for each city/season."""
    winners: list[dict[str, Any]] = []
    cities = [project["city"] for project in PROJECTS]
    for city in cities:
        for season in ("hot", "cool"):
            candidates = [
                row for row in validation
                if row.get("city") == city and str(row.get("season", "")).lower() == season
                and math.isfinite(number(row.get("RMSE_C")))
            ]
            if not candidates:
                continue
            best_rmse = min(number(row["RMSE_C"]) for row in candidates)
            tied = [row for row in candidates if abs(number(row["RMSE_C"]) - best_rmse) <= 1e-9]
            for row in tied:
                winners.append({
                    "city": city, "season": season, "best_model": row["model"],
                    "spatial_cv_RMSE_C": number(row["RMSE_C"]),
                    "spatial_cv_R2": number(row.get("R2")),
                    "selection_metric": "lowest spatial-CV RMSE within city and season",
                })
    return winners


def print_terminal_rankings(budget_rows: list[dict[str, Any]],
                            spend_rows: list[dict[str, Any]],
                            hotspot_rows: list[dict[str, Any]],
                            model_winners: list[dict[str, Any]]) -> None:
    print("\n========== CITY RANKINGS ==========")
    print("Budget limit rank (relative budget units; higher first):")
    if budget_rows:
        for row in budget_rows:
            print(f"  {row['rank']}. {row['city']}: {number(row['budget_relative_units']):,.0f} units")
    else:
        print("  Budget data unavailable.")
    print("Budget used rank (selected-plan cost in relative units; higher first):")
    if spend_rows:
        for row in spend_rows:
            cap = number(row.get("budget_relative_units"))
            spent = number(row.get("cost_relative_units"))
            use = f" ({100 * spent / cap:.1f}% of limit)" if cap > 0 and math.isfinite(spent) else ""
            print(f"  {row['rank']}. {row['city']}: {spent:,.1f} units{use}")
    else:
        print("  Spending data unavailable.")
    print("FDR-significant hotspot area rank (Gi* q <= 0.05; larger first):")
    available_hotspots = [row for row in hotspot_rows if math.isfinite(number(row.get("significant_hotspot_area_km2")))]
    if available_hotspots:
        for row in available_hotspots:
            print(f"  {row['rank']}. {row['city']}: {number(row['significant_hotspot_area_km2']):,.2f} km^2 ({row['significant_hotspot_cells']:,} cells)")
    else:
        print("  Hotspot-area data unavailable.")
    print("Best Phase 3 model by city and season (lowest spatial-CV RMSE):")
    if model_winners:
        for city in (project["city"] for project in PROJECTS):
            city_rows = [row for row in model_winners if row["city"] == city]
            if not city_rows:
                print(f"  {city}: model metrics unavailable.")
                continue
            results = []
            for season in ("hot", "cool"):
                rows = [row for row in city_rows if row["season"] == season]
                if rows:
                    models = "/".join(row["best_model"] for row in rows)
                    results.append(f"{season}={models} (RMSE {rows[0]['spatial_cv_RMSE_C']:.2f} deg C, R2 {rows[0]['spatial_cv_R2']:.3f})")
            print(f"  {city}: " + "; ".join(results))
    else:
        print("  Model metrics unavailable.")


def plot_phase2(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, season in zip(axes, ("hot", "cool")):
        selected = [r for r in rows if r["season"] == season]
        cities = [r["city"] for r in selected]
        x = list(range(len(cities)))
        width = .36
        ax.bar([v-width/2 for v in x], [number(r["mean_date_city_median_lst_c"]) for r in selected],
               width=width, label="Median LST", color=[CITY_COLORS[c] for c in cities])
        ax.bar([v+width/2 for v in x], [number(r["mean_date_city_p90_lst_c"]) for r in selected],
               width=width, label="P90 LST", color=[CITY_COLORS[c] for c in cities], alpha=.52)
        ax.set_xticks(x, cities)
        ax.set_title(f"{season.title()} season")
        ax.set_ylabel("°C")
        ax.grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Phase 2 LST by season, 2021–2025 (per-date means; solid=median, translucent=P90)",
                 y=0.98, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(FIGURES / "phase2_seasonal_lst_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase3(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, season in zip(axes, ("hot", "cool")):
        subset = [r for r in rows if r["season"] == season]
        models = list(dict.fromkeys(r["model"] for r in subset))
        cities = [p["city"] for p in PROJECTS]
        width = .8 / max(1, len(models))
        for index, model in enumerate(models):
            values = []
            for city in cities:
                values.append(next((number(r["RMSE_C"]) for r in subset if r["city"] == city and r["model"] == model), math.nan))
            positions = [i + (index - (len(models)-1)/2)*width for i in range(len(cities))]
            bars = ax.bar(positions, values, width=width, label=model)
            for bar, city in zip(bars, cities):
                bar.set_color(CITY_COLORS[city])
                bar.set_alpha(.55 if index else .95)
        ax.set_xticks(range(len(cities)), cities)
        ax.set_title(f"{season.title()} season")
        ax.set_ylabel("Spatial CV RMSE (°C; lower is better)")
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
    folds = sorted({str(r.get("cv_folds")) for r in rows if r.get("cv_folds") not in (None, "")})
    blocks = sorted({number(r.get("block_size_m")) for r in rows if number(r.get("block_size_m")) > 0})
    validation_label = f"{','.join(folds)}-fold" if folds else "spatial"
    block_label = f"{blocks[0] / 1000:g} km blocks" if len(blocks) == 1 else "project-specific blocks"
    fig.suptitle(f"Phase 3 native-model spatial validation ({validation_label}, {block_label})")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase3_spatial_cv_rmse_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase5(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in rows:
        x = number(row.get("mean_cooling_on_benefiting_area_C"))
        y = number(row.get("population_benefiting_at_least_0_1C_GHSL2020"))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        ax.scatter(x, y, s=55, color=CITY_COLORS[row["city"]], alpha=.8)
        ax.annotate(f"{row['city']} · {str(row.get('scenario','')).replace('_',' ')}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Mean cooling on benefiting area (°C)")
    ax.set_ylabel("Estimated population benefiting ≥0.1°C (GHSL 2020)")
    ax.set_title("Phase 5 scenario comparison")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase5_scenario_cooling_population.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase6(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cities = [r["city"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [
        ("selected_cells", "Selected cells"),
        ("exposure_weighted_cooling_person_C", "Exposure-weighted benefit (person·°C)"),
        ("wards_reached", "Wards reached"),
    ]
    for ax, (field, title) in zip(axes, metrics):
        ax.bar(cities, [number(r.get(field)) for r in rows], color=[CITY_COLORS[c] for c in cities])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Phase 6 recommended plans at the recorded relative budget")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase6_optimization_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(contexts: list[dict[str, Any]], availability_rows: list[dict[str, Any]],
                 p2: list[dict[str, Any]], p3: list[dict[str, Any]], p5: list[dict[str, Any]],
                 p6: list[dict[str, Any]], equity: list[dict[str, Any]],
                 sampling: list[dict[str, Any]],
                 budget_ranks: list[dict[str, Any]], spend_ranks: list[dict[str, Any]],
                 hotspot_ranks: list[dict[str, Any]], model_winners: list[dict[str, Any]]) -> None:
    cpcb_coverage = "; ".join(
        f"{c['city']}: {c['cpcb_match_count']} match rows, {c['phase2_scenes_with_cpcb_support']} Phase 2 scenes with CPCB support"
        for c in contexts
    )
    lines = [
        "# Urban Heat Mitigation: Three-city comparison",
        "",
        "This report summarizes outputs already present in the three project folders. It does not rerun city analyses.",
        "",
        "## Projects and input availability",
        "",
        "| City | Project folder | Study period | Unique Landsat dates | Item/tile records | Grid cells | Wards | Cell size (m) | ERA5 hours | CPCB match rows | Phase 2 scenes with CPCB support |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in contexts:
        lines.append(f"| {c['city']} | `{c['folder']}` | {c['study_period']} | {c['landsat_unique_dates']} | {c['landsat_item_scenes']} | {c['grid_cells']} | {c['wards']} | {c['cell_size_m']} | {c['era5_record_count']} | {c['cpcb_match_count']} | {c['phase2_scenes_with_cpcb_support']} |")
    lines += ["", "The detailed artifact checklist is in `outputs/tables/input_availability.csv`.",
              "The scene manifest supplies each acquisition's season; the sampling table distinguishes distinct dates from satellite tile/item records.", "",
              "## Landsat sampling by year and season", "",
              "| City | Year | Season | Distinct acquisition dates | Landsat item/tile records |",
              "|---|---:|---|---:|---:|"]
    for row in sampling:
        lines.append(f"| {row['city']} | {row['year']} | {row['season']} | {row['distinct_acquisition_dates']} | {row['landsat_item_or_tile_records']} |")
    lines += ["", "## City rankings", "",
              "Budget caps are relative units rather than currency. Equal values receive the same rank. Budget use is ranked separately using selected-plan cost.", "",
              "### Budget limit", "", "| Rank | City | Budget limit (relative units) |", "|---:|---|---:|"]
    for row in budget_ranks:
        lines.append(f"| {row['rank']} | {row['city']} | {number(row.get('budget_relative_units')):,.0f} |")
    lines += ["", "### Budget used", "", "| Rank | City | Selected-plan cost (relative units) |", "|---:|---|---:|"]
    for row in spend_ranks:
        lines.append(f"| {row['rank']} | {row['city']} | {number(row.get('cost_relative_units')):,.1f} |")
    lines += ["", "### FDR-significant hot-hotspot area", "",
              "Area is the summed grid-cell area for Gi* high-hotspot cells with FDR q ≤ 0.05.", "",
              "| Rank | City | Area (km²) | Significant cells |", "|---:|---|---:|---:|"]
    for row in hotspot_ranks:
        area = number(row.get("significant_hotspot_area_km2"))
        if math.isfinite(area):
            lines.append(f"| {row['rank']} | {row['city']} | {area:,.2f} | {row['significant_hotspot_cells']} |")
    lines += ["", "### Best Phase 3 model by season", "",
              "Selected by lowest spatial-CV RMSE within each city and season.", "",
              "| City | Season | Best model | RMSE (°C) | R² |", "|---|---|---|---:|---:|"]
    for row in model_winners:
        lines.append(f"| {row['city']} | {row['season'].title()} | {row['best_model']} | {row['spatial_cv_RMSE_C']:.3f} | {row['spatial_cv_R2']:.3f} |")
    lines += ["", "## Recommended comparisons", "",
              "- **Seasonal heat:** compare Phase 2 hot/cool city median and 90th-percentile LST summaries. Scene dates are averaged first so multiple Landsat path/row images from one acquisition date do not count as separate dates.",
              "- **Model reliability and drivers:** compare Phase 3 spatial-CV RMSE/R², cross-season transfer metrics, and the top native-SHAP drivers by city, season, and model.",
              "- **Mitigation scenarios:** compare Phase 5 treated area, cooling, and GHSL 2020 population benefit for matching scenario names.",
              "- **Optimization and equity:** compare Phase 6 plans under the recorded relative budget and inspect how selected cells/benefits align with each city’s highest heat-exposure wards.",
              "", "## Interpretation limits", "",
              f"The current configs use the 2021–2025 study window and 30 m analysis grids. The city projects now select all Landsat 8/9 scenes in configured hot/cool months that pass the shared scene-cloud threshold and per-pixel QA; date counts may differ by year, season and city. Phase 2 summarizes available dates rather than synchronized same-day observations. Tile/item records can exceed unique acquisition dates when an AOI spans multiple Landsat path/row tiles. CPCB coverage varies by city: {cpcb_coverage}. Phase 2 uses ERA5 background fallback when CPCB station support is insufficient. Cities differ in climate, AOI size, ward structure, and valid-cell coverage, so seasonal means and relative heat-stress scores are not direct measures of city-to-city causation. Heat-stress quintiles are city-relative; use absolute LST summaries and population measures for cross-city levels.",
              "", "Phase 3 SHAP values describe model associations, not causal cooling effects. Cross-season transfer is a stress test, not independent multi-year validation. Phase 5 estimates depend on each project’s scenario inputs and valid coverage. Phase 6 costs are relative units rather than currency; compare plans only alongside budget, candidate-pool size, and the stated assumptions. Ward-level equity shares are calculated within each city because ward areas and populations are not equivalent across cities.",
              "", "## Generated outputs", "",
              "Comparison CSVs are in `outputs/tables/`; charts are in `outputs/figures/`. Missing source artifacts are recorded in the availability table and omitted from the relevant comparison rather than treated as zero.", ""]
    missing = [f"{r['city']}: missing {', '.join(k for k,v in r.items() if k not in ('city','project_folder','all_comparison_inputs_present') and not v)}" for r in availability_rows if not r["all_comparison_inputs_present"]]
    if missing:
        lines.extend(["", "### Missing inputs", "", *[f"- {line}" for line in missing]])
    (REPORTS / "COMPARATIVE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for folder in (TABLES, FIGURES, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    contexts = [get_project_context(p) for p in PROJECTS]
    available = availability(contexts)
    write_csv("input_availability.csv", available)

    context_rows = [{k: c.get(k, "") for k in ("city", "folder", "study_period", "landsat_item_scenes", "landsat_unique_dates", "era5_record_count", "cpcb_match_count", "phase2_scenes_with_cpcb_support", "grid_cells", "wards", "cell_size_m", "population_source", "population_epoch", "population_tile", "heat_index_method", "heat_index_sources", "heat_index_interpretation")} for c in contexts]
    write_csv("project_context.csv", context_rows)

    sampling = landsat_sampling_comparison(contexts)
    write_csv("landsat_sampling_summary.csv", sampling)
    p2 = phase2_comparison(contexts)
    p3_validation, p3_transfer, p3_drivers = phase3_comparison(contexts)
    p4_validation, p4_perturbations = phase4_comparison(contexts)
    p5 = phase5_comparison(contexts)
    p6, allocations, equity, ward_detail = phase6_comparison(contexts)
    budget_ranks = rank_rows(p6, "budget_relative_units")
    spend_ranks = rank_rows(p6, "cost_relative_units")
    hotspot_ranks = hotspot_area_comparison(contexts)
    model_winners = best_phase3_models(p3_validation)

    outputs = {
        "phase2_seasonal_heat_summary.csv": p2,
        "phase3_spatial_cv_metrics.csv": p3_validation,
        "phase3_cross_season_transfer.csv": p3_transfer,
        "phase3_top_native_shap_drivers.csv": p3_drivers,
        "phase4_validation_metrics.csv": p4_validation,
        "phase4_physical_perturbations.csv": p4_perturbations,
        "phase5_scenario_comparison.csv": p5,
        "phase6_optimization_summary.csv": p6,
        "phase6_budget_limit_ranking.csv": budget_ranks,
        "phase6_budget_used_ranking.csv": spend_ranks,
        "phase2_significant_hotspot_area_ranking.csv": hotspot_ranks,
        "phase3_best_models_by_city_season.csv": model_winners,
        "phase6_ward_allocations.csv": allocations,
        "phase6_ward_equity_summary.csv": equity,
        "ward_heat_investment_equity.csv": ward_detail,
    }
    for name, rows in outputs.items():
        write_csv(name, rows)

    plot_phase2(p2)
    plot_phase3(p3_validation)
    plot_phase5(p5)
    plot_phase6(p6)
    write_report(contexts, available, p2, p3_validation, p5, p6, equity, sampling,
                 budget_ranks, spend_ranks, hotspot_ranks, model_winners)
    print_terminal_rankings(budget_ranks, spend_ranks, hotspot_ranks, model_winners)
    print(f"Wrote comparison tables to {TABLES}")
    print(f"Wrote comparison charts to {FIGURES}")
    print(f"Wrote report to {REPORTS / 'COMPARATIVE_REPORT.md'}")
    if not all(row["all_comparison_inputs_present"] for row in available):
        print("Some source outputs were missing; see input_availability.csv.")


if __name__ == "__main__":
    main()
