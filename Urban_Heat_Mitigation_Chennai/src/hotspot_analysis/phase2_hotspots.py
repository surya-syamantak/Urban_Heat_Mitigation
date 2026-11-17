"""Build Phase 2 LST anomaly, comfort, exposure, cluster and ranking outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from osgeo import gdal, osr
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsVectorLayer,
)
from scipy.ndimage import convolve
from scipy.special import ndtr
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
PROCESSED = ROOT / "data/processed"
LANDSAT_DIR = PROCESSED / "Landsat"
PHASE2_DIR = PROCESSED / "phase2"
MAP_DIR = ROOT / "outputs/maps"
TABLE_DIR = ROOT / "outputs/tables"
REPORT_DIR = ROOT / "outputs/reports"
GRID_PATH = ROOT / CONFIG["spatial"]["grid"]
CENTROID_PATH = ROOT / CONFIG["spatial"]["centroid_ward_table"]
WARD_PATH = ROOT / CONFIG["spatial"]["wards"]
SCENE_MANIFEST = LANDSAT_DIR / "Landsat_scene_manifest.csv"
CPCB_MATCHES = PROCESSED / "CPCB_Landsat_Overpass_Matched.csv"
ERA5_MATCHES = PROCESSED / "ERA5_Landsat_Overpass_Matched.csv"
ERA5_REQUEST_METADATA = ROOT / "data/raw/ERA5/Chennai_ERA5_request_metadata.json"
POP_SOURCE = ROOT / CONFIG["population"]["raster_path"]

SEVERITY_NAMES = ["Very low", "Low", "Moderate", "High", "Very high"]
SEVERITY_COLORS = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]
NODATA_FLOAT = -9999.0


def heat_index_celsius(air_c: float, rh: float) -> float:
    """NWS heat index using Rothfusz regression and its low-HI formula."""
    t = air_c * 9.0 / 5.0 + 32.0
    rh = min(max(rh, 0.0), 100.0)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    simple = (simple + t) / 2.0
    if simple < 80.0:
        return (simple - 32.0) * 5.0 / 9.0
    hi = (
        -42.379 + 2.04901523 * t + 10.14333127 * rh
        - 0.22475541 * t * rh - 0.00683783 * t * t
        - 0.05481717 * rh * rh + 0.00122874 * t * t * rh
        + 0.00085282 * t * rh * rh - 0.00000199 * t * t * rh * rh
    )
    if rh < 13.0 and 80.0 <= t <= 112.0:
        hi -= ((13.0 - rh) / 4.0) * math.sqrt(max(0.0, (17.0 - abs(t - 95.0)) / 17.0))
    elif rh > 85.0 and 80.0 <= t <= 87.0:
        hi += ((rh - 85.0) / 10.0) * ((87.0 - t) / 5.0)
    hi = max(hi, t)
    return (hi - 32.0) * 5.0 / 9.0


def grid_features():
    grid = QgsVectorLayer(str(GRID_PATH), "analysis_grid", "ogr")
    centroids = QgsVectorLayer(str(CENTROID_PATH), "centroid_ward", "ogr")
    if not grid.isValid() or not centroids.isValid():
        raise RuntimeError("Could not open canonical Phase 1 grid or centroid/ward table")
    ward_by_id = {int(f["grid_id"]): str(f["Name"]) for f in centroids.getFeatures()}
    feats = list(grid.getFeatures())
    ids = np.asarray([int(f["grid_id"]) for f in feats], dtype=np.int64)
    rows = np.asarray([int(f["row_index"]) for f in feats], dtype=np.int32)
    cols = np.asarray([int(f["col_index"]) for f in feats], dtype=np.int32)
    xs = np.asarray([(float(f["left"]) + float(f["right"])) / 2 for f in feats], dtype=np.float64)
    ys = np.asarray([(float(f["top"]) + float(f["bottom"])) / 2 for f in feats], dtype=np.float64)
    ward_names = np.asarray([ward_by_id.get(int(grid_id), "") for grid_id in ids], dtype=object)
    if np.any(ward_names == ""):
        raise RuntimeError("Missing ward labels in the canonical centroid/ward lookup")
    return ids, rows, cols, xs, ys, ward_names


def read_scene_manifest() -> list[dict[str, str]]:
    with SCENE_MANIFEST.open(newline="", encoding="utf-8") as stream:
        scenes = list(csv.DictReader(stream))
    if not scenes:
        raise RuntimeError("Landsat manifest is empty; run Phase 1 Landsat collection first")
    # Keep the Phase 1 manifest intact for auditing, but omit negligible AOI
    # fragments from Phase 2. They cannot support stable per-scene statistics.
    too_small = [scene for scene in scenes if int(scene.get("valid_pixels", 0) or 0) < 100]
    if too_small:
        scenes = [scene for scene in scenes if int(scene.get("valid_pixels", 0) or 0) >= 100]
        print(f"Skipping {len(too_small)} scene item(s) with fewer than 100 valid AOI pixels: "
              + ", ".join(scene["item_id"] for scene in too_small))
    if not scenes:
        raise RuntimeError("No Landsat scenes have at least 100 valid AOI pixels")
    manifest_ids = {scene["item_id"] for scene in scenes}
    seasons = {scene["season"] for scene in scenes}
    missing_seasons = set(CONFIG["study"]["seasons"]) - seasons
    if missing_seasons:
        raise RuntimeError(
            "Landsat manifest has no valid scenes for season(s): "
            + ", ".join(sorted(missing_seasons))
        )
    print(f"Using {len(scenes)} manifest scene items across "
          f"{len({scene['date_utc'][:10] for scene in scenes})} distinct acquisition dates")
    return scenes


def load_lst(scenes: list[dict[str, str]], rows: np.ndarray, cols: np.ndarray):
    sampled = np.full((len(scenes), len(rows)), np.nan, dtype=np.float32)
    raster_meta = None
    for si, scene in enumerate(scenes):
        item_id = scene["item_id"]
        raster_path = LANDSAT_DIR / item_id / f"{item_id}_LST_Celsius.tif"
        ds = gdal.Open(str(raster_path))
        if ds is None:
            raise FileNotFoundError(raster_path)
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32, copy=False)
        nodata = band.GetNoDataValue()
        good = np.isfinite(array)
        if nodata is not None:
            good &= array != nodata
        good &= (array > -100.0) & (array < 100.0)
        array[~good] = np.nan
        sampled[si, :] = array[rows, cols]
        meta = (ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform(), ds.GetProjection())
        if raster_meta is None:
            raster_meta = meta
        elif meta != raster_meta:
            raise RuntimeError(f"Scene raster alignment mismatch in {raster_path}")
        ds = None
    return sampled, raster_meta


def load_population_density(raster_meta, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Warp 100 m population counts, then express them as people per km2."""
    if not POP_SOURCE.is_file():
        raise FileNotFoundError(f"Run download_population.py first: {POP_SOURCE}")
    width, height, geotransform, projection = raster_meta
    xmin, xres, _, ymax, _, yres = geotransform
    xmax = xmin + width * xres
    ymin = ymax + height * yres
    ds = gdal.Warp(
        "", str(POP_SOURCE), format="MEM", dstSRS="EPSG:32644",
        outputBounds=(xmin, ymin, xmax, ymax), width=width, height=height,
        resampleAlg="bilinear", srcNodata=float(CONFIG["population"]["nodata"]),
        dstNodata=NODATA_FLOAT, outputType=gdal.GDT_Float32, multithread=False,
    )
    if ds is None or ds.RasterXSize != width or ds.RasterYSize != height:
        raise RuntimeError("Could not warp population to the master Landsat grid")
    counts_30m = ds.GetRasterBand(1).ReadAsArray().astype(np.float32, copy=False)
    valid = np.isfinite(counts_30m) & (counts_30m != NODATA_FLOAT) & (counts_30m >= 0)
    density = np.full_like(counts_30m, NODATA_FLOAT, dtype=np.float32)
    # Source pixels are 100 m x 100 m with population in persons per cell.
    density[valid] = counts_30m[valid] * 100.0
    population_raster = PHASE2_DIR / "Chennai_GHSL2020_population_density_30m.tif"
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    out = gdal.GetDriverByName("GTiff").Create(
        str(population_raster), width, height, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"],
    )
    out.SetGeoTransform(geotransform)
    out.SetProjection(projection)
    out.GetRasterBand(1).SetNoDataValue(NODATA_FLOAT)
    out.GetRasterBand(1).WriteArray(density)
    out.GetRasterBand(1).SetDescription("GHSL 2020 estimated residents per km2; bilinear-resampled density")
    out.FlushCache()
    out = None
    ds = None
    values = density[rows, cols]
    valid_cells = np.isfinite(values) & (values != NODATA_FLOAT) & (values >= 0)
    if valid_cells.mean() < 0.99:
        raise RuntimeError(f"Population coverage only {valid_cells.mean():.1%} of grid cells")
    values[~valid_cells] = 0.0
    return values


def cpcb_heat_index_by_scene(scenes, xs, ys):
    target_crs = QgsCoordinateReferenceSystem(CONFIG["spatial"]["crs"])
    source_crs = QgsCoordinateReferenceSystem("EPSG:4326")
    transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())
    station_coords = {}
    for station in CONFIG["cpcb"]["stations"]:
        point = QgsPointXY(float(station["longitude"]), float(station["latitude"]))
        transform.transform(point)
        station_coords[station["site_code"]] = np.array([point.x(), point.y()])
    grouped = {scene["item_id"]: [] for scene in scenes}
    if CPCB_MATCHES.is_file():
        with CPCB_MATCHES.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("item_id") in grouped: grouped[row["item_id"]].append(row)
    era5_by_scene = {}
    if ERA5_MATCHES.is_file():
        with ERA5_MATCHES.open(newline="", encoding="utf-8") as stream:
            era5_by_scene = {r["item_id"]: r for r in csv.DictReader(stream)}
    k = int(CONFIG["phase2"]["heat_index_idw_neighbors"])
    power = float(CONFIG["phase2"]["heat_index_idw_power"])
    hi_surfaces = np.full((len(scenes), len(xs)), np.nan, dtype=np.float32)
    scene_fields = []
    for si, scene in enumerate(scenes):
        pts, his = [], []
        for row in grouped.get(scene["item_id"], []):
            try:
                air_c, rh = float(row["air_temperature_c"]), float(row["relative_humidity_percent"])
            except (TypeError, ValueError): continue
            if not (math.isfinite(air_c) and math.isfinite(rh) and 0 <= rh <= 100): continue
            if row.get("site_code") not in station_coords: continue
            pts.append(station_coords[row["site_code"]]); his.append(heat_index_celsius(air_c,rh))
        source, interpolation = "CPCB CAAQMS", f"IDW power {power}"
        station_min = station_median = station_max = city_median = None
        if len(pts) < 3:
            climate = era5_by_scene.get(scene["item_id"], {})
            try:
                air_c, rh = float(climate["temperature_2m"]), float(climate["relative_humidity_2m"])
            except (KeyError,TypeError,ValueError):
                raise RuntimeError(f"No adequate CPCB support or matched ERA5 temperature/RH for {scene['item_id']}")
            if not (math.isfinite(air_c) and math.isfinite(rh) and 0 <= rh <= 100):
                raise RuntimeError(f"Invalid ERA5 temperature/RH for {scene['item_id']}")
            city_median=heat_index_celsius(air_c,rh); hi_surfaces[si,:]=city_median
            source="ERA5 background fallback"
            interpolation="spatially constant ERA5 city background; fewer than three paired CPCB stations"
        else:
            pts=np.asarray(pts,dtype=np.float64); his=np.asarray(his,dtype=np.float64)
            dx=xs[:,None]-pts[None,:,0]; dy=ys[:,None]-pts[None,:,1]
            distance=np.hypot(dx,dy); nearest=np.argsort(distance,axis=1)[:,:min(k,len(pts))]
            nearest_distance=np.take_along_axis(distance,nearest,axis=1)
            weights=1.0/np.maximum(nearest_distance,1.0)**power; values=his[nearest]
            hi_surfaces[si,:]=np.sum(weights*values,axis=1)/np.sum(weights,axis=1)
            station_min,station_median,station_max=float(his.min()),float(np.median(his)),float(his.max())
            city_median=float(np.median(hi_surfaces[si]))
            interpolation+=f", {min(k,len(pts))} nearest paired stations"
        scene_fields.append({"item_id":scene["item_id"],"scene_date_utc":scene["date_utc"],
            "paired_station_count":len(pts),"heat_index_source":source,
            "heat_index_station_min_c":station_min,"heat_index_station_median_c":station_median,
            "heat_index_station_max_c":station_max,"heat_index_city_median_idw_c":city_median,
            "interpolation":interpolation})
    return hi_surfaces,scene_fields


def city_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float32)
    indices = np.flatnonzero(valid & np.isfinite(values))
    if len(indices) == 0:
        return result
    result[indices] = (rankdata(values[indices], method="average") / len(indices)).astype(np.float32)
    return result


def safe_nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.nansum(values, axis=axis)
    output = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=output, where=count > 0)
    return output


def gistar(hot_stress: np.ndarray, grid_valid: np.ndarray, rows, cols, raster_meta):
    width, height, _, _ = raster_meta
    kernel = np.ones((3, 3), dtype=np.float64)  # queen contiguity including focal cell
    values_grid = np.full((height, width), 0.0, dtype=np.float64)
    mask_grid = np.zeros((height, width), dtype=np.float64)
    values_grid[rows[grid_valid], cols[grid_valid]] = hot_stress[grid_valid]
    mask_grid[rows[grid_valid], cols[grid_valid]] = 1.0
    n = int(grid_valid.sum())
    mean = float(np.mean(hot_stress[grid_valid]))
    std = float(np.std(hot_stress[grid_valid], ddof=0))
    local_count = convolve(mask_grid, kernel, mode="constant", cval=0.0)
    local_sum = convolve(values_grid, kernel, mode="constant", cval=0.0)
    variance_term = (n * local_count - local_count * local_count) / max(n - 1, 1)
    denom = std * np.sqrt(np.maximum(variance_term, 0.0))
    z_grid = np.full((height, width), np.nan, dtype=np.float64)
    eligible = (mask_grid > 0) & (denom > 0)
    z_grid[eligible] = (local_sum[eligible] - mean * local_count[eligible]) / denom[eligible]
    z = z_grid[rows, cols].astype(np.float64)
    z[~grid_valid] = np.nan
    p = np.full(z.shape, np.nan, dtype=np.float64)
    # Use the normal survival function form to avoid cancellation at large z.
    p[grid_valid] = np.clip(ndtr(-z[grid_valid]), 1e-300, 1.0)
    # Benjamini-Hochberg false-discovery correction across the AOI cells.
    idx = np.flatnonzero(grid_valid)
    order = np.argsort(p[idx])
    ordered_p = p[idx][order]
    m = len(ordered_p)
    q_sorted = ordered_p * m / np.arange(1, m + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q = np.full(z.shape, np.nan, dtype=np.float64)
    q[idx[order]] = q_sorted
    return z.astype(np.float32), p.astype(np.float32), q.astype(np.float32), float(q_sorted.max())


def write_geotiff(path: Path, array: np.ndarray, raster_meta, dtype, nodata, description: str):
    width, height, geotransform, projection = raster_meta
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), width, height, 1, dtype,
                       options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata))
    band.SetDescription(description)
    band.WriteArray(array)
    band.FlushCache()
    ds.FlushCache()
    ds = None


def write_map(array, raster_meta, output: Path, title: str, *, categorical=False,
              cmap=None, vmin=None, vmax=None, units="", category_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    width, height, gt, _ = raster_meta
    xmin, xres, _, ymax, _, yres = gt
    extent = (xmin, xmin + width * xres, ymax + height * yres, ymax)
    ward_layer = QgsVectorLayer(str(WARD_PATH), "wards", "ogr")
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    masked = np.ma.masked_invalid(array)
    if categorical:
        colors = cmap or SEVERITY_COLORS
        cm = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(0.5, len(colors) + 1.5, 1), cm.N)
        im = ax.imshow(masked, extent=extent, origin="upper", cmap=cm, norm=norm,
                       interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(1, len(colors) + 1), shrink=0.78)
        cbar.ax.set_yticklabels(category_labels or SEVERITY_NAMES[:len(colors)])
    else:
        im = ax.imshow(masked, extent=extent, origin="upper", cmap=cmap or "magma",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, shrink=0.78)
        if units:
            cbar.set_label(units)
    for feature in ward_layer.getFeatures():
        geom = feature.geometry()
        polygons = geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]
        for polygon in polygons:
            if not polygon:
                continue
            ring = polygon[0]
            ax.plot([p.x() for p in ring], [p.y() for p in ring],
                    color="#242424", linewidth=0.35, alpha=0.7)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Easting (m), EPSG:32644")
    ax.set_ylabel("Northing (m), EPSG:32644")
    ax.set_aspect("equal")
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for folder in (PHASE2_DIR, MAP_DIR, TABLE_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    scenes = read_scene_manifest()
    ids, rows, cols, xs, ys, ward_names = grid_features()
    lst, raster_meta = load_lst(scenes, rows, cols)
    pop_density = load_population_density(raster_meta, rows, cols)
    pop_valid = np.isfinite(pop_density) & (pop_density >= 0)
    pop_pct = city_percentile(pop_density, pop_valid)
    # The centroid table's iteration order is not guaranteed to match grid order;
    # reconstruct per-grid area by ID for deterministic exposure estimates.
    centroids = QgsVectorLayer(str(CENTROID_PATH), "c", "ogr")
    area_by_id = {int(f["grid_id"]): float(f["area_km2"]) for f in centroids.getFeatures()}
    cell_area_km2 = np.asarray([area_by_id[int(grid_id)] for grid_id in ids], dtype=np.float32)
    estimated_population = pop_density * cell_area_km2

    hi, scene_metadata = cpcb_heat_index_by_scene(scenes, xs, ys)
    n_scenes, n_cells = lst.shape
    anomaly = np.full_like(lst, np.nan, dtype=np.float32)
    hottest = np.zeros_like(lst, dtype=bool)
    lst_pct = np.full_like(lst, np.nan, dtype=np.float32)
    hi_pct = np.full_like(lst, np.nan, dtype=np.float32)
    hazard = np.full_like(lst, np.nan, dtype=np.float32)
    stress = np.full_like(lst, np.nan, dtype=np.float32)
    scene_metadata_by_id = {m["item_id"]: m for m in scene_metadata}
    for si, scene in enumerate(scenes):
        valid_lst = np.isfinite(lst[si])
        if valid_lst.sum() < 100:
            raise RuntimeError(f"Too few valid LST cells in {scene['item_id']}")
        city_median = float(np.nanmedian(lst[si, valid_lst]))
        anomaly[si, valid_lst] = lst[si, valid_lst] - city_median
        p90 = float(np.nanpercentile(lst[si, valid_lst], 90))
        hottest[si, valid_lst] = lst[si, valid_lst] >= p90
        valid = valid_lst & np.isfinite(hi[si]) & pop_valid
        lst_pct[si] = city_percentile(anomaly[si], valid)
        hi_pct[si] = city_percentile(hi[si], valid)
        hazard[si, valid] = 0.5 * lst_pct[si, valid] + 0.5 * hi_pct[si, valid]
        stress[si, valid] = 100.0 * hazard[si, valid] * pop_pct[valid]
        scene_metadata_by_id[scene["item_id"]].update({
            "city_median_lst_c": city_median,
            "city_p90_lst_c": p90,
            "valid_lst_cells": int(valid_lst.sum()),
            "heat_index_city_median_idw_c": float(np.nanmedian(hi[si])),
            "hottest_decile_cells": int(hottest[si].sum()),
        })

    scene_ids = [scene["item_id"] for scene in scenes]
    hot_idx = np.asarray([i for i, scene in enumerate(scenes) if scene["season"] == "hot"], dtype=int)
    cool_idx = np.asarray([i for i, scene in enumerate(scenes) if scene["season"] == "cool"], dtype=int)
    if len(hot_idx) < 2 or len(cool_idx) < 2:
        raise RuntimeError("Need at least two LST observations in each season")
    hot_stress = safe_nanmean(stress[hot_idx], axis=0)
    cool_stress = safe_nanmean(stress[cool_idx], axis=0)
    hot_hazard = safe_nanmean(hazard[hot_idx], axis=0)
    hot_anomaly = safe_nanmean(anomaly[hot_idx], axis=0)
    cool_anomaly = safe_nanmean(anomaly[cool_idx], axis=0)
    hot_hi = safe_nanmean(hi[hot_idx], axis=0)
    cool_hi = safe_nanmean(hi[cool_idx], axis=0)
    n_valid = np.isfinite(lst).sum(axis=0).astype(np.uint8)
    hot_count = hottest.sum(axis=0).astype(np.uint8)
    persistence = np.divide(hot_count.astype(np.float32), n_valid,
                            out=np.full(n_cells, np.nan, dtype=np.float32), where=n_valid > 0)
    valid_analysis = np.isfinite(hot_stress) & np.isfinite(pop_density) & (pop_density >= 0)
    if valid_analysis.sum() < n_cells * 0.98:
        raise RuntimeError(f"Only {valid_analysis.mean():.1%} of cells support a Phase 2 score")

    breaks = np.nanpercentile(hot_stress[valid_analysis], [20, 40, 60, 80])
    hot_severity = np.full(n_cells, 0, dtype=np.uint8)
    hot_severity[valid_analysis] = np.digitize(hot_stress[valid_analysis], breaks, right=True) + 1
    cool_severity = np.full(n_cells, 0, dtype=np.uint8)
    cool_valid = np.isfinite(cool_stress)
    cool_breaks = np.nanpercentile(cool_stress[cool_valid], [20, 40, 60, 80])
    cool_severity[cool_valid] = np.digitize(cool_stress[cool_valid], cool_breaks, right=True) + 1

    gistar_z, gistar_p, gistar_q, q_max = gistar(hot_stress, valid_analysis, rows, cols, raster_meta)
    hot_cluster = valid_analysis & (gistar_q <= float(CONFIG["phase2"]["cluster_fdr_q"])) & (gistar_z > 0)

    width, height, gt, projection = raster_meta
    def as_raster(values, fill=NODATA_FLOAT, dtype=np.float32):
        arr = np.full((height, width), fill, dtype=dtype)
        good = np.isfinite(values)
        arr[rows[good], cols[good]] = values[good].astype(dtype)
        return arr

    raster_outputs = [
        ("Chennai_hot_LST_anomaly_city_median_C.tif", as_raster(hot_anomaly), gdal.GDT_Float32,
         NODATA_FLOAT, "Hot-season mean LST anomaly relative to each scene's Chennai median, C"),
        ("Chennai_hot_heat_index_C.tif", as_raster(hot_hi), gdal.GDT_Float32, NODATA_FLOAT,
         "IDW-interpolated hot-season NOAA Heat Index from CPCB AT and RH, C"),
        ("Chennai_hot_heat_stress_index_0_100.tif", as_raster(hot_stress), gdal.GDT_Float32,
         NODATA_FLOAT, "Hazard percentile x GHSL population-density percentile x 100"),
        ("Chennai_hot_heat_stress_severity_1_5.tif", as_raster(hot_severity, 0, np.uint8),
         gdal.GDT_Byte, 0, "City-relative hot-season heat-stress quintile: 1 very low to 5 very high"),
        ("Chennai_cool_heat_stress_index_0_100.tif", as_raster(cool_stress), gdal.GDT_Float32,
         NODATA_FLOAT, "Cool-season heat-stress index: hazard percentile x exposure percentile x 100"),
        ("Chennai_cool_heat_stress_severity_1_5.tif", as_raster(cool_severity, 0, np.uint8),
         gdal.GDT_Byte, 0, "City-relative cool-season heat-stress quintile"),
        ("Chennai_hot_LST_hottest_decile_persistence.tif", as_raster(persistence), gdal.GDT_Float32,
         NODATA_FLOAT, "Share of valid Landsat scenes in which cell is in scene-specific AOI hottest LST decile"),
        ("Chennai_hot_heat_stress_GiStar_z.tif", as_raster(gistar_z), gdal.GDT_Float32,
         NODATA_FLOAT, "Getis-Ord Gi-star z-score on hot-season heat-stress index"),
        ("Chennai_hot_heat_stress_GiStar_hotspot_FDR05.tif", as_raster(hot_cluster.astype(np.uint8), 255, np.uint8),
         gdal.GDT_Byte, 255, "High Gi-star clusters, one-sided p-values adjusted by Benjamini-Hochberg q<=0.05"),
        ("Chennai_GHSL2020_population_density_30m.tif", as_raster(pop_density), gdal.GDT_Float32,
         NODATA_FLOAT, "GHSL 2020 estimated persons per km2; bilinear resampling from 100 m counts"),
    ]
    for name, arr, dtype, nodata, description in raster_outputs:
        write_geotiff(PHASE2_DIR / name, arr, raster_meta, dtype, nodata, description)

    # Grid ranking and the input metrics used to interpret it.
    rank_indices = np.flatnonzero(valid_analysis)
    order = np.lexsort((-np.nan_to_num(gistar_z[rank_indices], nan=-999.0),
                        -np.nan_to_num(persistence[rank_indices], nan=-1.0),
                        -hot_stress[rank_indices]))
    ranked_indices = rank_indices[order]
    rank_by_cell = np.full(n_cells, 0, dtype=np.int32)
    rank_by_cell[ranked_indices] = np.arange(1, len(ranked_indices) + 1)
    grid_rank_path = TABLE_DIR / "Chennai_grid_hotspot_priority_ranking.csv"
    with grid_rank_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "priority_rank", "grid_id", "ward", "easting_m", "northing_m", "cell_area_km2",
            "population_density_people_km2_2020", "estimated_population_2020",
            "mean_hot_LST_anomaly_C", "mean_cool_LST_anomaly_C", "mean_hot_heat_index_C",
            "mean_cool_heat_index_C", "hot_hazard_percentile", "population_density_percentile",
            "hot_heat_stress_index_0_100", "hot_severity", "cool_heat_stress_index_0_100",
            "hottest_decile_observations", "valid_LST_observations", "persistence_share",
            "GiStar_z_hot_stress", "GiStar_p_one_sided", "GiStar_FDR_q", "GiStar_hot_cluster_FDR05",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for idx in ranked_indices:
            writer.writerow({
                "priority_rank": int(rank_by_cell[idx]), "grid_id": int(ids[idx]),
                "ward": ward_names[idx], "easting_m": round(float(xs[idx]), 2),
                "northing_m": round(float(ys[idx]), 2), "cell_area_km2": float(cell_area_km2[idx]),
                "population_density_people_km2_2020": round(float(pop_density[idx]), 3),
                "estimated_population_2020": round(float(estimated_population[idx]), 4),
                "mean_hot_LST_anomaly_C": round(float(hot_anomaly[idx]), 4),
                "mean_cool_LST_anomaly_C": round(float(cool_anomaly[idx]), 4),
                "mean_hot_heat_index_C": round(float(hot_hi[idx]), 4),
                "mean_cool_heat_index_C": round(float(cool_hi[idx]), 4),
                "hot_hazard_percentile": round(float(hot_hazard[idx]), 6),
                "population_density_percentile": round(float(pop_pct[idx]), 6),
                "hot_heat_stress_index_0_100": round(float(hot_stress[idx]), 6),
                "hot_severity": SEVERITY_NAMES[int(hot_severity[idx]) - 1],
                "cool_heat_stress_index_0_100": round(float(cool_stress[idx]), 6) if np.isfinite(cool_stress[idx]) else "",
                "hottest_decile_observations": int(hot_count[idx]),
                "valid_LST_observations": int(n_valid[idx]),
                "persistence_share": round(float(persistence[idx]), 6) if np.isfinite(persistence[idx]) else "",
                "GiStar_z_hot_stress": round(float(gistar_z[idx]), 6) if np.isfinite(gistar_z[idx]) else "",
                "GiStar_p_one_sided": float(gistar_p[idx]) if np.isfinite(gistar_p[idx]) else "",
                "GiStar_FDR_q": float(gistar_q[idx]) if np.isfinite(gistar_q[idx]) else "",
                "GiStar_hot_cluster_FDR05": int(hot_cluster[idx]),
            })

    # Ward summaries: relative severity plus the estimated population in
    # high/very-high cells. Both per-cell intensity and exposed-population ranks
    # are included, since they answer distinct planning questions.
    unique_wards, ward_codes = np.unique(ward_names, return_inverse=True)
    ward_count = np.bincount(ward_codes, minlength=len(unique_wards)).astype(np.float64)
    mean_stress = np.bincount(ward_codes, weights=np.nan_to_num(hot_stress), minlength=len(unique_wards)) / ward_count
    mean_hazard = np.bincount(ward_codes, weights=np.nan_to_num(hot_hazard), minlength=len(unique_wards)) / ward_count
    mean_persistence = np.bincount(ward_codes, weights=np.nan_to_num(persistence), minlength=len(unique_wards)) / ward_count
    high_cells = np.bincount(ward_codes, weights=(hot_severity >= 4).astype(float), minlength=len(unique_wards))
    high_population = np.bincount(ward_codes, weights=estimated_population * (hot_severity >= 4), minlength=len(unique_wards))
    cluster_cells = np.bincount(ward_codes, weights=hot_cluster.astype(float), minlength=len(unique_wards))
    hot_rank_order = np.argsort(-mean_stress)
    rank_mean = np.empty(len(unique_wards), dtype=np.int32)
    rank_mean[hot_rank_order] = np.arange(1, len(unique_wards) + 1)
    rank_pop_order = np.argsort(-high_population)
    rank_pop = np.empty(len(unique_wards), dtype=np.int32)
    rank_pop[rank_pop_order] = np.arange(1, len(unique_wards) + 1)
    ward_rank_path = TABLE_DIR / "Chennai_ward_hotspot_priority_ranking.csv"
    with ward_rank_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["priority_rank_mean_stress", "priority_rank_high_severity_population",
                  "ward", "grid_cells", "mean_hot_heat_stress_index_0_100",
                  "mean_hot_hazard_percentile", "mean_LST_persistence_share",
                  "high_or_very_high_cells", "high_or_very_high_cell_share",
                  "estimated_population_in_high_or_very_high_cells_2020", "GiStar_hot_cluster_cells"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for wi in hot_rank_order:
            writer.writerow({
                "priority_rank_mean_stress": int(rank_mean[wi]),
                "priority_rank_high_severity_population": int(rank_pop[wi]),
                "ward": unique_wards[wi], "grid_cells": int(ward_count[wi]),
                "mean_hot_heat_stress_index_0_100": round(float(mean_stress[wi]), 4),
                "mean_hot_hazard_percentile": round(float(mean_hazard[wi]), 5),
                "mean_LST_persistence_share": round(float(mean_persistence[wi]), 5),
                "high_or_very_high_cells": int(high_cells[wi]),
                "high_or_very_high_cell_share": round(float(high_cells[wi] / ward_count[wi]), 5),
                "estimated_population_in_high_or_very_high_cells_2020": round(float(high_population[wi]), 1),
                "GiStar_hot_cluster_cells": int(cluster_cells[wi]),
            })

    # Phase 2 discovery maps.
    hot_grid = as_raster(hot_severity.astype(np.float32), np.nan)
    hot_grid[hot_grid == 0] = np.nan
    cool_grid = as_raster(cool_severity.astype(np.float32), np.nan)
    cool_grid[cool_grid == 0] = np.nan
    persistence_grid = as_raster(persistence, np.nan)
    stress_grid = as_raster(hot_stress)
    gistar_grid = as_raster(gistar_z, np.nan)
    write_map(hot_grid, raster_meta, MAP_DIR / "Chennai_hot_season_heat_stress_severity.png",
              "Chennai hot-season heat stress severity\nCity-relative quintiles; LST + CPCB Heat Index × population exposure",
              categorical=True)
    write_map(cool_grid, raster_meta, MAP_DIR / "Chennai_cool_season_heat_stress_severity.png",
              "Chennai cool-season heat stress severity\nCity-relative quintiles; descriptive seasonal comparison",
              categorical=True)
    write_map(persistence_grid, raster_meta, MAP_DIR / "Chennai_LST_hottest_decile_persistence.png",
              "Landsat LST hottest-decile persistence\nShare of valid observations, 2021–2025",
              cmap="viridis", vmin=0, vmax=1, units="Share of valid scenes")
    write_map(gistar_grid, raster_meta, MAP_DIR / "Chennai_hot_heat_stress_GiStar_zscore.png",
              "Hot-season heat-stress Getis-Ord Gi* z-score\n3×3 queen neighborhood; FDR q≤0.05 clusters in companion raster",
              cmap="coolwarm", vmin=-4, vmax=4, units="Gi* z-score")
    cluster_class = np.where(valid_analysis, np.where(hot_cluster, 1.0, 2.0), np.nan)
    write_map(as_raster(cluster_class, np.nan), raster_meta,
              MAP_DIR / "Chennai_hot_heat_stress_GiStar_hot_clusters.png",
              "Significant hot clusters of population-weighted heat stress\nGetis-Ord Gi*; one-sided Benjamini-Hochberg FDR q≤0.05",
              categorical=True, cmap=["#b2182b", "#e6e6e6"],
              category_labels=["Significant hot cluster", "Not significant"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ward_top = hot_rank_order[:15]
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ax.barh(unique_wards[ward_top][::-1], mean_stress[ward_top][::-1], color="#d95f02")
    ax.set_xlabel("Mean hot-season heat-stress index (0–100)")
    ax.set_title("Top 15 Chennai wards by mean hot-season heat stress", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(MAP_DIR / "Chennai_top15_ward_priority.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    scene_meta_path = TABLE_DIR / "Chennai_phase2_scene_metrics.csv"
    with scene_meta_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(scene_metadata_by_id[scene_ids[0]].keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in scenes:
            writer.writerow(scene_metadata_by_id[scene["item_id"]])

    metadata = {
        "study_period": [CONFIG["study"]["start_date"], CONFIG["study"]["end_date"]],
        "scenes": len(scenes), "grid_cells": int(n_cells), "wards": int(len(unique_wards)),
        "population_source": CONFIG["population"]["source"], "population_epoch": CONFIG["population"]["epoch"],
        "population_tile": CONFIG["population"]["tile_id"],
        "population_density_units": "estimated residents per km2, bilinear-resampled from GHSL 100m counts",
        "lst_anomaly_reference": "scene-specific Chennai AOI median",
        "heat_index": "NOAA Heat Index from CPCB station AT/RH where matched; otherwise ERA5 overpass temperature/RH at one background grid point. IDW power 2 over available support points; ERA5 fallback is spatially constant.",
        "heat_index_sources": sorted({row["heat_index_source"] for row in scene_metadata}),
        "hazard_index": "mean of scene-wise city percentile ranks for LST anomaly and interpolated Heat Index; equal 0.5 weights",
        "exposure_index": "GHSL population-density percentile across Chennai grid cells",
        "heat_stress_index": "100 x mean hazard percentile x population-density percentile; averaged per season",
        "severity_breaks_hot_quintiles": [float(v) for v in breaks],
        "severity_breaks_cool_quintiles": [float(v) for v in cool_breaks],
        "cluster_method": "Getis-Ord Gi* on hot-season heat-stress index; 3x3 queen weights include focal cell; one-sided normal p and Benjamini-Hochberg FDR q<=0.05",
        "cluster_cells_fdr05": int(hot_cluster.sum()),
        "max_bh_qvalue": q_max,
        "persistence": "share of valid scenes where cell LST >= that scene's AOI 90th percentile",
        "population_interpretation": "GHSL epoch 2020 is a modeled residential population estimate, not a 2023 census count; 30m values are resampled density estimates",
        "heat_index_interpretation": "shade/light-wind apparent temperature; it is not UTCI/WBGT and does not include mean radiant temperature",
        "severity_interpretation": "city-relative quintiles, not public-health warning thresholds",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpcb_station_pairs_by_scene": scene_metadata,
        "outputs": {
            "grid_ranking": str(grid_rank_path.relative_to(ROOT)),
            "ward_ranking": str(ward_rank_path.relative_to(ROOT)),
            "maps": [str(path.relative_to(ROOT)) for path in sorted(MAP_DIR.glob("Chennai_*.png"))],
        },
    }
    (REPORT_DIR / "Chennai_phase2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Phase 2: {n_cells:,} grid cells; {len(unique_wards)} wards; {int(hot_cluster.sum()):,} FDR-significant hot-cluster cells")
    print(f"Grid ranking: {grid_rank_path}")
    print(f"Ward ranking: {ward_rank_path}")
    print(f"Maps: {MAP_DIR}")


if __name__ == "__main__":
    main()
