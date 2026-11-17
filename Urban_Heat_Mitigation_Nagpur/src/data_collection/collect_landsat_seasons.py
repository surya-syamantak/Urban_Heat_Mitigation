"""Collect Landsat C2 L2 AOI windows and build seasonal LST/index composites.

Uses STAC metadata and signed Cloud-Optimized GeoTIFF windows from the
Microsoft Planetary Computer. Source scenes remain remotely hosted; metadata
JSON is saved locally, and all AOI outputs are written on the existing 30 m
Nagpur grid. Run in the QGIS Python environment via ``main.py landsat``.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import yaml
from osgeo import gdal
from landsat_catalog import selected_item_ids


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
GRID = ROOT / CONFIG["spatial"]["grid"]
AOI = ROOT / CONFIG["spatial"]["wards"]
OUT_DIR = ROOT / "data/processed/Landsat"
META_DIR = ROOT / "data/raw/Landsat/Nagpur/metadata"
MANIFEST = OUT_DIR / "Landsat_scene_manifest.csv"
RUN_LOG = OUT_DIR / "landsat_collection.log"

ST_SCALE = float(CONFIG["landsat"]["surface_temperature_scale"])
ST_OFFSET = float(CONFIG["landsat"]["surface_temperature_offset_kelvin"])
SR_SCALE = float(CONFIG["landsat"]["reflectance_scale"])
SR_OFFSET = float(CONFIG["landsat"]["reflectance_offset"])
ST_QA_SCALE = float(CONFIG["landsat"]["st_qa_scale_kelvin"])
NODATA = -9999.0

ASSETS = {
    "qa": ("stqa", -9999),
    "qa_pixel": ("qa", 0),
    "blue": ("blue", 0),
    "green": ("green", 0),
    "red": ("red", 0),
    "nir08": ("nir", 0),
    "swir16": ("swir1", 0),
    "swir22": ("swir2", 0),
}


def get_json(url: str) -> dict:
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "Urban-Heat-Mitigation/1.0"})
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def log(message: str) -> None:
    print(message, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def grid_parameters() -> tuple[tuple[float, float, float, float], int, int]:
    connection = sqlite3.connect(GRID)
    try:
        table = connection.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
        ).fetchone()[0]
        left, bottom, right, top = connection.execute(
            f'SELECT MIN("left"), MIN("bottom"), MAX("right"), MAX("top") FROM "{table}"'
        ).fetchone()
    finally:
        connection.close()
    width = round((right - left) / 30.0)
    height = round((top - bottom) / 30.0)
    if abs(right - left - width * 30.0) > 0.01 or abs(top - bottom - height * 30.0) > 0.01:
        raise ValueError("The configured master grid is not a regular 30 m grid")
    return (left, bottom, right, top), width, height


def signed_asset_url(href: str, token: str) -> str:
    return href + ("&" if "?" in href else "?") + token.lstrip("?")


def warp_window(href: str, token: str, bounds: tuple[float, float, float, float],
                width: int, height: int, fill: int) -> tuple[np.ndarray, gdal.Dataset]:
    source = "/vsicurl/" + signed_asset_url(href, token)
    result = None
    for attempt in range(4):
        try:
            result = gdal.Warp(
                "", source, format="MEM", cutlineDSName=str(AOI),
                outputBounds=bounds, width=width, height=height, dstNodata=fill,
                resampleAlg="near", multithread=False,
            )
            if result is not None:
                break
        except Exception:
            if attempt == 3:
                raise
        if attempt < 3:
            time.sleep(2 ** attempt)
    if result is None:
        raise RuntimeError(f"GDAL could not read source asset: {href}")
    return result.GetRasterBand(1).ReadAsArray(), result


def write_raster(path: Path, array: np.ndarray, reference: gdal.Dataset,
                 dtype: int, nodata: float, description: str, tags: dict[str, str]) -> None:
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(path), reference.RasterXSize, reference.RasterYSize, 1, dtype,
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    if dataset is None:
        raise RuntimeError(f"Could not create raster {path}")
    dataset.SetGeoTransform(reference.GetGeoTransform())
    dataset.SetProjection(reference.GetProjection())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.SetDescription(description)
    band.WriteArray(array)
    band.SetMetadata(tags)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def season_for(date: str) -> str:
    month = int(date[5:7])
    for name, definition in CONFIG["study"]["seasons"].items():
        if month in [int(value) for value in definition.get("months", [])]:
            return name
    raise ValueError(f"Scene {date} does not fall in a configured analysis season")


def main() -> None:
    gdal.UseExceptions()
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF")
    for path in (GRID, AOI):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("Starting Landsat collection\n", encoding="utf-8")

    log("Requesting Planetary Computer token")
    token_response = get_json(CONFIG["landsat"]["sas_api"])
    token = token_response["token"]
    stac = CONFIG["landsat"]["stac_api"].rstrip("/")
    collection = CONFIG["study"]["landsat_stac_collection"]
    bounds, width, height = grid_parameters()
    seasonal_stack: dict[str, dict[str, list[np.ndarray]]] = {}
    manifest_rows: list[dict[str, object]] = []
    output_tags_base = {
        "grid": GRID.name,
        "crs": CONFIG["spatial"]["crs"],
        "pixel_size_m": "30",
        "qa_pixel_rejected_bits": ",".join(map(str, CONFIG["landsat"]["qa_pixel_reject_bits"])),
        "st_qa_cutoff_kelvin": "not applied; retain and report uncertainty",
    }

    for item_number, item_id in enumerate(selected_item_ids(CONFIG), start=1):
        if item_number > 1 and (item_number - 1) % 5 == 0:
            log("Refreshing Planetary Computer asset token")
            token = get_json(CONFIG["landsat"]["sas_api"])["token"]
        log(f"Reading STAC item {item_id}")
        item = get_json(f"{stac}/collections/{collection}/items/{item_id}")
        properties = item["properties"]
        acquisition = properties["datetime"]
        date = acquisition[:10]
        season = season_for(date)
        scene_dir = OUT_DIR / item_id
        lst_path = scene_dir / f"{item_id}_LST_Celsius.tif"
        mask_path = scene_dir / f"{item_id}_QA_Mask.tif"
        index_paths = {name: scene_dir / f"{item_id}_{name.upper()}.tif"
                       for name in ("ndvi", "ndbi", "mndwi", "albedo")}
        cached = (lst_path.is_file() and mask_path.is_file()
                  and all(path.is_file() for path in index_paths.values()))
        arrays: dict[str, np.ndarray] = {}
        reference = None
        cached_lst = None
        cached_indices = None
        platform = str(properties.get("platform", "")).lower()
        thermal_asset = "lwir11" if platform in ("landsat-8", "landsat-9") else "lwir"
        if thermal_asset not in item.get("assets", {}):
            raise RuntimeError(f"{item_id} ({platform}) has no supported thermal asset {thermal_asset!r}")
        scene_assets = {**ASSETS, thermal_asset: ("st", 0)}

        if cached:
            log(f"Reusing cached scene rasters for {item_id}")
            reference = gdal.Open(str(lst_path))
            mask_ds = gdal.Open(str(mask_path))
            if reference is None or mask_ds is None:
                raise RuntimeError(f"Could not reopen cached scene rasters for {item_id}")
            cached_mask = mask_ds.GetRasterBand(1).ReadAsArray()
            mask_ds = None
            lst_ds = gdal.Open(str(lst_path))
            cached_lst = lst_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            lst_ds = None
            cached_lst[cached_lst == NODATA] = np.nan
            cached_indices = {}
            for name, path in index_paths.items():
                index_ds = gdal.Open(str(path))
                if index_ds is None:
                    raise RuntimeError(f"Could not reopen cached index raster {path}")
                value = index_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
                index_ds = None
                value[value == NODATA] = np.nan
                cached_indices[name] = value
            cached_clear = cached_mask == 1
            for asset_key, short_name, fill in ((thermal_asset, "st", 0), ("qa", "stqa", -9999)):
                log(f"Reading {item_id} QA metadata asset {asset_key}")
                arrays[short_name], _ = warp_window(
                    item["assets"][asset_key]["href"], token, bounds, width, height, fill
                )
            arrays["qa"] = np.where(cached_clear, 0, 1).astype(np.uint16)
        else:
            for asset_key, (short_name, fill) in scene_assets.items():
                log(f"Reading {item_id} asset {asset_key}")
                href = item["assets"][asset_key]["href"]
                array, raster = warp_window(href, token, bounds, width, height, fill)
                arrays[short_name] = array
                if reference is None:
                    reference = raster
                elif (raster.RasterXSize, raster.RasterYSize, raster.GetGeoTransform()) != (
                    reference.RasterXSize, reference.RasterYSize, reference.GetGeoTransform()
                ):
                    raise ValueError(f"Misaligned source bands for {item_id}")

        qa_pixel = arrays["qa"].astype(np.uint16, copy=False)
        raw_st = arrays["st"].astype(np.uint16, copy=False)
        stqa_dn = arrays["stqa"].astype(np.int16, copy=False)
        thermal_valid = raw_st != 0
        qa_clear = np.ones(raw_st.shape, dtype=bool)
        for bit in CONFIG["landsat"]["qa_pixel_reject_bits"]:
            qa_clear &= (qa_pixel & (1 << int(bit))) == 0
        stqa_valid = stqa_dn != -9999
        clear = thermal_valid & qa_clear & stqa_valid
        lst_c = raw_st.astype(np.float32) * ST_SCALE + ST_OFFSET - 273.15
        clear &= np.isfinite(lst_c) & (lst_c > -80.0) & (lst_c < 80.0)
        if not clear.any():
            log(
                f"Skipping {item_id}: no valid AOI pixels after QA "
                f"(thermal={int(thermal_valid.sum()):,}, "
                f"QA-clear={int(qa_clear.sum()):,}, "
                f"ST-QA-valid={int(stqa_valid.sum()):,})"
            )
            continue

        if cached:
            indices = cached_indices
            lst_c = np.where(clear, cached_lst, np.nan).astype(np.float32)
        else:
            reflectance: dict[str, np.ndarray] = {}
            for band in ("blue", "green", "red", "nir", "swir1", "swir2"):
                dn = arrays[band].astype(np.float32, copy=False)
                scaled = dn * SR_SCALE + SR_OFFSET
                scaled[(arrays[band] == 0) | ~clear] = np.nan
                reflectance[band] = scaled

            def safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                denominator = a + b
                return np.divide(a - b, denominator, out=np.full_like(a, np.nan),
                                 where=np.isfinite(denominator) & (np.abs(denominator) > 1e-6))

            indices = {
                "ndvi": safe_index(reflectance["nir"], reflectance["red"]),
                "ndbi": safe_index(reflectance["swir1"], reflectance["nir"]),
                "mndwi": safe_index(reflectance["green"], reflectance["swir1"]),
            }
            indices["albedo"] = (
                0.356 * reflectance["blue"] + 0.130 * reflectance["red"]
                + 0.373 * reflectance["nir"] + 0.085 * reflectance["swir1"]
                + 0.072 * reflectance["swir2"] - 0.0018
            )
        tags = {
            **output_tags_base,
            "landsat_item_id": item_id,
            "acquisition_utc": acquisition,
            "season": season,
            "year": str(int(date[:4])),
            "st_scale": str(ST_SCALE),
            "st_offset_kelvin": str(ST_OFFSET),
            "sr_scale": str(SR_SCALE),
            "sr_offset": str(SR_OFFSET),
            "st_qa_scale_kelvin": str(ST_QA_SCALE),
            "st_qa_uncertainty_p50_kelvin": f"{np.nanpercentile(stqa_dn[clear] * ST_QA_SCALE, 50):.3f}",
            "source": "USGS Landsat Collection 2 Level-2 via Microsoft Planetary Computer STAC",
        }
        scene_dir.mkdir(parents=True, exist_ok=True)
        if not cached:
            write_raster(lst_path, np.where(clear, lst_c, NODATA).astype(np.float32),
                         reference, gdal.GDT_Float32, NODATA, "Surface temperature (C)", tags)
            write_raster(mask_path, clear.astype(np.uint8), reference, gdal.GDT_Byte,
                         255, "1=clear, 0=masked", tags)
            for name, array in indices.items():
                path = scene_dir / f"{item_id}_{name.upper()}.tif"
                write_raster(path, np.where(np.isfinite(array), array, NODATA).astype(np.float32),
                             reference, gdal.GDT_Float32, NODATA, name.upper(), tags)
            log(f"Wrote scene rasters for {item_id}")

        (META_DIR / f"{item_id}_STAC.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8"
        )
        log(f"Saved STAC item metadata for {item_id}")
        mtl_path = META_DIR / f"{item_id}_MTL.txt"
        if not mtl_path.is_file():
            metadata_url = signed_asset_url(item["assets"]["mtl.txt"]["href"], token)
            log(f"Downloading MTL text for {item_id}")
            request = Request(metadata_url, headers={"User-Agent": "Urban-Heat-Mitigation/1.0"})
            with urlopen(request, timeout=60) as response:
                mtl_path.write_bytes(response.read())
            log(f"Saved MTL text for {item_id}")

        stack = seasonal_stack.setdefault(season, {key: [] for key in ("lst_c", *indices.keys())})
        stack["lst_c"].append(np.where(clear, lst_c, np.nan).astype(np.float32))
        for name, array in indices.items():
            stack[name].append(array.astype(np.float32))

        populated = int((raw_st != 0).sum())
        manifest_rows.append({
            "item_id": item_id,
            "product_id": properties.get("landsat:scene_id", item_id),
            "date_utc": acquisition,
            "season": season,
            "year": int(date[:4]),
            "platform": properties.get("platform"),
            "wrs_path": properties.get("landsat:wrs_path"),
            "wrs_row": properties.get("landsat:wrs_row"),
            "global_cloud_percent": properties.get("eo:cloud_cover"),
            "valid_pixels": int(clear.sum()),
            "st_b10_populated_pixels": populated,
            "valid_fraction_of_populated": f"{clear.sum() / max(populated, 1):.6f}",
            "lst_min_c": f"{float(lst_c[clear].min()):.3f}",
            "lst_median_c": f"{float(np.median(lst_c[clear])):.3f}",
            "lst_max_c": f"{float(lst_c[clear].max()):.3f}",
            "stqa_p50_k": f"{float(np.percentile(stqa_dn[clear] * ST_QA_SCALE, 50)):.3f}",
            "stqa_p90_k": f"{float(np.percentile(stqa_dn[clear] * ST_QA_SCALE, 90)):.3f}",
        })
        print(f"{item_id}: {int(clear.sum()):,} clear pixels; median LST {np.median(lst_c[clear]):.2f} C")

    if not manifest_rows:
        raise RuntimeError("No selected Landsat scenes produced valid pixels over the AOI")
    required_seasons = set(CONFIG["study"]["seasons"])
    missing_seasons = required_seasons - seasonal_stack.keys()
    if missing_seasons:
        raise RuntimeError(
            "No valid Landsat scenes for season(s): " + ", ".join(sorted(missing_seasons))
        )

    composite_reference = reference
    for season, bands in seasonal_stack.items():
        season_dir = OUT_DIR / f"seasonal_{season}"
        season_dir.mkdir(parents=True, exist_ok=True)
        composites = {
            name: np.nanmedian(np.stack(images, axis=0), axis=0)
            for name, images in bands.items()
        }
        season_tags = {
            **output_tags_base,
            "season": season,
            "composite": "per-pixel median of QA-screened scenes",
            "scene_count": str(len(bands["lst_c"])),
        }
        for name, array in composites.items():
            path = season_dir / f"Nagpur_{season}_{name.upper()}_median.tif"
            write_raster(path, np.where(np.isfinite(array), array, NODATA).astype(np.float32),
                         composite_reference, gdal.GDT_Float32, NODATA,
                         f"{season} median {name.upper()}", season_tags)

        ndvi, ndbi, mndwi = composites["ndvi"], composites["ndbi"], composites["mndwi"]
        lulc = np.zeros(ndvi.shape, dtype=np.uint8)
        valid = np.isfinite(ndvi) & np.isfinite(ndbi) & np.isfinite(mndwi)
        lulc[valid] = 1  # other/bare/impervious ambiguous
        lulc[valid & (mndwi > 0.20)] = 4  # open water; threshold heuristic
        lulc[valid & (ndvi > 0.30) & (mndwi <= 0.20)] = 2  # vegetation/tree proxy
        lulc[valid & (ndbi > 0.0) & (ndbi > ndvi) & (mndwi <= 0.20)] = 3  # built-up proxy
        write_raster(
            season_dir / f"Nagpur_{season}_LULC_rule_based.tif", lulc,
            composite_reference, gdal.GDT_Byte, 0, "Rule-based LULC proxy",
            {**season_tags, "class_1": "other/bare/ambiguous", "class_2": "vegetation proxy",
             "class_3": "built-up proxy", "class_4": "water proxy",
             "method": "thresholds on seasonal median NDVI, NDBI, MNDWI; requires validation"},
        )

    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    catalog_path = ROOT / "data/dataset_audit.csv"
    distinct_dates = sorted({str(row["date_utc"])[:10] for row in manifest_rows})
    seasons = sorted({str(row["season"]) for row in manifest_rows})
    with catalog_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Dataset", "Source", "Resolution or frequency", "Date/season",
                         "Spatial coverage", "Processing and status", "Limitations / path"])
        writer.writerow(["Landsat 8/9 Collection 2 Level-2 surface temperature",
            "USGS Landsat via Microsoft Planetary Computer STAC", "30 m thermal product aligned to 30 m grid",
            f"{distinct_dates[0]} to {distinct_dates[-1]}; {len(distinct_dates)} acquisition dates; seasons={','.join(seasons)}",
            CONFIG["project"]["city"] + " AOI",
            f"{len(manifest_rows)} item scenes processed; QA_PIXEL bits 0-5 masked; ST scale/offset and item metadata recorded",
            "data/processed/Landsat/Landsat_scene_manifest.csv; LST is surface temperature, not air temperature"])
        writer.writerow(["Landsat spectral indices and albedo",
            "USGS Landsat 8/9 Collection 2 Level-2 surface reflectance", "30 m aligned grid",
            f"{len(distinct_dates)} acquisitions, {CONFIG['study']['start_date']} to {CONFIG['study']['end_date']}",
            CONFIG["project"]["city"] + " AOI",
            "Per-scene NDVI, NDBI, MNDWI and empirical broadband albedo; hot/cool median composites regenerated",
            "data/processed/Landsat; empirical albedo and rule-based LULC require validation"])
        writer.writerow(["Hourly ERA5 and overpass matches", CONFIG["era5"]["provider"],
            "Hourly; coarse regional grid", f"{CONFIG['study']['start_date']} to {CONFIG['study']['end_date']}",
            f"Configured city grid point: {CONFIG['era5']['latitude']}, {CONFIG['era5']['longitude']}",
            "Nearest hourly record matched to every manifest item after Landsat collection",
            "data/raw/ERA5 and data/processed/ERA5_Landsat_Overpass_Matched.csv; background, not neighborhood weather"])
        writer.writerow(["CPCB station observations", "CPCB CAAQMS; local archives and public repository",
            "Hourly station records", f"{CONFIG['study']['start_date']} to {CONFIG['study']['end_date']}",
            CONFIG["cpcb"]["city"] + " configured stations",
            "Availability in CPCB manifest; paired observations matched where available; Phase 2 uses ERA5 when support is insufficient",
            "data/raw/CPCB and data/processed/CPCB_Landsat_Overpass_Matched.csv; inspect per-scene source and missingness"])
        writer.writerow(["Seasonal composites and urban-form inputs", "Derived from Landsat, ERA5, CPCB and OpenStreetMap",
            "30 m project grid", f"{len(distinct_dates)} dates; multi-year hot/cool composites",
            CONFIG["project"]["city"] + " AOI and wards",
            "Downstream inputs regenerated from selected 2021-2025 acquisitions",
            "data/processed; OSM is a current snapshot and some building attributes are proxies"])
    print(f"Wrote {MANIFEST} and {len(manifest_rows)} item scenes on the 30 m grid")
    print(f"Wrote data catalogue: {catalog_path}")


if __name__ == "__main__":
    main()
