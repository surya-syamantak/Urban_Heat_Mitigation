"""Clip, QA-mask, and scale a Landsat C2 Level-2 ST scene with QGIS GDAL.

Run at the repository root with QGIS 3 Python:
  python-qgis-ltr.bat src/preprocessing/landsat_c2_st_qgis.py

This writes a 30 m Celsius LST and validity mask without editing source files.
It does not apply a hard ST_QA uncertainty cutoff; the scene uncertainty is
summarized so a threshold can be chosen and documented later.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
from osgeo import gdal


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "data/raw/Landsat/Chennai"
AOI = ROOT / "data/processed/Chennai_Wards_UTM44N_area.gpkg"
GRID = ROOT / "data/processed/Chennai_Grid_30m_Clipped.gpkg"
ST = SCENE / "ST_B10.TIF"
ST_QA = SCENE / "ST_QA.TIF"
QA_PIXEL = SCENE / "QA_PIXEL.TIF"
OUTPUT = ROOT / "data/processed/Chennai_LST_Celsius_QA.tif"
MASK_OUTPUT = ROOT / "data/processed/Chennai_LST_ValidMask.tif"

ST_SCALE = 0.00341802
ST_OFFSET_K = 149.0
ST_QA_SCALE_K = 0.01
NODATA_FLOAT = -9999.0


def analysis_grid_bounds() -> tuple[float, float, float, float, int, int]:
    connection = sqlite3.connect(GRID)
    try:
        layer_name = connection.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
        ).fetchone()[0]
        left, bottom, right, top = connection.execute(
            f'SELECT MIN("left"), MIN("bottom"), MAX("right"), MAX("top") '
            f'FROM "{layer_name}"'
        ).fetchone()
    finally:
        connection.close()
    width = round((right - left) / 30.0)
    height = round((top - bottom) / 30.0)
    if abs((right - left) - width * 30.0) > 0.01 or abs((top - bottom) - height * 30.0) > 0.01:
        raise ValueError("Master grid cell bounds are not a regular 30 m grid")
    return left, bottom, right, top, width, height


def cut_to_aoi(
    source: Path, nodata: float,
    bounds: tuple[float, float, float, float, int, int],
) -> gdal.Dataset:
    left, bottom, right, top, width, height = bounds
    result = gdal.Warp(
        "", str(source), format="MEM", cutlineDSName=str(AOI),
        outputBounds=(left, bottom, right, top), width=width, height=height,
        dstNodata=nodata, resampleAlg="near",
    )
    if result is None:
        raise RuntimeError(f"Could not clip {source.name} to AOI")
    return result


def write_single_band(
    path: Path, array: np.ndarray, reference: gdal.Dataset, data_type: int,
    nodata: float, description: str, tags: dict[str, str],
) -> None:
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        str(path), reference.RasterXSize, reference.RasterYSize, 1, data_type,
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    if out is None:
        raise RuntimeError(f"Could not create {path}")
    out.SetGeoTransform(reference.GetGeoTransform())
    out.SetProjection(reference.GetProjection())
    band = out.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.SetDescription(description)
    band.WriteArray(array)
    band.SetMetadata(tags)
    band.FlushCache()
    out.FlushCache()
    out = None


def main() -> None:
    for path in (ST, ST_QA, QA_PIXEL, AOI, GRID):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists() or MASK_OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite an existing QA output")

    bounds = analysis_grid_bounds()
    st_ds = cut_to_aoi(ST, 0, bounds)
    stqa_ds = cut_to_aoi(ST_QA, -9999, bounds)
    qa_ds = cut_to_aoi(QA_PIXEL, 0, bounds)
    datasets = (st_ds, stqa_ds, qa_ds)
    grid = (st_ds.RasterXSize, st_ds.RasterYSize, st_ds.GetGeoTransform(), st_ds.GetProjection())
    for label, ds in zip(("ST_QA", "QA_PIXEL"), datasets[1:]):
        candidate = (ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform(), ds.GetProjection())
        if candidate != grid:
            raise ValueError(f"{label} clipped grid does not match ST_B10")

    raw = st_ds.GetRasterBand(1).ReadAsArray().astype(np.uint16, copy=False)
    qa = qa_ds.GetRasterBand(1).ReadAsArray().astype(np.uint16, copy=False)
    stqa = stqa_ds.GetRasterBand(1).ReadAsArray().astype(np.int16, copy=False)

    valid = (raw != 0)
    # QA_PIXEL: fill, dilated cloud, cirrus, cloud, cloud shadow, snow.
    for bit in range(6):
        valid &= (qa & (1 << bit)) == 0
    valid &= stqa != -9999

    lst_c = raw.astype(np.float32) * ST_SCALE + ST_OFFSET_K - 273.15
    valid &= np.isfinite(lst_c) & (lst_c > -80.0) & (lst_c < 80.0)
    if not valid.any():
        raise ValueError("No valid pixels remain; check product pairing, AOI, and QA flags")

    lst_out = np.where(valid, lst_c, NODATA_FLOAT).astype(np.float32)
    valid_out = valid.astype(np.uint8)
    uncertainty_k = stqa[valid].astype(np.float32) * ST_QA_SCALE_K
    tags = {
        "units": "degrees_Celsius",
        "source": "Landsat Collection 2 Level-2 ST_B10",
        "scale": str(ST_SCALE),
        "offset_kelvin": str(ST_OFFSET_K),
        "QA_PIXEL_rejected_bits": "0,1,2,3,4,5",
        "ST_QA_scale_kelvin": str(ST_QA_SCALE_K),
        "ST_QA_cutoff": "not_applied; review uncertainty distribution",
        "AOI": AOI.name,
        "analysis_grid": GRID.name,
        "pixel_size_m": "30",
    }
    write_single_band(
        OUTPUT, lst_out, st_ds, gdal.GDT_Float32, NODATA_FLOAT,
        "Landsat surface temperature (degrees Celsius)", tags,
    )
    write_single_band(
        MASK_OUTPUT, valid_out, st_ds, gdal.GDT_Byte, 255,
        "1=valid, 0=masked", {"source": "ST_B10 + ST_QA + QA_PIXEL"},
    )

    print(f"Wrote {OUTPUT}")
    print(f"Wrote {MASK_OUTPUT}")
    populated = int((raw != 0).sum())
    print(
        f"Valid pixels: {int(valid.sum()):,}; "
        f"{int(valid.sum()) / populated:.1%} of ST_B10-populated AOI/grid pixels; "
        f"output grid {st_ds.RasterXSize} x {st_ds.RasterYSize}"
    )
    print(f"LST range (C): {float(lst_c[valid].min()):.2f} to {float(lst_c[valid].max()):.2f}")
    print(
        "ST_QA uncertainty in retained pixels (K), p50/p90/max: "
        f"{float(np.percentile(uncertainty_k, 50)):.2f}/"
        f"{float(np.percentile(uncertainty_k, 90)):.2f}/"
        f"{float(uncertainty_k.max()):.2f}"
    )


if __name__ == "__main__":
    gdal.UseExceptions()
    main()
