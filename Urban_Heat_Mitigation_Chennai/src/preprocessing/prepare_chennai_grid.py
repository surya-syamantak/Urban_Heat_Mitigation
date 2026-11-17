"""Project the Chennai AOI and create the canonical 30 m analysis grid."""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from osgeo import ogr, osr


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def delete_gpkg(path: Path) -> None:
    if path.exists():
        path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            sidecar.unlink()


def create_layer(ds, name, geom_type, srs, fields):
    layer = ds.CreateLayer(name, srs=srs, geom_type=geom_type, options=["SPATIAL_INDEX=YES"])
    if layer is None:
        raise RuntimeError(f"Could not create layer {name}")
    for field_name, field_type, width in fields:
        field = ogr.FieldDefn(field_name, field_type)
        if width:
            field.SetWidth(width)
        if layer.CreateField(field) != 0:
            raise RuntimeError(f"Could not create field {field_name}")
    return layer


def main() -> None:
    source_path = ROOT / "data/raw/Wards.geojson"
    ward_source_path = ROOT / "data/raw/Chennai_Wards.geojson"
    if not source_path.is_file():
        raise FileNotFoundError(f"Run `python main.py aoi` first: {source_path}")
    if not ward_source_path.is_file():
        raise FileNotFoundError(f"Run `python main.py wards` first: {ward_source_path}")

    target_epsg = int(CONFIG["spatial"]["crs"].split(":")[1])
    resolution = float(CONFIG["spatial"]["resolution_m"])
    name = CONFIG["project"]["aoi_name"]
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    source = ogr.Open(str(source_path))
    if source is None:
        raise RuntimeError(f"GDAL could not open {source_path}")
    src_layer = source.GetLayer(0)
    src_srs = src_layer.GetSpatialRef()
    if src_srs is None:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(4326)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src_srs, target_srs)
    geometries = []
    for feature in src_layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        projected = geom.Clone()
        if projected.Transform(transform) != 0:
            raise RuntimeError("Could not reproject an AOI feature")
        if projected.IsValid() == 0:
            projected = projected.MakeValid()
        if projected is not None and not projected.IsEmpty():
            geometries.append(projected)
    source = None
    if not geometries:
        raise RuntimeError("No valid polygon features found in data/raw/Wards.geojson")
    aoi = geometries[0]
    for geom in geometries[1:]:
        merged = aoi.Union(geom)
        if merged is None:
            raise RuntimeError("Could not dissolve AOI polygons")
        aoi = merged
    if aoi.IsValid() == 0:
        aoi = aoi.MakeValid()
    osm_aoi = aoi

    # Load individual wards, reproject them, clip to the municipal AOI, and
    # dissolve split polygons that share one ward name.
    ward_source = ogr.Open(str(ward_source_path))
    if ward_source is None:
        raise RuntimeError(f"GDAL could not open ward polygons: {ward_source_path}")
    ward_source_layer = ward_source.GetLayer(0)
    ward_source_srs = ward_source_layer.GetSpatialRef()
    if ward_source_srs is None:
        ward_source_srs = osr.SpatialReference()
        ward_source_srs.ImportFromEPSG(4326)
    ward_source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ward_transform = osr.CoordinateTransformation(ward_source_srs, target_srs)
    ward_geometries: dict[str, object] = {}
    for feature in ward_source_layer:
        props = feature.items()
        raw_name = (
            props.get("ward_name") or props.get("wardname")
            or props.get("ward_number") or props.get("wardcode") or props.get("ward")
        )
        if raw_name is None or not str(raw_name).strip():
            continue
        raw_name = str(raw_name).strip()
        ward_name = raw_name if "ward" in raw_name.casefold() else f"Ward {raw_name}"
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        projected = geometry.Clone()
        if projected.Transform(ward_transform) != 0:
            raise RuntimeError(f"Could not reproject {ward_name}")
        if projected.IsValid() == 0:
            projected = projected.MakeValid()
        if projected is None or projected.IsEmpty() or projected.GetArea() <= 0:
            continue
        if ward_name in ward_geometries:
            ward_geometries[ward_name] = ward_geometries[ward_name].Union(projected)
        else:
            ward_geometries[ward_name] = projected
    ward_source = None
    ward_geometries = {
        name: geom for name, geom in ward_geometries.items()
        if geom is not None and not geom.IsEmpty() and geom.GetArea() > 0
    }
    if not 150 <= len(ward_geometries) <= 250:
        raise RuntimeError(
            f"Expected approximately 200 GCC wards, found {len(ward_geometries)} "
            f"inside the AOI. Inspect {ward_source_path} and the AOI before continuing."
        )
    ward_union = None
    for geom in ward_geometries.values():
        ward_union = geom.Clone() if ward_union is None else ward_union.Union(geom)
    osm_overlap = ward_union.Intersection(osm_aoi).GetArea() / ward_union.GetArea()
    if osm_overlap < 0.80:
        raise RuntimeError(
            f"Only {osm_overlap:.1%} of the ward-derived AOI overlaps the cached OSM "
            "city polygon. Inspect both sources before proceeding."
        )
    # The union of the ward polygons defines the analysis AOI. This keeps the
    # grid, ward labels, and city outline on one consistent boundary source.
    aoi = ward_union
    area_km2 = aoi.GetArea() / 1_000_000.0
    if area_km2 < 100:
        raise RuntimeError(f"Ward-derived Chennai AOI is only {area_km2:.1f} km²")
    xmin, xmax, ymin, ymax = aoi.GetEnvelope()

    processed = ROOT / CONFIG["paths"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)
    wards_path = ROOT / CONFIG["spatial"]["wards"]
    grid_path = ROOT / CONFIG["spatial"]["grid"]
    centroid_path = ROOT / CONFIG["spatial"]["centroid_ward_table"]
    for path in (wards_path, grid_path, centroid_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        delete_gpkg(path)
    driver = ogr.GetDriverByName("GPKG")

    wards_ds = driver.CreateDataSource(str(wards_path))
    wards = create_layer(wards_ds, "ward_boundaries", ogr.wkbUnknown, target_srs, [
        ("Name", ogr.OFTString, 120), ("ward_no", ogr.OFTInteger, 0),
        ("area_km2", ogr.OFTReal, 0),
    ])
    for ward_name, ward_geom in sorted(ward_geometries.items()):
        ward_feature = ogr.Feature(wards.GetLayerDefn())
        ward_feature.SetField("Name", ward_name)
        digits = "".join(ch for ch in ward_name if ch.isdigit())
        if digits:
            ward_feature.SetField("ward_no", int(digits))
        ward_feature.SetField("area_km2", ward_geom.GetArea() / 1_000_000.0)
        ward_feature.SetGeometry(ward_geom)
        if wards.CreateFeature(ward_feature) != 0:
            raise RuntimeError(f"Could not write {ward_name}")
        ward_feature = None
    wards_ds = None

    ward_records = []
    for ward_name, ward_geom in ward_geometries.items():
        envelope = ward_geom.GetEnvelope()
        ward_records.append((ward_name, ward_geom, envelope))

    def assign_ward(cell, cx: float, cy: float) -> str:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(cx, cy)
        matches = []
        for ward_name, ward_geom, (wx0, wx1, wy0, wy1) in ward_records:
            if wx0 <= cx <= wx1 and wy0 <= cy <= wy1 and ward_geom.Intersects(point):
                matches.append(ward_name)
        if matches:
            return sorted(matches)[0]
        # Boundary cells may have their centroid just across a ward edge.
        # Assign those cells to the ward with greatest cell overlap.
        left, right, bottom, top = cell.GetEnvelope()
        best_name, best_area = None, 0.0
        for ward_name, ward_geom, (wx0, wx1, wy0, wy1) in ward_records:
            if wx1 < left or wx0 > right or wy1 < bottom or wy0 > top:
                continue
            overlap = cell.Intersection(ward_geom)
            overlap_area = overlap.GetArea() if overlap is not None else 0.0
            if overlap_area > best_area:
                best_name, best_area = ward_name, overlap_area
        if best_name is None:
            nearest_name, nearest_distance = min(
                ((ward_name, ward_geom.Distance(point))
                 for ward_name, ward_geom, _ in ward_records),
                key=lambda item: item[1],
            )
            if nearest_distance > 1000:
                raise RuntimeError(
                    f"Grid cell centroid is {nearest_distance:.0f} m from the nearest ward; "
                    "the ward and AOI boundaries may not match."
                )
            return nearest_name
        return best_name

    grid_ds = driver.CreateDataSource(str(grid_path))
    grid = create_layer(grid_ds, "grid", ogr.wkbPolygon, target_srs, [
        ("grid_id", ogr.OFTInteger64, 0), ("row_index", ogr.OFTInteger, 0),
        ("col_index", ogr.OFTInteger, 0), ("left", ogr.OFTReal, 0),
        ("right", ogr.OFTReal, 0), ("top", ogr.OFTReal, 0),
        ("bottom", ogr.OFTReal, 0), ("area_km2", ogr.OFTReal, 0),
    ])
    cent_ds = driver.CreateDataSource(str(centroid_path))
    centroids = create_layer(cent_ds, "centroids", ogr.wkbPoint, target_srs, [
        ("grid_id", ogr.OFTInteger64, 0), ("Name", ogr.OFTString, 120),
        ("area_km2", ogr.OFTReal, 0),
    ])

    left0 = math.floor(xmin / resolution) * resolution
    bottom0 = math.floor(ymin / resolution) * resolution
    right0 = math.ceil(xmax / resolution) * resolution
    top0 = math.ceil(ymax / resolution) * resolution
    ncols = int(round((right0 - left0) / resolution))
    nrows = int(round((top0 - bottom0) / resolution))
    grid.StartTransaction()
    centroids.StartTransaction()
    grid_id = 0
    for row in range(nrows):
        # GDAL raster arrays are indexed from the north/top edge downward;
        # keep row_index in that same convention for Phase 2 pixel sampling.
        top = top0 - row * resolution
        bottom = top - resolution
        for col in range(ncols):
            left = left0 + col * resolution
            right = left + resolution
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in ((left, bottom), (right, bottom), (right, top),
                         (left, top), (left, bottom)):
                ring.AddPoint_2D(x, y)
            cell = ogr.Geometry(ogr.wkbPolygon)
            cell.AddGeometry(ring)
            if not cell.Intersects(aoi):
                continue
            grid_id += 1
            gf = ogr.Feature(grid.GetLayerDefn())
            for key, val in (("grid_id", grid_id), ("row_index", row),
                             ("col_index", col), ("left", left), ("right", right),
                             ("top", top), ("bottom", bottom),
                             ("area_km2", resolution * resolution / 1_000_000)):
                gf.SetField(key, val)
            gf.SetGeometry(cell)
            if grid.CreateFeature(gf) != 0:
                raise RuntimeError(f"Could not write grid cell {grid_id}")
            gf = None
            point = ogr.Geometry(ogr.wkbPoint)
            cx, cy = (left + right) / 2, (bottom + top) / 2
            point.AddPoint_2D(cx, cy)
            cf = ogr.Feature(centroids.GetLayerDefn())
            cf.SetField("grid_id", grid_id)
            cf.SetField("Name", assign_ward(cell, cx, cy))
            cf.SetField("area_km2", resolution * resolution / 1_000_000)
            cf.SetGeometry(point)
            if centroids.CreateFeature(cf) != 0:
                raise RuntimeError(f"Could not write centroid {grid_id}")
            cf = None
            if grid_id % 5000 == 0:
                grid.CommitTransaction()
                centroids.CommitTransaction()
                grid.StartTransaction()
                centroids.StartTransaction()
    grid.CommitTransaction()
    centroids.CommitTransaction()
    grid_ds = None
    cent_ds = None
    print(f"Projected AOI: {wards_path}")
    print(f"AOI area: {area_km2:.1f} km²")
    print(f"Ward polygons: {len(ward_geometries)}; overlap with cached OSM outline: {osm_overlap:.1%}")
    print(f"Grid cells: {grid_id} at {resolution:g} m; CRS EPSG:{target_epsg}")
    print(f"Grid: {grid_path}")
    print(f"Centroid lookup: {centroid_path}")


if __name__ == "__main__":
    main()
