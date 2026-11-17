"""Turn the downloaded OSM snapshot into aligned 30 m urban-form predictors.

OSM heights are incomplete, so height and sky-view outputs are explicitly
tag-based proxies. This script creates point-to-grid road/building metrics and
distances to mapped green/water features; it does not claim a full 3-D sky-view
factor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsSpatialIndex,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)


ROOT = Path(__file__).resolve().parents[2]
OSM_JSON = ROOT / "data/raw/OSM/Chennai_overpass_ways.json"
GRID_PATH = ROOT / "data/processed/Chennai_Grid_30m_Clipped.gpkg"
OUTPUT_GPKG = ROOT / "data/processed/OSM_Chennai.gpkg"
OUTPUT_CSV = ROOT / "data/processed/urban_form/Chennai_urban_form_30m.csv"
TARGET_CRS = QgsCoordinateReferenceSystem("EPSG:32644")
SOURCE_CRS = QgsCoordinateReferenceSystem("EPSG:4326")
CELL_AREA_M2 = 900.0


def height_from_tags(tags: dict[str, str]) -> tuple[float | None, str]:
    raw_height = tags.get("height")
    if raw_height:
        try:
            text = raw_height.lower().replace("meters", "").replace("meter", "").replace("m", "").strip()
            height = float(text)
            if 1.0 <= height <= 300.0:
                return height, "height_tag"
        except ValueError:
            pass
    raw_levels = tags.get("building:levels") or tags.get("levels")
    if raw_levels:
        try:
            height = float(raw_levels) * 3.0
            if 1.0 <= height <= 300.0:
                return height, "levels_x_3m"
        except ValueError:
            pass
    return None, "missing"


def numeric_levels(tags: dict[str, str]) -> float:
    raw = tags.get("building:levels") or tags.get("levels") or "0"
    # OSM occasionally stores ranges/lists such as "1,2,3". Keep a stable
    # numeric proxy by using the largest parseable listed level.
    values = []
    for token in str(raw).replace(";", ",").split(","):
        try:
            value = float(token.strip())
            if 0 <= value <= 100:
                values.append(value)
        except ValueError:
            continue
    return max(values) if values else 0.0


def way_geometry(element: dict, polygon: bool) -> QgsGeometry | None:
    coordinates = element.get("geometry") or []
    if len(coordinates) < (4 if polygon else 2):
        return None
    pts = [QgsPointXY(float(p["lon"]), float(p["lat"])) for p in coordinates]
    if polygon:
        if pts[0] != pts[-1]:
            return None
        return QgsGeometry.fromPolygonXY([pts])
    return QgsGeometry.fromPolylineXY(pts)


def export_layer(layer: QgsVectorLayer, path: Path, layer_name: str, *, first: bool) -> None:
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = layer_name
    options.fileEncoding = "UTF-8"
    if not first:
        options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, str(path), QgsProject.instance().transformContext(), options
    )
    if result[0] != QgsVectorFileWriter.NoError:
        raise RuntimeError(f"GeoPackage write failed for {layer_name}: {result}")


def make_layer(name: str, geometry: str, fields: list[tuple[str, str]]) -> QgsVectorLayer:
    layer = QgsVectorLayer(f"{geometry}?crs=EPSG:32644", name, "memory")
    provider = layer.dataProvider()
    from qgis.PyQt.QtCore import QVariant
    types = {"string": QVariant.String, "int": QVariant.LongLong, "double": QVariant.Double}
    from qgis.core import QgsField
    provider.addAttributes([QgsField(field_name, types[field_type]) for field_name, field_type in fields])
    layer.updateFields()
    return layer


def main() -> None:
    if not OSM_JSON.is_file():
        raise FileNotFoundError(f"Run src/data_collection/download_osm.py first: {OSM_JSON}")
    # This is a fixed, derived output produced by this script. Rebuild it so
    # repeated pipeline runs do not retain stale geometry or grid metrics.
    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    raw = json.loads(OSM_JSON.read_text(encoding="utf-8"))
    transform = QgsCoordinateTransform(SOURCE_CRS, TARGET_CRS, QgsProject.instance())
    buildings = make_layer("buildings", "MultiPolygon", [
        ("osm_id", "int"), ("building", "string"), ("levels", "double"),
        ("height_m", "double"), ("height_src", "string"),
    ])
    roads = make_layer("roads", "MultiLineString", [
        ("osm_id", "int"), ("highway", "string"), ("name", "string"),
    ])
    greenwater = make_layer("green_water", "MultiPolygon", [
        ("osm_id", "int"), ("feature", "string"),
    ])
    waterways = make_layer("waterways", "MultiLineString", [
        ("osm_id", "int"), ("feature", "string"),
    ])
    building_index = QgsSpatialIndex()
    road_index = QgsSpatialIndex()
    green_index = QgsSpatialIndex()
    building_geoms: dict[int, tuple[QgsGeometry, float | None]] = {}
    road_geoms: dict[int, QgsGeometry] = {}
    green_geoms: dict[int, QgsGeometry] = {}

    for element in raw.get("elements", []):
        if element.get("type") != "way":
            continue
        tags = element.get("tags", {})
        is_building = "building" in tags
        is_road = "highway" in tags
        is_waterline = tags.get("waterway") in {"river", "canal", "stream"}
        feature = None
        layer = None

        if is_building:
            geometry = way_geometry(element, polygon=True)
            if geometry is None:
                continue
            geometry.transform(transform)
            height_m, height_src = height_from_tags(tags)
            feature = QgsFeature(buildings.fields())
            feature.setGeometry(geometry)
            feature.setAttributes([
                int(element["id"]), str(tags.get("building", "yes")),
                numeric_levels(tags),
                height_m, height_src,
            ])
            layer = buildings
        elif is_road:
            geometry = way_geometry(element, polygon=False)
            if geometry is None:
                continue
            geometry.transform(transform)
            feature = QgsFeature(roads.fields())
            feature.setGeometry(geometry)
            feature.setAttributes([int(element["id"]), str(tags["highway"]), str(tags.get("name", ""))])
            layer = roads
        else:
            green_tags = (
                tags.get("leisure") in {"park", "garden", "nature_reserve"}
                or tags.get("landuse") in {"grass", "forest", "recreation_ground", "cemetery"}
            )
            water_tags = (
                tags.get("natural") in {"water", "wetland"} or "water" in tags or is_waterline
            )
            if not (green_tags or water_tags):
                continue
            polygon = not is_waterline
            geometry = way_geometry(element, polygon=polygon)
            if geometry is None:
                continue
            geometry.transform(transform)
            if polygon:
                feature = QgsFeature(greenwater.fields())
                feature.setGeometry(geometry)
                feature.setAttributes([
                    int(element["id"]), "water" if water_tags else "green",
                ])
                layer = greenwater
            else:
                feature = QgsFeature(waterways.fields())
                feature.setGeometry(geometry)
                feature.setAttributes([int(element["id"]), "water"])
                layer = waterways

        if feature is not None and layer is not None:
            layer.dataProvider().addFeature(feature)

    for layer in (buildings, roads, greenwater, waterways):
        layer.updateExtents()
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    for i, (layer, name) in enumerate(((buildings, "buildings"), (roads, "roads"), (greenwater, "green_water"), (waterways, "waterways"))):
        export_layer(layer, OUTPUT_GPKG, name, first=i == 0)

    # Build fast spatial indexes and per-feature geometry maps from the saved
    # projected layers, then summarize line/polygon overlap by master grid ID.
    buildings = QgsVectorLayer(f"{OUTPUT_GPKG}|layername=buildings", "buildings", "ogr")
    roads = QgsVectorLayer(f"{OUTPUT_GPKG}|layername=roads", "roads", "ogr")
    greenwater = QgsVectorLayer(f"{OUTPUT_GPKG}|layername=green_water", "green_water", "ogr")
    waterways = QgsVectorLayer(f"{OUTPUT_GPKG}|layername=waterways", "waterways", "ogr")
    grid = QgsVectorLayer(str(GRID_PATH), "grid", "ogr")
    if not all(layer.isValid() for layer in (buildings, roads, greenwater, waterways, grid)):
        raise RuntimeError("Could not reopen projected OSM or grid layers")
    grid_features = list(grid.getFeatures())
    grid_index = QgsSpatialIndex()
    grid_index.addFeatures(grid_features)
    fid_to_grid_id = {feature.id(): int(feature["grid_id"]) for feature in grid_features}
    cell_geometry: dict[int, QgsGeometry] = {feature.id(): feature.geometry() for feature in grid_features}
    cell_values = {
        int(feature["grid_id"]): {
            "building_equivalents": 0.0, "building_area_m2": 0.0,
            "height_area_sum": 0.0, "height_known_area_m2": 0.0,
            "road_length_m": 0.0,
        }
        for feature in grid_features
    }

    for feature in buildings.getFeatures():
        geom = feature.geometry()
        if not geom or geom.isEmpty():
            continue
        full_area = max(geom.area(), 1e-6)
        height_value = feature["height_m"]
        try:
            height_m = float(height_value) if height_value is not None else None
            if height_m is not None and height_m <= 0:
                height_m = None
        except (TypeError, ValueError):
            height_m = None
        for cell_fid in grid_index.intersects(geom.boundingBox()):
            overlap = cell_geometry[cell_fid].intersection(geom)
            if not overlap or overlap.isEmpty():
                continue
            overlap_area = overlap.area()
            if overlap_area <= 0:
                continue
            value = cell_values[fid_to_grid_id[cell_fid]]
            value["building_equivalents"] += overlap_area / full_area
            value["building_area_m2"] += overlap_area
            if height_m is not None:
                value["height_area_sum"] += overlap_area * height_m
                value["height_known_area_m2"] += overlap_area

    for feature in roads.getFeatures():
        geom = feature.geometry()
        if not geom or geom.isEmpty():
            continue
        for cell_fid in grid_index.intersects(geom.boundingBox()):
            overlap = cell_geometry[cell_fid].intersection(geom)
            if overlap and not overlap.isEmpty():
                length = overlap.length()
                if length > 0:
                    cell_values[fid_to_grid_id[cell_fid]]["road_length_m"] += length

    # Nearest mapped park/green/water feature distances are calculated from
    # grid-cell centroids. OSM relations are not included by this Overpass query.
    green_features = list(greenwater.getFeatures()) + list(waterways.getFeatures())
    category_indexes = {"green": QgsSpatialIndex(), "water": QgsSpatialIndex()}
    category_features: dict[str, dict[int, QgsFeature]] = {"green": {}, "water": {}}
    synthetic_id = 1
    for feature in green_features:
        category = str(feature["feature"])
        indexed_feature = QgsFeature(feature)
        indexed_feature.setId(synthetic_id)
        category_indexes[category].addFeature(indexed_feature)
        category_features[category][synthetic_id] = indexed_feature
        synthetic_id += 1

    def nearest_distance(point: QgsGeometry, category: str) -> float | None:
        index = category_indexes[category]
        if not category_features[category]:
            return None
        candidates = index.nearestNeighbor(point.asPoint(), 1)
        if not candidates:
            return None
        return point.distance(category_features[category][candidates[0]].geometry())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "grid_id", "building_equivalents_per_ha", "building_coverage_fraction",
        "mean_tagged_building_height_m", "height_tag_area_coverage_fraction",
        "road_length_m", "road_density_km_per_km2", "distance_to_green_m",
        "distance_to_water_m", "svf_proxy",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for feature in grid_features:
            grid_id = int(feature["grid_id"])
            values = cell_values[grid_id]
            height_known_area = values["height_known_area_m2"]
            mean_height = (
                values["height_area_sum"] / height_known_area if height_known_area > 0 else ""
            )
            coverage = min(values["building_area_m2"] / CELL_AREA_M2, 1.0)
            height_factor = float(mean_height) / 30.0 if mean_height != "" else 0.0
            svf_proxy = math.exp(-coverage * height_factor)
            centroid = feature.geometry().centroid()
            d_green = nearest_distance(centroid, "green")
            d_water = nearest_distance(centroid, "water")
            writer.writerow({
                "grid_id": grid_id,
                "building_equivalents_per_ha": values["building_equivalents"] / (CELL_AREA_M2 / 10000.0),
                "building_coverage_fraction": coverage,
                "mean_tagged_building_height_m": mean_height,
                "height_tag_area_coverage_fraction": height_known_area / max(values["building_area_m2"], 1e-6),
                "road_length_m": values["road_length_m"],
                "road_density_km_per_km2": values["road_length_m"] / CELL_AREA_M2 * 1000.0,
                "distance_to_green_m": "" if d_green is None else d_green,
                "distance_to_water_m": "" if d_water is None else d_water,
                "svf_proxy": svf_proxy,
            })
    print(f"Wrote {OUTPUT_GPKG} with buildings, roads, green/water features")
    print(f"Wrote {OUTPUT_CSV} for {len(grid_features):,} grid cells")
    print("SVF column is a building-coverage/height proxy, not a true 3-D sky-view factor.")


if __name__ == "__main__":
    main()
