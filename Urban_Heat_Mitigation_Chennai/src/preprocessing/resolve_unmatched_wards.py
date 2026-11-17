"""Fill null ward names in the 30 m centroid layer using max cell overlap.

The original point-in-polygon join leaves some centroids unassigned where a
clipped grid-cell centroid falls outside all ward polygons. This creates a new
version and a small audit crosswalk; all source GeoPackages remain unchanged.

Run with the QGIS Python environment from the repository root:
  python-qgis-ltr.bat src/preprocessing/resolve_unmatched_wards.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from qgis.core import QgsApplication, QgsFeatureRequest, QgsSpatialIndex, QgsVectorLayer


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "data/processed/Chennai_Grid_30m_Clipped.gpkg"
CENTROIDS = ROOT / "data/processed/Chennai_Grid_30m_Centroids_Ward_v2.gpkg"
WARDS = ROOT / "data/processed/Chennai_Wards_UTM44N_area.gpkg"
OUTPUT = ROOT / "data/processed/Chennai_Grid_30m_Centroids_Ward_v3.gpkg"
CROSSWALK = ROOT / "data/processed/Ward_Assignment_Crosswalk.csv"


def load_layer(path: Path, name: str) -> QgsVectorLayer:
    layer = QgsVectorLayer(str(path), name, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"Could not open vector layer: {path}")
    return layer


def main() -> None:
    app = QgsApplication([], False)
    app.initQgis()
    try:
        grid = load_layer(GRID, "grid")
        centroids = load_layer(CENTROIDS, "centroids")
        wards = load_layer(WARDS, "wards")
        if not (grid.crs() == centroids.crs() == wards.crs()):
            raise ValueError("Grid, centroids, and wards must share a CRS")
        for layer, field in ((grid, "grid_id"), (centroids, "grid_id"),
                             (centroids, "Name"), (wards, "Name")):
            if layer.fields().indexFromName(field) < 0:
                raise ValueError(f"Missing field {field!r} in {layer.name()}")

        ward_features = list(wards.getFeatures())
        ward_by_fid = {feature.id(): feature for feature in ward_features}
        index = QgsSpatialIndex()
        index.addFeatures(ward_features)
        null_features = []
        for feature in centroids.getFeatures():
            ward_name = feature["Name"]
            is_null = (
                ward_name is None
                or getattr(ward_name, "isNull", lambda: False)()
                or str(ward_name).strip() == ""
            )
            if is_null:
                null_features.append(feature)
        if not null_features:
            print("No null ward names found; no output written.")
            return

        grid_ids = [int(feature["grid_id"]) for feature in null_features]
        expression = '"grid_id" IN (' + ",".join(map(str, grid_ids)) + ")"
        grid_by_id = {
            int(feature["grid_id"]): feature
            for feature in grid.getFeatures(QgsFeatureRequest().setFilterExpression(expression))
        }

        assignments: list[tuple[int, str, float]] = []
        for centroid in null_features:
            grid_id = int(centroid["grid_id"])
            cell = grid_by_id.get(grid_id)
            if cell is None:
                raise ValueError(f"No clipped grid cell for grid_id={grid_id}")
            cell_geom = cell.geometry()
            if not cell_geom or cell_geom.isEmpty():
                raise ValueError(f"Empty grid cell for grid_id={grid_id}")

            best_name = None
            best_area = 0.0
            for ward_fid in index.intersects(cell_geom.boundingBox()):
                ward = ward_by_fid[ward_fid]
                ward_geom = ward.geometry()
                if not ward_geom or ward_geom.isEmpty():
                    continue
                overlap = cell_geom.intersection(ward_geom)
                if overlap and not overlap.isEmpty():
                    overlap_area = overlap.area()
                    if overlap_area > best_area:
                        best_area = overlap_area
                        best_name = str(ward["Name"])

            if best_name is None:
                raise ValueError(f"No ward overlap found for grid_id={grid_id}")
            assignments.append((grid_id, best_name, best_area / cell_geom.area()))

        if OUTPUT.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
        import shutil
        shutil.copy2(CENTROIDS, OUTPUT)
        output_layer = load_layer(OUTPUT, "centroids_v3")
        if not output_layer.startEditing():
            raise RuntimeError("Could not start editing the copied centroid GeoPackage")
        ward_field = output_layer.fields().indexFromName("Name")
        feature_by_grid_id = {
            int(feature["grid_id"]): feature
            for feature in output_layer.getFeatures(
                QgsFeatureRequest().setFilterExpression(expression)
            )
        }
        for grid_id, ward_name, _ in assignments:
            output_feature = feature_by_grid_id.get(grid_id)
            if output_feature is None:
                raise ValueError(f"Missing copied centroid for grid_id={grid_id}")
            if not output_layer.changeAttributeValue(
                output_feature.id(), ward_field, ward_name
            ):
                raise RuntimeError(f"Could not assign ward for grid_id={grid_id}")
        if not output_layer.commitChanges():
            raise RuntimeError("Could not commit centroid ward assignments")
        output_layer.reload()
        feature_count = output_layer.featureCount()
        null_count = sum(
            feature["Name"] is None
            or getattr(feature["Name"], "isNull", lambda: False)()
            or str(feature["Name"]).strip() == ""
            for feature in output_layer.getFeatures()
        )
        ward_count = len({str(feature["Name"]) for feature in output_layer.getFeatures()})

        with CROSSWALK.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["grid_id", "assigned_ward", "cell_area_fraction_in_ward", "method"])
            writer.writerows(
                (grid_id, name, f"{fraction:.6f}", "maximum polygon overlap")
                for grid_id, name, fraction in assignments
            )

        if null_count != 0 or feature_count != centroids.featureCount() or ward_count != wards.featureCount():
            raise RuntimeError(
                "Output QA failed: "
                f"features={feature_count}, null ward names={null_count}, wards={ward_count}"
            )
        print(f"Wrote {OUTPUT}")
        print(f"Wrote {CROSSWALK}")
        print(
            f"Filled {len(assignments)} null names; {feature_count:,} cells, "
            f"{ward_count} wards, 0 null names."
        )
    finally:
        app.exitQgis()


if __name__ == "__main__":
    main()
