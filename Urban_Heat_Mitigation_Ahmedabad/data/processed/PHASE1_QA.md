# Phase 1: data engineering and QA

## Status

The AOI/grid, Landsat, ERA5, CPCB, and OSM preprocessing deliverables are built.
Population was not required by Phase 1; the GHSL population tile is downloaded
and used as an exposure input by Phase 2. The AOI and ward crosswalk still need visual sign-off.
The available CPCB data have station/variable gaps and are not treated as a
full validation dataset.

## Study design and canonical geography

- City: Ahmedabad, Gujarat; 48 municipal wards.
- Period: 2021-01-01 to 2025-12-31. Recurring hot season: March-May; cool
  season: December-February. STAC selection yielded observations in both
  seasons for each of the five study years.
- Coordinate reference: WGS 84 / UTM zone 43N (EPSG:32643); 30 m grid.
- The ward layer has 48 valid polygons and sums to 441.058 km2. The clipped
  grid has 492,386 unique cells. Raster outputs use its 30 m origin/alignment.
- `Ahmedabad_Grid_30m_Centroids_Ward_v3.gpkg` has 492,386 records, 48 ward
  labels, unique grid IDs, and no missing labels. Its crosswalk documents 374
  maximum-overlap assignments; four cells have under 50% overlap with the
  assigned ward and need visual review.

## Landsat scenes, QA, and outputs

Twenty USGS Landsat Collection 2 Level-2 scenes (Landsat 8/9, WRS 148/044;
two hot and two cool acquisitions per year from 2021 through 2025) are listed
with product IDs, UTC acquisition times, cloud metadata, valid-pixel
counts, LST statistics, and ST_QA summaries in
`Landsat/Landsat_scene_manifest.csv`. Per-scene ST, QA mask, NDVI, NDBI, MNDWI,
and albedo rasters are stored under `data/processed/Landsat/`; hot and cool
season median rasters and provisional rule-based LULC rasters are also stored
there.

Processing uses QA_PIXEL bits 0-5 (fill, dilated cloud, cirrus, cloud, cloud
shadow, snow), the USGS ST_B10 scale `DN * 0.00341802 + 149 K`, and ST_QA scale
0.01 K. Reflectance uses `DN * 0.0000275 - 0.2`. ST_B10 is already a Level-2
surface-temperature retrieval; no second emissivity correction is applied.
Product STAC JSON and MTL text were saved under
`data/raw/Landsat/Ahmedabad/metadata/`. The resulting product is land-surface
temperature, not near-surface air temperature or human thermal comfort.

Each season has ten observations across the five years. Per-scene QA retention
and ST_QA uncertainty are in the manifest. Seasonal medians pool all observed
years and are descriptive, not year-specific estimates; the rule-based LULC has
not been independently accuracy-assessed.

## Meteorology and urban form

- ERA5 hourly regional background conditions: 43,824 rows from 2021-01-01 to
  2025-12-31, including 2 m temperature, relative humidity, dew point, wind,
  and radiation. Nearest-hour observations are matched to all 20 Landsat
  overpass timestamps in `ERA5_Landsat_Overpass_Matched.csv`; offsets are
  recorded. ERA5's ~0.25-degree grid cannot resolve neighborhood microclimate.
- OSM Overpass raw snapshot: 169,207 ways, server base timestamp
  2026-07-15T15:22:01Z. It is temporally newer than the Landsat scenes.
  Projected building (110,857), road (57,470), mapped green/water polygon (716),
  and waterway line layers are in
  `OSM_Ahmedabad.gpkg`; the aligned per-cell table is
  `urban_form/Ahmedabad_urban_form_30m.csv` (492,386 grid IDs). It contains
  building equivalents/coverage, tagged-height mean and tag coverage, road
  density, distances to mapped green and water (water polygons plus waterways),
  and `svf_proxy`. Nonzero building coverage occurs in 81,419 cells, tagged
  building heights in 23,048, and roads in 204,163.
- Height coverage depends on optional OSM tags. `svf_proxy` is a heuristic from
  tagged height and building coverage, not a geometric 3-D sky-view factor.
  OSM relations are not included; green/water distance means distance to mapped
  way features only. OSM is a current snapshot, not historical 2023 geometry.

## CPCB, population, and remaining gates

Existing CPCB CAAQMS hourly raw files cover nine Ahmedabad stations for 2023
and 2024 (18 station-year files). The matched table contains 72 rows across the
20 Landsat overpasses; eight scenes have at least three complete AT/RH station
pairs, while the remaining scenes use matched ERA5 AT/RH for Phase 2 comfort.
CPCB source timestamps were interpreted as India Standard Time and converted
to UTC; nearest-hour records are within 60 minutes. The CPCB repository returned
404 for unavailable station-years; these gaps are recorded. The download
manifest records local files and requests;
`match_cpcb_overpasses.py` creates the aligned table. GHSL 2020 population data are downloaded for Phase 2 and are documented in
the dataset catalogue. The four low-overlap ward assignments and AOI
need visual review in QGIS.

The full source, resolution, dates, process, and limitations inventory is
`data/dataset_audit.csv`. Keep raw data immutable. The seven-file keep/archive
recommendation from the earlier QA remains: keep the clipped grid, wards, and
the original ST_B10/ST_QA/QA_PIXEL; use centroid/ward v3; archive v2 and the
unmatched subset for audit instead of deleting them before visual review.

## Reproduction

From the project root, run `python main.py phase1` with QGIS 3 Python available
and network access. The pipeline downloads ERA5/Landsat/OSM/CPCB, builds OSM
metrics, and matches CPCB data to the Landsat overpasses. Individual steps can
be run with `python main.py era5`, `landsat`, `osm`, or `cpcb`. Landsat, CPCB
matching, and OSM vector processing use the QGIS Python launcher selected in
`main.py` or `QGIS_PYTHON`.
