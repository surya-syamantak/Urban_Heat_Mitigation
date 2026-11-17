# Nagpur data layout

No Ahmedabad observations or generated Ahmedabad layers were copied into this project. The GHSL tile-index files are global reference data; the Nagpur covering tile is identified as R7_C26 in `config.yaml`.

Use `python main.py aoi` to fetch the municipal-outline source into `raw/Wards.geojson`, then `python main.py prepare_grid` to make the projected AOI, 30 m grid, and centroid/AOI lookup in `processed/`. Run the existing collection and analysis phases to populate the remaining directories. The public boundary source should be checked against the current NMC limits before use.
