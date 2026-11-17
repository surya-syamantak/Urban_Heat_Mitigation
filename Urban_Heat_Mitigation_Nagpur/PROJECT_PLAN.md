# Nagpur project adaptation status

The Ahmedabad source project remains unchanged. This copy retains the phase analysis scripts and method structure, with Nagpur city names, paths, coordinates, and UTM zone 44N settings.

## Before running the analysis

- Obtain and verify the Nagpur Municipal Corporation AOI geometry.
- Create the projected AOI layer and 30 m grid expected by `config.yaml`.
- Confirm the configured CPCB archive site ID and the downloaded AOI boundary.
- Collect Nagpur ERA5, Landsat, CPCB, and OSM observations, and download GHSL tile R7_C26.
- Run data collection and each analysis phase, then review the resulting QA outputs.

The copy contains no Ahmedabad raw, processed, or output datasets. The original phase code assumes that the prepared spatial layers and required observations already exist; replacing filenames and city labels does not create those Nagpur source layers or results.

## AOI source lead

The AOI downloader searches OpenStreetMap through Nominatim, with an Overpass relation lookup fallback. Confirm the mapped outline against current NMC boundary records before analysis.

The layer is a source lead and must be verified against the current NMC limit before analysis.
