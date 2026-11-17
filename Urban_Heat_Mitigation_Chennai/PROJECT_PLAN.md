# Chennai project adaptation

The original Nagpur folder remains unchanged. This project retains its analysis workflow and folder structure, with city-specific names, AOI, paths, and source-data settings adapted to Greater Chennai Corporation (GCC).

## Chennai-specific setup

- Download GCC ward polygons from the public GCC_AdminBoundary Ward_Boundary layer and use those polygons as the AOI and ward layer.
- Use EPSG:32644 and the existing 30 m grid resolution.
- Collect Chennai ERA5, Landsat, CPCB, OSM, and GHSL R8_C26 population inputs.
- Run phases 1 through 6 after the required inputs have been collected.

The GCC ward-service layer is current public GIS data. Check its boundary vintage against current delimitation records and verify reuse terms. The project copy contains no Nagpur raw measurements, processed layers, or phase outputs.
