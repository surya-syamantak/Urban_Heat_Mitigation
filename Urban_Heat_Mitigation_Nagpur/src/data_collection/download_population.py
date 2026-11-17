"""Download and extract the official GHSL GHS-POP tile covering Nagpur."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
POP = CONFIG["population"]
ARCHIVE = ROOT / POP["archive_path"]
RASTER = ROOT / POP["raster_path"]
METADATA = RASTER.with_suffix(".metadata.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace local download")
    args = parser.parse_args()
    RASTER.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if RASTER.is_file() and not args.force:
        print(f"Using existing GHSL tile: {RASTER}")
    else:
        request = Request(POP["download_url"], headers={
            "User-Agent": "UrbanHeatMitigation-Nagpur/1.0",
        })
        with urlopen(request, timeout=300) as response:
            payload = response.read()
        if not payload.startswith(b"PK\x03\x04"):
            raise RuntimeError("GHSL download did not return a ZIP archive")
        fd, temporary_name = tempfile.mkstemp(prefix="ghsl_", suffix=".zip", dir=ARCHIVE.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            temporary.write_bytes(payload)
            temporary.replace(ARCHIVE)
        finally:
            temporary.unlink(missing_ok=True)
        with ZipFile(ARCHIVE) as archive:
            tif_names = [name for name in archive.namelist() if name.lower().endswith(".tif")]
            if len(tif_names) != 1:
                raise RuntimeError(f"Expected one TIFF in GHSL archive; found {tif_names}")
            with archive.open(tif_names[0]) as source, RASTER.open("wb") as target:
                target.write(source.read())
    METADATA.write_text(json.dumps({
        "source": POP["source"],
        "epoch": POP["epoch"],
        "tile_id": POP["tile_id"],
        "resolution_m": POP["resolution_m"],
        "crs": POP["crs"],
        "units": POP["unit"],
        "nodata": POP["nodata"],
        "download_url": POP["download_url"],
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "license": "Creative Commons Attribution 4.0 International",
        "citation": "Schiavina et al. (2023), GHS-POP R2023A, European Commission JRC, DOI: 10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE",
    }, indent=2), encoding="utf-8")
    print(f"Population raster: {RASTER} ({RASTER.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
