"""Fetch current OSM building, road, green-space, and water ways for Nagpur."""

from __future__ import annotations

import json
import time
from http.client import HTTPException
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
OUTPUT = ROOT / "data/raw/OSM/Nagpur_overpass_ways.json"
METADATA = ROOT / "data/raw/OSM/Nagpur_overpass_metadata.json"


def main() -> None:
    osm = CONFIG["osm"]
    south, west, north, east = osm["bbox"]
    bbox = f"{south},{west},{north},{east}"
    query = f"""[out:json][timeout:180];
(
  way[building]({bbox});
  way[highway]({bbox});
  way[leisure~\"park|garden|nature_reserve\"]({bbox});
  way[landuse~\"grass|forest|recreation_ground|cemetery\"]({bbox});
  way[natural~\"water|wetland\"]({bbox});
  way[water]({bbox});
  way[waterway~\"river|canal|stream\"]({bbox});
);
out body geom;"""
    endpoints = [osm["endpoint"], "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
    errors = []
    raw = None
    endpoint_used = None
    for endpoint in dict.fromkeys(endpoints):
        for attempt in range(3):
            request = Request(
                endpoint,
                data=query.encode("utf-8"),
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "User-Agent": "UrbanHeatMitigation-Nagpur/1.0 (research project)",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=300) as response:
                    raw = response.read()
                endpoint_used = endpoint
                break
            except (URLError, HTTPError, HTTPException, TimeoutError, OSError) as exc:
                errors.append(f"{endpoint} (attempt {attempt + 1}/3): {exc}")
                print(f"Overpass request failed: {errors[-1]}", flush=True)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if raw is not None:
            break
    if raw is None:
        raise RuntimeError("All Overpass endpoints failed: " + "; ".join(errors))
    parsed = json.loads(raw.decode("utf-8"))
    elements = parsed.get("elements", [])
    if not elements:
        raise RuntimeError("Overpass returned no features")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(raw)
    METADATA.write_text(json.dumps({
        "source": "OpenStreetMap via Overpass API",
        "endpoint": endpoint_used,
        "bbox_south_west_north_east": osm["bbox"],
        "tags": ["building", "highway", "leisure", "landuse", "natural", "water", "waterway"],
        "snapshot_date_utc": datetime.now(timezone.utc).isoformat(),
        "element_count": len(elements),
        "ways": sum(element.get("type") == "way" for element in elements),
        "historical_status": "current snapshot; not a 2023 extract",
        "query": query,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(elements):,} OSM elements)")
    print(f"Wrote {METADATA}")


if __name__ == "__main__":
    main()
