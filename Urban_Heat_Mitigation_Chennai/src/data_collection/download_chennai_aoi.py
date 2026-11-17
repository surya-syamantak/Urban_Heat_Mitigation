"""Download the official Greater Chennai Corporation ward boundaries for the AOI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
LAYER = "https://gisgcc.chennaicorporation.gov.in/server/rest/services/GCCPublic/GCC_AdminBoundary/MapServer/4"
QUERY = LAYER + "/query"
OUTPUT = ROOT / "data/raw/Wards.geojson"
METADATA = ROOT / "data/raw/Chennai_AOI_source.json"
USER_AGENT = "UrbanHeatMitigation-Chennai/1.0 (Greater Chennai Corporation AOI data retrieval)"

def fetch() -> dict:
    params = urlencode({"where": "1=1", "outFields": "*", "returnGeometry": "true", "outSR": "4326", "f": "geojson"})
    request = Request(QUERY + "?" + params, headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json,application/json"})
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))

def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.is_file():
        document = json.loads(OUTPUT.read_text(encoding="utf-8"))
    else:
        document = fetch()
    features = [f for f in document.get("features", []) if f.get("geometry", {}).get("type") in {"Polygon", "MultiPolygon"}]
    if not 150 <= len(features) <= 250:
        raise RuntimeError(f"Expected approximately 200 GCC ward polygons; received {len(features)}. No AOI files were written.")
    # Set consistent AOI labels while preserving each official ward identifier.
    for f in features:
        props = f.setdefault("properties", {})
        props["aoi_name"] = "Greater Chennai Corporation (GCC)"
        props["ward_name"] = str(props.get("ward") or props.get("ward_no") or props.get("wardcode") or "").strip()
    OUTPUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    METADATA.write_text(json.dumps({
        "source": "Greater Chennai Corporation GIS public service, GCC_AdminBoundary / Ward_Boundary (layer 4)",
        "service_url": LAYER, "query_url": QUERY, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(features), "aoi_name": "Greater Chennai Corporation (GCC)",
        "source_crs": "EPSG:32644", "output_crs": "EPSG:4326",
        "license": "The service directory does not state a reuse license; verify reuse terms with GCC.",
        "note": "Current GCC GIS ward polygons are the analysis AOI source. Check the boundary vintage against current GCC delimitation records before interpreting ward-level results."
    }, indent=2), encoding="utf-8")
    print(f"Wrote official GCC ward boundary AOI: {OUTPUT}")
    print(f"Feature count: {len(features)}; AOI: Greater Chennai Corporation (GCC)")
    print(f"Wrote source metadata: {METADATA}")

if __name__ == "__main__":
    main()
