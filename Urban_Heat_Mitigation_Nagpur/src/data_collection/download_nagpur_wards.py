"""Fetch Nagpur's 38 ward polygons from the BharatViz ward-boundary dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/raw/Nagpur_Wards.geojson"
METADATA = ROOT / "data/raw/Nagpur_Wards_source.json"
SOURCE_URL = (
    "https://gist.githubusercontent.com/saketkc/"
    "081d56fd97703f70c578ff87de677ebc/raw/nagpur.geojson"
)
SOURCE_PAGE = "https://gist.github.com/saketkc/081d56fd97703f70c578ff87de677ebc"


def inspect(document: dict) -> tuple[list[dict], int]:
    features = [
        feature for feature in document.get("features", [])
        if feature.get("geometry", {}).get("type") in {"Polygon", "MultiPolygon"}
    ]
    names = {
        str((feature.get("properties") or {}).get("ward_name") or
            (feature.get("properties") or {}).get("wardname") or "").strip()
        for feature in features
    }
    names.discard("")
    if len(features) < 30 or len(names) < 30:
        raise RuntimeError(
            f"Ward file validation failed: {len(features)} polygons and "
            f"{len(names)} named wards; expected about 38 wards."
        )
    return features, len(names)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.is_file():
        document = json.loads(OUTPUT.read_text(encoding="utf-8"))
        features, ward_count = inspect(document)
        print(f"Using saved Nagpur ward polygons: {OUTPUT}")
    else:
        request = Request(SOURCE_URL, headers={
            "User-Agent": "UrbanHeatMitigation-Nagpur/1.0",
            "Accept": "application/geo+json,application/json",
        })
        with urlopen(request, timeout=120) as response:
            document = json.loads(response.read().decode("utf-8"))
        features, ward_count = inspect(document)
        OUTPUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}),
                          encoding="utf-8")

    metadata = {
        "source": "BharatViz city ward boundaries (community-maintained GeoJSON)",
        "source_page": SOURCE_PAGE,
        "download_url": SOURCE_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "polygon_feature_count": len(features),
        "unique_ward_count": ward_count,
        "license": "The source page does not state a license; verify reuse terms with the publisher.",
        "note": "Compare the boundary vintage with the NMC ward map before relying on ward-level results.",
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Ward polygons: {len(features)} features across {ward_count} named wards")
    print(f"Wrote source metadata: {METADATA}")


if __name__ == "__main__":
    main()
