"""Fetch an OpenStreetMap polygon for the Nagpur Municipal Corporation AOI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
NOMINATIM = "https://nominatim.openstreetmap.org"
OVERPASS = "https://overpass-api.de/api/interpreter"
OUTPUT = ROOT / "data/raw/Wards.geojson"
METADATA = ROOT / "data/raw/Nagpur_AOI_source.json"
USER_AGENT = "UrbanHeatMitigation-Nagpur/1.0 (Nagpur Municipal Corporation AOI data retrieval)"


def get_json(url: str, *, data: bytes | None = None) -> dict:
    request = Request(url, data=data, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json,application/json",
        **({"Content-Type": "application/x-www-form-urlencoded"} if data else {}),
    })
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def polygonal(feature: dict) -> bool:
    return feature.get("geometry", {}).get("type") in {"Polygon", "MultiPolygon"}


def find_nominatim_polygon(features: list[dict]) -> dict | None:
    candidates = []
    for feature in features:
        if not polygonal(feature):
            continue
        props = feature.get("properties", {})
        name = str(props.get("name", "")).casefold()
        display = str(props.get("display_name", "")).casefold()
        category = str(props.get("category", props.get("class", ""))).casefold()
        kind = str(props.get("type", "")).casefold()
        if "nagpur" not in (name + " " + display):
            continue
        # A place-name search can return the NMC headquarters building, whose
        # name matches but whose polygon is only a few hundred metres wide.
        # Accept only administrative boundary relations for the AOI.
        if (props.get("osm_type") != "relation"
                or str(props.get("category", props.get("class", ""))).casefold() != "boundary"
                or str(props.get("type", "")).casefold() != "administrative"):
            continue
        score = 0
        if "nagpur municipal corporation" in name or "nagpur municipal corporation" in display:
            score += 100
        if "municipal corporation" in name:
            score += 50
        if category == "boundary" and kind == "administrative":
            score += 20
        score += 5
        candidates.append((score, feature))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def get_area_via_relation() -> dict | None:
    # Nominatim's text search can favor city-centre points over boundary areas.
    # Ask Overpass for candidate administrative relations, then ask Nominatim
    # for each relation's assembled polygon geometry.
    query = """[out:json][timeout:60];
(
  relation["boundary"="administrative"]["name"~"Nagpur",i](20.90,78.80,21.40,79.45);
);
out tags;"""
    raw = get_json(OVERPASS, data=urlencode({"data": query}).encode("utf-8"))
    candidates = []
    for element in raw.get("elements", []):
        tags = element.get("tags", {})
        name = str(tags.get("name", "")).casefold()
        level = str(tags.get("admin_level", ""))
        score = 0
        if "municipal corporation" in name:
            score += 100
        elif name == "nagpur" and level in {"8", "9"}:
            score += 50
        elif "nagpur" in name and level in {"8", "9"}:
            score += 30
        if score:
            candidates.append((score, element["id"], tags))
    candidates.sort(reverse=True, key=lambda item: item[0])
    for _, relation_id, tags in candidates:
        params = urlencode({
            "osm_ids": f"R{relation_id}", "format": "geojson",
            "polygon_geojson": 1, "addressdetails": 1, "extratags": 1,
        })
        result = get_json(f"{NOMINATIM}/lookup?{params}")
        feature = find_nominatim_polygon(result.get("features", []))
        if feature:
            props = feature.setdefault("properties", {})
            props.setdefault("name", tags.get("name", "Nagpur Municipal Corporation"))
            props["osm_admin_level"] = tags.get("admin_level")
            return feature
    return None


def main() -> None:
    # Reuse a previously downloaded administrative polygon. This makes the
    # AOI step repeatable when the public lookup services are temporarily down.
    if OUTPUT.is_file():
        try:
            cached = json.loads(OUTPUT.read_text(encoding="utf-8"))
            features = cached.get("features", [])
            props = features[0].get("properties", {}) if features else {}
            if (features and polygonal(features[0])
                    and props.get("osm_type") == "relation"
                    and str(props.get("category", "")).casefold() == "boundary"
                    and str(props.get("type", "")).casefold() == "administrative"
                    and "nagpur" in str(props.get("name", "")).casefold()):
                print(f"Using saved administrative boundary: {OUTPUT}")
                print(f"Boundary feature: {props.get('name')} | OSM relation {props.get('osm_id')}")
                print("No network refresh was attempted; verify the saved outline matches the NMC limit.")
                return
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            pass

    params = urlencode({
        "q": "Nagpur Municipal Corporation, Nagpur, Maharashtra, India",
        "format": "geojson", "polygon_geojson": 1, "addressdetails": 1,
        "extratags": 1, "namedetails": 1, "limit": 20,
    })
    search = get_json(f"{NOMINATIM}/search?{params}")
    feature = find_nominatim_polygon(search.get("features", []))
    if feature is None:
        try:
            feature = get_area_via_relation()
        except Exception as exc:
            raise RuntimeError(
                "OpenStreetMap could not be reached for the Nagpur boundary. "
                "Retry when the service is available; any saved AOI file was left untouched."
            ) from exc
    if feature is None:
        raise RuntimeError(
            "Could not find a Nagpur administrative boundary polygon in OpenStreetMap. "
            "No file was written; use a verified NMC boundary GeoJSON as data/raw/Wards.geojson."
        )

    document = {"type": "FeatureCollection", "features": [feature]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document), encoding="utf-8")
    props = feature.get("properties", {})
    METADATA.write_text(json.dumps({
        "source": "OpenStreetMap via Nominatim and Overpass API",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "osm_type": props.get("osm_type"),
        "osm_id": props.get("osm_id"),
        "name": props.get("name"),
        "display_name": props.get("display_name"),
        "admin_level": props.get("osm_admin_level"),
        "license": "Open Database License (ODbL); attribute OpenStreetMap contributors",
        "note": "Verify this mapped outline against the current Nagpur Municipal Corporation limit before analysis.",
    }, indent=2), encoding="utf-8")
    print(f"Wrote municipal boundary source: {OUTPUT}")
    print(f"Boundary feature: {props.get('name')} | OSM {props.get('osm_type')} {props.get('osm_id')}")
    print(f"Wrote source metadata: {METADATA}")


if __name__ == "__main__":
    main()
