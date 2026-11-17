"""Select all quality-screened Landsat 8/9 items in configured seasons."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from urllib.request import Request, urlopen


def request_json(url, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": "UrbanHeatMitigation/1.0", "Accept": "application/geo+json"}
    if data is not None:
        headers["Content-Type"] = "application/geo+json"
    req = Request(url, data=data, headers=headers, method=method or ("GET" if data is None else "POST"))
    with urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def all_features(url, payload):
    """Follow STAC pagination so no qualifying acquisition is silently omitted."""
    features = []
    page = request_json(url, payload)
    while True:
        features.extend(page.get("features", []))
        link = next((x for x in page.get("links", []) if x.get("rel") == "next"), None)
        if not link:
            return features
        method = str(link.get("method", "GET")).upper()
        body = link.get("body", payload) if method == "POST" else None
        page = request_json(link["href"], body, method)


def season_for(day, seasons):
    month = date.fromisoformat(day).month
    for name, definition in seasons.items():
        if month in [int(value) for value in definition.get("months", [])]:
            return name
    return None


def selected_item_ids(config):
    study = config["study"]
    if study.get("landsat_items"):
        return list(study["landsat_items"])

    south, west, north, east = config["osm"]["bbox"]
    url = config["landsat"]["stac_api"].rstrip("/") + "/search"
    payload = {
        "collections": [study["landsat_stac_collection"]],
        "bbox": [west, south, east, north],
        "datetime": f"{study['start_date']}T00:00:00Z/{study['end_date']}T23:59:59Z",
        "limit": 100,
    }
    features = all_features(url, payload)
    wanted_path = int(config["landsat"]["wrs_path"])
    wanted_rows = {int(x) for x in config["landsat"]["wrs_rows"]}
    max_cloud = float(study["landsat_max_cloud_cover_percent"])
    by_day = defaultdict(dict)
    skipped_platform = skipped_asset = skipped_cloud = 0

    for feature in features:
        item_id = str(feature.get("id", ""))
        props = feature.get("properties", {})
        platform = str(props.get("platform", "")).lower()
        if not (platform in ("landsat-8", "landsat-9") or item_id.startswith(("LC08_", "LC09_"))):
            skipped_platform += 1
            continue
        if "lwir11" not in feature.get("assets", {}):
            skipped_asset += 1
            continue
        try:
            path = int(props["landsat:wrs_path"])
            row = int(props["landsat:wrs_row"])
            cloud = float(props.get("eo:cloud_cover", 100))
        except (KeyError, TypeError, ValueError):
            continue
        if path != wanted_path or row not in wanted_rows:
            continue
        if cloud > max_cloud:
            skipped_cloud += 1
            continue
        day = str(props.get("datetime", ""))[:10]
        if len(day) != 10 or not (study["start_date"] <= day <= study["end_date"]):
            continue
        if season_for(day, study["seasons"]) is None:
            continue
        by_day[day][item_id] = feature

    if not by_day:
        raise RuntimeError(
            f"No Landsat 8/9 items passed the date, season, WRS, and {max_cloud:g}% cloud filters "
            f"for {config['project']['city']}. STAC returned {len(features)} records."
        )

    selected_dates = sorted(by_day)
    items = [feature for day in selected_dates
             for feature in sorted(by_day[day].values(), key=lambda f: (
                 int(f["properties"].get("landsat:wrs_row", -1)), str(f.get("id", ""))))]
    ids = [str(feature["id"]) for feature in items]
    counts = defaultdict(set)
    for day in selected_dates:
        counts[(day[:4], season_for(day, study["seasons"]))].add(day)
    distribution = ", ".join(f"{year} {season}={len(days)}" for (year, season), days in sorted(counts.items()))
    print(f"{config['project']['city']}: selected all {len(selected_dates)} qualifying acquisition dates "
          f"({len(ids)} WRS item/tile records); {distribution}")
    print(f"Scene filters: Landsat 8/9, configured WRS rows, cloud <= {max_cloud:g}%; "
          "per-pixel QA is applied during processing.")
    print("Dates: " + ", ".join(selected_dates))
    print(f"STAC candidates skipped: platform={skipped_platform}, no thermal asset={skipped_asset}, "
          f"cloud={skipped_cloud}")
    return ids
