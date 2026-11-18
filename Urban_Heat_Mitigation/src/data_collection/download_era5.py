"""Download ERA5 hourly background conditions and align to Landsat overpasses."""

from __future__ import annotations

import csv
import json
import datetime as dt
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
RAW_DIR = ROOT / "data/raw/ERA5"
PROCESSED_DIR = ROOT / "data/processed"
RAW_CSV = RAW_DIR / f"Ahmedabad_ERA5_hourly_{CONFIG['study']['start_date']}_{CONFIG['study']['end_date']}.csv"
MATCHED_CSV = PROCESSED_DIR / "ERA5_Landsat_Overpass_Matched.csv"
META_JSON = RAW_DIR / "Ahmedabad_ERA5_request_metadata.json"


def request_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Urban-Heat-Mitigation/1.0"})
    with urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    study = CONFIG["study"]
    era5 = CONFIG["era5"]
    variables = era5["hourly_variables"]
    params = {
        "latitude": era5["latitude"],
        "longitude": era5["longitude"],
        "start_date": study["start_date"],
        "end_date": study["end_date"],
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "models": era5["model"],
    }
    url = era5["endpoint"] + "?" + urlencode(params)
    response = request_json(url)
    hourly = response["hourly"]
    if not hourly.get("time"):
        raise RuntimeError("ERA5 request returned no hourly timestamps")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["time", *variables]
    with RAW_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row_index, time_value in enumerate(hourly["time"]):
            writer.writerow({
                field: time_value if field == "time" else hourly[field][row_index]
                for field in fields
            })

    metadata = {
        "provider": era5["provider"],
        "api_endpoint": era5["endpoint"],
        "request_parameters": params,
        "grid_latitude": response.get("latitude"),
        "grid_longitude": response.get("longitude"),
        "grid_elevation_m": response.get("elevation"),
        "hourly_units": response.get("hourly_units"),
        "first_hour_utc": hourly["time"][0],
        "last_hour_utc": hourly["time"][-1],
        "record_count": len(hourly["time"]),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "ERA5 0.25-degree regional background, not neighborhood-scale weather.",
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Read scene timestamps from the authoritative STAC items. ERA5 is hourly,
    # so use the nearest ERA5 hour and preserve its UTC offset from overpass.
    stac = CONFIG["landsat"]["stac_api"].rstrip("/")
    acquisitions = []
    manifest = ROOT / "data/processed/Landsat/Landsat_scene_manifest.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            acquisitions = [{"item_id": r["item_id"], "acquisition_utc": r["date_utc"],
                             "cloud_cover_percent": r.get("global_cloud_percent", "")}
                            for r in csv.DictReader(stream)]
    if not acquisitions:
        raise RuntimeError("No Landsat scene manifest found. Run `python main.py landsat` before `python main.py era5`.")
    climate_rows = []
    with RAW_CSV.open(newline="", encoding="utf-8") as stream:
        climate_rows = list(csv.DictReader(stream))

    matched_fields = ["item_id", "acquisition_utc", "era5_time_utc", "offset_minutes",
                      "cloud_cover_percent", *variables]

    def era5_datetime(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed

    with MATCHED_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=matched_fields)
        writer.writeheader()
        for acquisition in acquisitions:
            target = datetime.fromisoformat(
                acquisition["acquisition_utc"].replace("Z", "+00:00")
            )
            nearest = min(
                climate_rows,
                key=lambda row: abs(era5_datetime(row["time"]) - target),
            )
            nearest_time = era5_datetime(nearest["time"])
            offset_minutes = int((nearest_time - target).total_seconds() / 60)
            if abs(offset_minutes) > 60:
                raise ValueError(f"No ERA5 hour within 60 min of {target.isoformat()}")
            writer.writerow({
                "item_id": acquisition["item_id"],
                "acquisition_utc": acquisition["acquisition_utc"],
                "era5_time_utc": nearest["time"],
                "offset_minutes": offset_minutes,
                "cloud_cover_percent": acquisition["cloud_cover_percent"],
                **{name: nearest[name] for name in variables},
            })

    print(f"Wrote {RAW_CSV} ({len(hourly['time']):,} hourly rows)")
    print(f"Wrote {MATCHED_CSV} ({len(acquisitions)} overpass matches)")
    print(f"ERA5 grid point: {response['latitude']}, {response['longitude']} (UTC)")


if __name__ == "__main__":
    main()
