"""Match CPCB hourly station weather observations to Landsat overpasses."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
RAW_DIR = ROOT / "data/raw/CPCB"
SCENE_MANIFEST = ROOT / "data/processed/Landsat/Landsat_scene_manifest.csv"
OUTPUT = ROOT / "data/processed/CPCB_Landsat_Overpass_Matched.csv"
SOURCE_TZ = ZoneInfo(CONFIG["cpcb"]["timestamp_timezone"])
SOURCE_FIELDS = {
    "AT (°C)": "air_temperature_c",
    "RH (%)": "relative_humidity_percent",
    "WS (m/s)": "wind_speed_m_s",
    "WD (deg)": "wind_direction_deg",
    "RF (mm)": "rainfall_mm",
    "SR (W/mt2)": "solar_radiation_w_m2",
    "BP (mmHg)": "pressure_mmhg",
}


def load_station_years() -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {}
    for station in CONFIG["cpcb"]["stations"]:
        code = station["site_code"]
        rows[code] = []
        for path in RAW_DIR.glob(f"{code}_*_*.csv"):
            with path.open(newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    stamp = row.get("Timestamp", "").strip()
                    if not stamp:
                        continue
                    local_time = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=SOURCE_TZ)
                    row["_local_dt"] = local_time.isoformat()
                    row["_utc_epoch"] = str(local_time.astimezone(timezone.utc).timestamp())
                    rows[code].append(row)
        rows[code].sort(key=lambda row: float(row["_utc_epoch"]))
    return rows


def main() -> None:
    if not SCENE_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing Landsat manifest: {SCENE_MANIFEST}")
    station_rows = load_station_years()
    fields = [
        "item_id", "scene_overpass_utc", "site_code", "station", "latitude", "longitude",
        "cpcb_timestamp_ist", "cpcb_timestamp_utc", "offset_minutes", *SOURCE_FIELDS.values(),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with SCENE_MANIFEST.open(newline="", encoding="utf-8") as scene_stream, OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as output_stream:
        writer = csv.DictWriter(output_stream, fieldnames=fields)
        writer.writeheader()
        for scene in csv.DictReader(scene_stream):
            target = datetime.fromisoformat(scene["date_utc"].replace("Z", "+00:00"))
            target_epoch = target.timestamp()
            for station in CONFIG["cpcb"]["stations"]:
                candidates = station_rows.get(station["site_code"], [])
                if not candidates:
                    continue
                match = min(candidates, key=lambda row: abs(float(row["_utc_epoch"]) - target_epoch))
                matched_local = datetime.fromisoformat(match["_local_dt"])
                matched_utc = matched_local.astimezone(timezone.utc)
                offset = int((matched_utc - target).total_seconds() / 60)
                if abs(offset) > 60:
                    continue
                writer.writerow({
                    "item_id": scene["item_id"],
                    "scene_overpass_utc": scene["date_utc"],
                    "site_code": station["site_code"], "station": station["station"],
                    "latitude": station["latitude"], "longitude": station["longitude"],
                    "cpcb_timestamp_ist": matched_local.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "cpcb_timestamp_utc": matched_utc.isoformat(), "offset_minutes": offset,
                    **{out_name: match.get(src_name, "") for src_name, out_name in SOURCE_FIELDS.items()},
                })
                row_count += 1
    print(f"Wrote {OUTPUT}: {row_count} scene-station matches; CPCB source timestamps interpreted as {SOURCE_TZ.key}.")


if __name__ == "__main__":
    main()
