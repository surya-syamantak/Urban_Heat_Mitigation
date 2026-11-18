"""Download hourly raw CAAQMS station files through CPCB's public repository."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
RAW_DIR = ROOT / "data/raw/CPCB"
FILE_PATH_URL = CONFIG["cpcb"]["file_path_endpoint"]
DOWNLOAD_URL = CONFIG["cpcb"]["download_endpoint"]


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={
        "Content-Type": "application/json", "Accept": "application/json",
        "User-Agent": "UrbanHeatMitigation-Ahmedabad/1.0",
    }, method="POST")
    with urlopen(request, timeout=45) as response:
        if response.status == 204:
            return {}
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    attempts = []
    first_year = int(CONFIG["study"]["start_date"][:4])
    last_year = int(CONFIG["study"]["end_date"][:4])
    # CPCB archive gaps are recorded per station/year; missing historical files
    # do not stop Landsat/ERA5 collection or the rest of the pipeline.
    for station in CONFIG["cpcb"]["stations"]:
        for year in map(str, range(first_year, last_year + 1)):
            existing = sorted(RAW_DIR.glob(f"{station['site_code']}_{year}_*"))
            if existing:
                attempts.append({"site_code": station["site_code"], "year": year,
                                 "status": "existing_local_archive",
                                 "local_files": [str(path.relative_to(ROOT)) for path in existing]})
                continue
            station_name = station["station"].split(", Ahmedabad")[0]
            payload = {
                "station_id": station["site_code"],
                "station_name": station_name,
                "state": "Gujarat",
                "city": "Ahmedabad",
                "year": year,
                "frequency": "1H",
                "dataType": "raw",
            }
            try:
                result = post_json(FILE_PATH_URL, payload)
                entries = result.get("data", []) if result.get("status") == "success" else []
                if not entries:
                    attempts.append({"site_code": station["site_code"], "year": year,
                                     "status": "no_file_path", "detail": result})
                    continue
                for entry in entries:
                    path = entry.get("filepath")
                    if not path or str(entry.get("year", year)) != year:
                        continue
                    query = urlencode({"file_name": path})
                    request = Request(DOWNLOAD_URL + "?" + query, headers={
                        "Accept": "*/*", "User-Agent": "UrbanHeatMitigation-Ahmedabad/1.0",
                    })
                    with urlopen(request, timeout=120) as response:
                        content = response.read()
                    filename = f"{station['site_code']}_{year}_{Path(path).name}"
                    if not filename.lower().endswith((".csv", ".xlsx", ".xls")):
                        filename += ".csv"
                    output = RAW_DIR / filename
                    output.write_bytes(content)
                    attempts.append({"site_code": station["site_code"], "year": year,
                                     "status": "downloaded", "filepath": str(path),
                                     "local_file": str(output.relative_to(ROOT)),
                                     "bytes": len(content)})
            except HTTPError as exc:
                status = "unavailable_404" if exc.code == 404 else "error"
                attempts.append({"site_code": station["site_code"], "year": year,
                                 "status": status, "detail": str(exc)})
                if exc.code != 404:
                    print(f"CPCB {station['site_code']} {year}: {exc}", flush=True)
            except (URLError, TimeoutError, ValueError) as exc:
                attempts.append({"site_code": station["site_code"], "year": year,
                                 "status": "error", "detail": str(exc)})
                print(f"CPCB {station['site_code']} {year}: {exc}", flush=True)
    manifest = {
        "source": "CPCB CAAQMS Data Repository",
        "file_path_endpoint": FILE_PATH_URL,
        "download_endpoint": DOWNLOAD_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "frequency": "hourly raw files; variables depend on station",
        "study_period": [CONFIG["study"]["start_date"], CONFIG["study"]["end_date"]],
        "attempts": attempts,
    }
    (RAW_DIR / "Ahmedabad_CPCB_download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    successes = sum(row["status"] in ("downloaded", "existing_local_archive") for row in attempts)
    print(f"CPCB repository results: {successes} station-year archives downloaded or already present; manifest: "
          f"{RAW_DIR / 'Ahmedabad_CPCB_download_manifest.json'}")


if __name__ == "__main__":
    main()
