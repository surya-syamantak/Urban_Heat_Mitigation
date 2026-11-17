"""Download ERA5 hourly background data and match it to the Landsat manifest."""
from __future__ import annotations
import csv,json
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
import yaml
ROOT=Path(__file__).resolve().parents[2]
CONFIG=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))
RAW_DIR=ROOT/"data/raw/ERA5"; PROCESSED=ROOT/"data/processed"; CITY=CONFIG["project"]["city"]; STUDY=CONFIG["study"]
RAW_CSV=RAW_DIR/f"{CITY}_ERA5_hourly_{STUDY['start_date']}_{STUDY['end_date']}.csv"
META=RAW_DIR/f"{CITY}_ERA5_request_metadata.json"; MATCHED=PROCESSED/"ERA5_Landsat_Overpass_Matched.csv"
MANIFEST=ROOT/"data/processed/Landsat/Landsat_scene_manifest.csv"
def get_json(url):
    with urlopen(Request(url,headers={"User-Agent":f"UrbanHeatMitigation-{CITY}/1.0"}),timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))
def parse_time(value):
    x=datetime.fromisoformat(value.replace("Z","+00:00"))
    return x.replace(tzinfo=timezone.utc) if x.tzinfo is None else x
def main():
    if not MANIFEST.is_file(): raise RuntimeError("Run python main.py landsat before ERA5.")
    with MANIFEST.open(newline="",encoding="utf-8") as f: scenes=list(csv.DictReader(f))
    if not scenes: raise RuntimeError("Landsat manifest is empty.")
    era=CONFIG["era5"]; variables=era["hourly_variables"]
    params={"latitude":era["latitude"],"longitude":era["longitude"],"start_date":STUDY["start_date"],
        "end_date":STUDY["end_date"],"hourly":",".join(variables),"timezone":"UTC","models":era["model"]}
    response=get_json(era["endpoint"]+"?"+urlencode(params)); hourly=response.get("hourly",{})
    if not hourly.get("time"): raise RuntimeError("ERA5 API returned no hourly records.")
    RAW_DIR.mkdir(parents=True,exist_ok=True); PROCESSED.mkdir(parents=True,exist_ok=True)
    fields=["time",*variables]
    with RAW_CSV.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i,t in enumerate(hourly["time"]): w.writerow({k:t if k=="time" else hourly[k][i] for k in fields})
    META.write_text(json.dumps({"provider":era["provider"],"request_parameters":params,
        "grid_latitude":response.get("latitude"),"grid_longitude":response.get("longitude"),
        "hourly_units":response.get("hourly_units"),"first_hour_utc":hourly["time"][0],
        "last_hour_utc":hourly["time"][-1],"record_count":len(hourly["time"]),
        "retrieved_utc":datetime.now(timezone.utc).isoformat(),
        "notes":"ERA5 is coarse regional background, not neighborhood-scale weather."},indent=2),encoding="utf-8")
    climate=list(csv.DictReader(RAW_CSV.open(newline="",encoding="utf-8")))
    matched_fields=["item_id","acquisition_utc","era5_time_utc","offset_minutes","cloud_cover_percent",*variables]
    with MATCHED.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=matched_fields); w.writeheader()
        for s in scenes:
            stamp=s["date_utc"]; target=parse_time(stamp)
            nearest=min(climate,key=lambda r:abs((parse_time(r["time"])-target).total_seconds()))
            when=parse_time(nearest["time"]); offset=int((when-target).total_seconds()/60)
            if abs(offset)>60: raise ValueError(f"No ERA5 hour within 60 min of {stamp}")
            w.writerow({"item_id":s["item_id"],"acquisition_utc":stamp,"era5_time_utc":nearest["time"],
                "offset_minutes":offset,"cloud_cover_percent":s.get("global_cloud_percent",""),
                **{v:nearest[v] for v in variables}})
    dates={s["date_utc"][:10] for s in scenes}
    print(f"Wrote {RAW_CSV} ({len(hourly['time']):,} hourly records)")
    print(f"Wrote {MATCHED} ({len(scenes)} scene matches across {len(dates)} distinct dates)")
if __name__=="__main__": main()
