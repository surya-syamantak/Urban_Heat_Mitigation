"""Download available CPCB station archives for the configured period."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.parse import urlencode
from urllib.request import Request,urlopen
import yaml
ROOT=Path(__file__).resolve().parents[2]; CONFIG=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))
RAW=ROOT/"data/raw/CPCB"; CITY=CONFIG["cpcb"]["city"]
def post_json(url,payload):
    req=Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Accept":"application/json",
        "User-Agent":f"UrbanHeatMitigation-{CITY}/1.0"},method="POST")
    with urlopen(req,timeout=45) as r:
        if r.status==204:return {}
        return json.loads(r.read().decode())
def main():
    RAW.mkdir(parents=True,exist_ok=True); attempts=[]
    years=range(int(CONFIG["study"]["start_date"][:4]),int(CONFIG["study"]["end_date"][:4])+1)
    for station in CONFIG["cpcb"]["stations"]:
        for year in map(str,years):
            local=list(RAW.glob(f"{station['site_code']}_{year}_*"))
            if local:
                attempts.append({"site_code":station["site_code"],"year":year,"status":"existing_local_archive",
                    "local_files":[str(p.relative_to(ROOT)) for p in local]}); continue
            payload={"station_id":station["site_code"],"station_name":station["station"].split(f", {CITY}")[0],
                "state":CONFIG["project"]["state"],"city":CITY,"year":year,"frequency":"1H","dataType":"raw"}
            try:
                result=post_json(CONFIG["cpcb"]["file_path_endpoint"],payload)
                entries=result.get("data",[]) if result.get("status")=="success" else []
                if not entries:
                    attempts.append({"site_code":station["site_code"],"year":year,"status":"no_file_path"}); continue
                for entry in entries:
                    remote=entry.get("filepath")
                    if not remote or str(entry.get("year",year))!=year: continue
                    req=Request(CONFIG["cpcb"]["download_endpoint"]+"?"+urlencode({"file_name":remote}),
                        headers={"Accept":"*/*","User-Agent":f"UrbanHeatMitigation-{CITY}/1.0"})
                    with urlopen(req,timeout=120) as r: content=r.read()
                    name=f"{station['site_code']}_{year}_{Path(remote).name}"
                    if not name.lower().endswith((".csv",".xlsx",".xls")): name+=".csv"
                    path=RAW/name; path.write_bytes(content)
                    attempts.append({"site_code":station["site_code"],"year":year,"status":"downloaded",
                        "filepath":str(remote),"local_file":str(path.relative_to(ROOT)),"bytes":len(content)})
            except HTTPError as e:
                attempts.append({"site_code":station["site_code"],"year":year,
                    "status":"unavailable_404" if e.code==404 else "error","detail":str(e)})
                if e.code!=404: print(f"CPCB {station['site_code']} {year}: {e}")
            except (URLError,TimeoutError,ValueError) as e:
                attempts.append({"site_code":station["site_code"],"year":year,"status":"error","detail":str(e)})
                print(f"CPCB {station['site_code']} {year}: {e}")
    manifest={"source":"CPCB CAAQMS Data Repository","retrieved_utc":datetime.now(timezone.utc).isoformat(),
        "frequency":"hourly raw files; variables depend on station","study_period":[CONFIG["study"]["start_date"],CONFIG["study"]["end_date"]],
        "attempts":attempts}
    out=RAW/f"{CITY}_CPCB_download_manifest.json"; out.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"CPCB archives downloaded or available locally: {sum(x['status'] in ('downloaded','existing_local_archive') for x in attempts)}; manifest: {out}")
if __name__=="__main__": main()
