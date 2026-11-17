"""Save the GCC ward polygons as the project ward source layer."""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"data/raw/Wards.geojson"
OUTPUT=ROOT/"data/raw/Chennai_Wards.geojson"
METADATA=ROOT/"data/raw/Chennai_Wards_source.json"
def main() -> None:
    if not SOURCE.is_file(): raise FileNotFoundError(f"Run `python main.py aoi` first: {SOURCE}")
    doc=json.loads(SOURCE.read_text(encoding="utf-8")); features=doc.get("features",[])
    ward_ids={str((f.get("properties") or {}).get("ward_name") or (f.get("properties") or {}).get("ward") or "").strip() for f in features}
    ward_ids.discard("")
    if not 150 <= len(ward_ids) <= 250: raise RuntimeError(f"Expected approximately 200 GCC wards, found {len(ward_ids)}")
    OUTPUT.write_text(json.dumps({"type":"FeatureCollection","features":features}),encoding="utf-8")
    METADATA.write_text(json.dumps({"source":"Greater Chennai Corporation public GIS Ward_Boundary layer","source_url":"https://gisgcc.chennaicorporation.gov.in/server/rest/services/GCCPublic/GCC_AdminBoundary/MapServer/4","retrieved_utc":datetime.now(timezone.utc).isoformat(),"polygon_feature_count":len(features),"unique_ward_count":len(ward_ids),"aoi_name":"Greater Chennai Corporation (GCC)"},indent=2),encoding="utf-8")
    print(f"GCC wards: {len(ward_ids)} unique ward identifiers in {len(features)} features")
    print(f"Wrote ward source: {OUTPUT}")
if __name__=="__main__": main()
