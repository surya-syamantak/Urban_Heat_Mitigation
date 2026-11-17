"""Create a date-balanced, year-aware Phase 3 table from existing scene rasters."""
from __future__ import annotations
import csv, json, zlib
from pathlib import Path
import numpy as np
import yaml
from osgeo import gdal
from qgis.core import QgsVectorLayer

ROOT=Path(__file__).resolve().parents[2]
CFG=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
CITY=CFG['project']['city']; LAND=ROOT/'data/processed/Landsat'
GRID=ROOT/CFG['spatial']['grid']; CENTROID=ROOT/CFG['spatial']['centroid_ward_table']
URBAN=ROOT/f"data/processed/urban_form/{CITY}_urban_form_30m.csv"
OUT=ROOT/'outputs/tables'; OUT.mkdir(parents=True,exist_ok=True)
CAP_PER_DATE=4000; SEED=int(CFG['project'].get('random_seed',42)); NODATA=-9999.0
URBAN_NAMES=['building_equivalents_per_ha','building_coverage_fraction','mean_tagged_building_height_m',
 'height_tag_area_coverage_fraction','road_density_km_per_km2','distance_to_green_m','distance_to_water_m','svf_proxy']
FEATURES=['NDVI','NDBI','MNDWI','albedo',*URBAN_NAMES,'impervious_fraction','LULC_vegetation','LULC_built','LULC_water','NDVI_x_building_density']

def read_raster(path):
 ds=gdal.Open(str(path))
 if ds is None: raise FileNotFoundError(f'Cannot open required raster: {path}')
 band=ds.GetRasterBand(1); a=band.ReadAsArray().astype(np.float32)
 nd=band.GetNoDataValue()
 if nd is not None: a[a==nd]=np.nan
 a[~np.isfinite(a)]=np.nan
 return a,(ds.RasterXSize,ds.RasterYSize,ds.GetGeoTransform(),ds.GetProjection())

def main():
 gl=QgsVectorLayer(str(GRID),'grid','ogr'); cl=QgsVectorLayer(str(CENTROID),'centroids','ogr')
 if not gl.isValid() or not cl.isValid(): raise RuntimeError('Phase 3 grid or centroid/ward layer is missing or invalid.')
 wards={int(f['grid_id']):str(f['Name']) for f in cl.getFeatures()}
 feats=list(gl.getFeatures()); ids=np.array([int(f['grid_id']) for f in feats],dtype=np.int64)
 rows=np.array([int(f['row_index']) for f in feats]); cols=np.array([int(f['col_index']) for f in feats])
 east=np.array([(float(f['left'])+float(f['right']))/2 for f in feats]); north=np.array([(float(f['top'])+float(f['bottom']))/2 for f in feats])
 ward=np.array([wards.get(int(i),'') for i in ids],dtype=object)
 if np.any(ward==''): raise RuntimeError('One or more grid cells have no ward assignment.')
 urban_map={}
 with URBAN.open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   try: urban_map[int(r['grid_id'])]=r
   except (ValueError,KeyError): continue
 urban=np.array([[float(urban_map.get(int(i),{}).get(k) or 'nan') for k in URBAN_NAMES] for i in ids],dtype=float)
 for j in range(urban.shape[1]):
  ok=np.isfinite(urban[:,j]); urban[~ok,j]=float(np.median(urban[ok,j])) if ok.any() else 0.
 manifest_path=LAND/'Landsat_scene_manifest.csv'
 if not manifest_path.exists(): raise FileNotFoundError(f'Missing Phase 1 scene manifest: {manifest_path}')
 with manifest_path.open(newline='',encoding='utf-8') as f: manifest=list(csv.DictReader(f))
 if not manifest: raise RuntimeError('Landsat scene manifest contains no scenes; run Phase 1 first.')
 by_date={}
 for scene in manifest: by_date.setdefault(scene['date_utc'][:10],[]).append(scene)
 field=['item_id','date','year','season','grid_id','ward','lst_median_c',*FEATURES,'easting_m','northing_m']
 outpath=OUT/f'{CITY}_phase3_per_scene_predictor_table.csv'; summary=[]; missing=[]
 with outpath.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=field); w.writeheader()
  for date,scenes in sorted(by_date.items()):
   parts=[]; seen=set()
   for scene in scenes:
    item=scene['item_id']; season=scene['season'].lower(); year=int(scene.get('year') or date[:4])
    if season not in ('hot','cool'): raise ValueError(f'Unexpected season {season!r} for {item}')
    folder=LAND/item
    paths={k:folder/f'{item}_{suffix}.tif' for k,suffix in [('lst','LST_Celsius'),('NDVI','NDVI'),('NDBI','NDBI'),('MNDWI','MNDWI'),('albedo','ALBEDO')]}
    absent=[str(p) for p in paths.values() if not p.exists()]
    if absent: missing.extend(absent); continue
    arrays={}; meta=None
    for key,path in paths.items():
     arr,info=read_raster(path); arrays[key]=arr
     if meta is None: meta=info
     elif info[:3]!=meta[:3]: raise RuntimeError(f'Raster grids are misaligned for scene {item}: {path.name}')
    lulc_path=LAND/f'seasonal_{season}'/f'{CITY}_{season}_LULC_rule_based.tif'
    lulc,lmeta=read_raster(lulc_path)
    if lmeta[:3]!=meta[:3]: raise RuntimeError(f'Scene and {season} LULC rasters do not share a grid.')
    vals={k:a[rows,cols] for k,a in arrays.items()}; cat=lulc[rows,cols]
    use=np.column_stack([vals['NDVI'],vals['NDBI'],vals['MNDWI'],vals['albedo'],urban,
     np.clip(.5*(cat==3).astype(np.float32)+.5*np.nan_to_num(urban[:,1],nan=0),0,1),
     (cat==2).astype(np.float32),(cat==3).astype(np.float32),(cat==4).astype(np.float32),
     vals['NDVI']*np.log1p(np.maximum(urban[:,0],0))]).astype(np.float32,copy=False)
    valid=np.isfinite(vals['lst'])&np.all(np.isfinite(use),axis=1)
    if seen: valid &= np.array([int(i) not in seen for i in ids])
    ix=np.flatnonzero(valid)
    if not len(ix): continue
    seen.update(int(ids[i]) for i in ix)
    parts.append((item,year,season,ix,use[ix],vals['lst'][ix]))
    del arrays, vals, cat, lulc, use
   total=sum(len(part[3]) for part in parts)
   if not total: continue
   if total>CAP_PER_DATE:
    rng=np.random.default_rng(SEED+zlib.crc32(date.encode('utf-8')))
    keep=np.sort(rng.choice(total,CAP_PER_DATE,replace=False))
   else: keep=np.arange(total)
   offset=0
   for item,year,season,ix,features,target in parts:
    local=keep[(keep>=offset)&(keep<offset+len(ix))]-offset
    for j in local:
     i=int(ix[j]); xv=features[j]
     w.writerow({'item_id':item,'date':date,'year':year,'season':season,'grid_id':int(ids[i]),'ward':ward[i],
      'lst_median_c':float(target[j]),**{k:float(v) for k,v in zip(FEATURES,xv)},'easting_m':float(east[i]),'northing_m':float(north[i])})
    offset+=len(ix)
   summary.append({'date':date,'year':parts[0][1],'season':parts[0][2],'n':len(keep),'scene_items':len(parts)})
 if missing: raise FileNotFoundError('Required per-scene predictor rasters are missing; first missing: '+missing[0])
 if not summary: raise RuntimeError('No valid per-scene samples were created; check raster/grid alignment and QA masks.')
 years=sorted({int(r['year']) for r in summary}); seasons={s:sum(r['n'] for r in summary if r['season']==s) for s in ('hot','cool')}
 if len(years)<3 or not all(seasons.values()): raise RuntimeError(f'Year-held-out comparison needs >=3 years and both seasons; got years={years}, season counts={seasons}')
 metadata={'project_city':CITY,'records':sum(r['n'] for r in summary),'sample_cap_per_acquisition_date':CAP_PER_DATE,'years':years,
  'season_counts':seasons,'dates':summary,'features':FEATURES,'source':'existing Phase 1 per-scene rasters; no Phase 1/2 changes'}
 (OUT/f'{CITY}_phase3_per_scene_sample_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
 print(f'Per-scene year-aware table: {outpath} ({sum(r["n"] for r in summary):,} rows; {len(summary)} dates; years {years})',flush=True)
 
if __name__=='__main__': main()
