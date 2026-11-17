"""Phase 4: pure-ML and physics-informed daytime LST models for Nagpur.

Net radiation uses the requested balance equation. Downwelling longwave and
surface emissivity are explicitly estimated because those source products were
not collected. EF and thermal inertia are labeled proxies, not measured fluxes.
"""
from __future__ import annotations

import csv, json, math
from pathlib import Path
import sys

import numpy as np
import yaml
from osgeo import gdal
from qgis.core import QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject, QgsPointXY
from scipy.optimize import lsq_linear
from scipy.ndimage import convolve
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/driver_analysis'))
from phase_shared import forest_fit, ensemble_predict, grid_data, read_raster, load_urban, median_fill

CFG=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
P=ROOT/'data/processed'; LAND=P/'Landsat'; TABLE=ROOT/'outputs/tables'; MAPS=ROOT/'outputs/maps'; REPORT=ROOT/'outputs/reports'
OUT=P/'phase4'; SEED=int(CFG['project']['random_seed']); SIGMA=5.670374419e-8; NODATA=-9999.0
BASE_NAMES=['NDVI','NDBI','MNDWI','albedo','building_density_per_ha','building_coverage_fraction',
 'mean_tagged_building_height_m','height_tag_coverage','road_density_km_per_km2','distance_to_green_m',
 'distance_to_water_m','svf_proxy','ERA5_T2m_C','ERA5_RH_percent','ERA5_wind_ms','ERA5_Kdown_Wm2']
PHYS_NAMES=['net_radiation_proxy_at_Tair_Wm2','evaporative_fraction_proxy','thermal_inertia_proxy']


def read_csv(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def env_longwave(tair_c,dew_c):
 """Brutsaert clear-sky emissivity approximation; returns W m-2."""
 tair_k=tair_c+273.15
 vapor_kpa=.6108*np.exp(17.27*dew_c/(dew_c+237.3))
 eps_atm=np.clip(1.24*np.maximum(vapor_kpa/tair_k,1e-5)**(1/7),.5,1.0)
 return eps_atm*SIGMA*tair_k**4


def physics_features(base, lst_c, ldown):
 ndvi,ndbi,mndwi,alpha=base[:,0],base[:,1],base[:,2],base[:,3]
 tair_k=base[:,12]+273.15
 # Landsat C2 ST uses product-native emissivity; this proxy is used only to
 # estimate radiative balance, not to re-correct the already scaled LST.
 emissivity=np.clip(.95+.04*np.clip(ndvi,0,1),.94,.99)
 # Use air temperature as the reference emission temperature so the predictor
 # does not contain the Landsat target through its outgoing-longwave term.
 tref_k=base[:,12]+273.15
 rn_proxy=(1-alpha)*base[:,15]+emissivity*ldown-emissivity*SIGMA*tref_k**4
 ts_k=lst_c+273.15
 rn_at_lst=(1-alpha)*base[:,15]+emissivity*ldown-emissivity*SIGMA*ts_k**4
 ef=np.clip(.70*ndvi+.30*mndwi+.30,0,1)
 built=np.clip(np.nan_to_num(base[:,5],nan=0),0,1)
 ti=np.clip(.60*ndvi+.40*(1-np.clip(ndbi,-1,1)),0,1)
 return np.column_stack([rn_proxy,ef,ti]),emissivity,rn_at_lst


def assemble(ids,rows,cols,x,y,urban_names,urban,scenes,era):
 urban=median_fill(urban)
 names=BASE_NAMES; records=[]; targets=[]; metas=[]; phys_rows=[]; rn_rows=[]; scene_lut={r['item_id']:r for r in era}
 rng=np.random.default_rng(SEED)
 for scene in scenes:
  item=scene['item_id']; d=LAND/item
  lst_r,meta=read_raster(d/f'{item}_LST_Celsius.tif')
  def sample(name):
   a,_=read_raster(d/f'{item}_{name}.tif');return a[rows,cols].astype(float)
  ndvi=sample('NDVI');ndbi=sample('NDBI');mndwi=sample('MNDWI');alb=sample('ALBEDO')
  lst=lst_r[rows,cols].astype(float)
  e=scene_lut[item]; ta=float(e['temperature_2m']);rh=float(e['relative_humidity_2m']);wind=float(e['wind_speed_10m']);kd=float(e['shortwave_radiation']);dew=float(e['dew_point_2m'])
  base=np.column_stack([ndvi,ndbi,mndwi,alb,urban[:,0],urban[:,1],urban[:,2],urban[:,3],urban[:,4],urban[:,5],urban[:,6],urban[:,7],
      np.full(len(ids),ta),np.full(len(ids),rh),np.full(len(ids),wind),np.full(len(ids),kd)])
  good=np.isfinite(lst)&np.all(np.isfinite(base),axis=1)&(lst>-40)&(lst<80)
  valid=np.flatnonzero(good)
  if len(valid)>5000: valid=rng.choice(valid,5000,replace=False)
  ldown=float(env_longwave(ta,dew));phys,emis,rn_actual=physics_features(base[valid],lst[valid],ldown)
  full=np.column_stack([base[valid],phys]);features=full
  for idx in valid:records.append(base[idx])
  if len(records)>0 and records[-1].shape[0]!=len(names): raise RuntimeError('Phase 4 feature column mismatch')
  targets.extend(lst[valid]);phys_rows.extend(phys);rn_rows.extend(rn_actual);metas.extend([(item,scene['season'],int(idx),int(rows[idx]),int(cols[idx])) for idx in valid])
 print(f"Phase 4 sample: {len(targets):,} scene-cell records from {len(scenes)} scenes",flush=True)
 return np.asarray(records,float),np.asarray(phys_rows,float),np.asarray(targets,float),np.asarray(rn_rows,float),metas


def standardize(x):
 mu=np.mean(x,axis=0);sd=np.std(x,axis=0);sd[sd<1e-10]=1
 return (x-mu)/sd,mu,sd


def physics_fit(x,y,names):
 xs,mu,sd=standardize(x);a=np.column_stack([np.ones(len(xs)),xs])
 lo=np.full(a.shape[1],-np.inf);hi=np.full(a.shape[1],np.inf)
 # LST must not increase for higher vegetation/albedo, while greater available
 # net radiation must not lower LST; EF and daytime thermal-inertia proxies cool.
 for n,sign in [('NDVI',-1),('albedo',-1),('net_radiation_proxy_at_Tair_Wm2',1),('evaporative_fraction_proxy',-1),('thermal_inertia_proxy',-1)]:
  j=names.index(n)+1
  if sign<0:hi[j]=0
  else:lo[j]=0
 # Small ridge stabilization controls severe collinearity between radiation,
 # weather, vegetation, and duplicated hot/cool scene context.
 penalty=np.eye(a.shape[1])*math.sqrt(1.0);penalty[0,0]=0
 a_aug=np.vstack([a,penalty]);y_aug=np.r_[y,np.zeros(a.shape[1])]
 fit=lsq_linear(a_aug,y_aug,bounds=(lo,hi),max_iter=500,tol=1e-9)
 return fit.x,mu,sd


def physics_predict(model,x):
 b,mu,sd=model;return np.column_stack([np.ones(len(x)),(x-mu)/sd])@b


def metrics(y,p):
 e=p-y
 return {'n':int(len(y)),'R2':float(1-np.sum(e*e)/max(np.sum((y-y.mean())**2),1e-12)),
  'RMSE_C':float(np.sqrt(np.mean(e*e))),'MAE_C':float(np.mean(np.abs(e))),'bias_C':float(np.mean(e))}


def moran_i(values,rr,cc,shape):
 valid=np.isfinite(values);h,w=shape
 z=np.zeros((h,w),float);ncell=np.zeros((h,w),float)
 z[rr[valid],cc[valid]]=values[valid];ncell[rr[valid],cc[valid]]=1
 mean=float(np.mean(values[valid]));z[rr[valid],cc[valid]]-=mean
 kernel=np.ones((3,3));kernel[1,1]=0
 neighbors=convolve(z,kernel,mode='constant',cval=0)
 count=convolve(ncell,kernel,mode='constant',cval=0)
 n=int(valid.sum());s0=float(count[rr[valid],cc[valid]].sum())
 den=float(np.sum(z[rr[valid],cc[valid]]**2))
 return float(n/s0*np.sum(z[rr[valid],cc[valid]]*neighbors[rr[valid],cc[valid]])/den) if s0>0 and den>0 else float('nan')


def aggregate_moran(resid,meta,shape):
 bycell={}
 for e,m in zip(resid,meta):bycell.setdefault(m[2],[]).append(float(e))
 ids=list(bycell);vals=np.array([np.mean(bycell[i]) for i in ids]);mlookup={m[2]:m for m in meta}
 rr=np.array([mlookup[i][3] for i in ids]);cc=np.array([mlookup[i][4] for i in ids])
 return moran_i(vals,rr,cc,shape)


def run_spatial_cv(Xb,Xp,y,meta,base_names,phys_names,shape):
 # Convert master-grid cell positions to metres before building spatial folds.
 xy=np.array([[m[4]*30.,m[3]*30.] for m in meta],float)
 # 5 km blocks; modulo assignment creates separated held-out neighborhoods.
 bx=np.floor(xy[:,0]/5000).astype(int);by=np.floor(xy[:,1]/5000).astype(int)
 nxb=int(bx.max()-bx.min()+1);fold=((bx-bx.min())+(by-by.min())*nxb)%5
 pred_b=np.full(len(y),np.nan);pred_p=pred_b.copy();fold_rows=[]
 for f in range(5):
  test=np.flatnonzero(fold==f);train=np.flatnonzero(fold!=f)
  model=forest_fit(Xb[train],y[train],trees=10,depth=3,seed=SEED+f)
  pred_b[test]=ensemble_predict(model,Xb[test])
  pm=physics_fit(np.column_stack([Xb[train],Xp[train]]),y[train],base_names+phys_names)
  pred_p[test]=physics_predict(pm,np.column_stack([Xb[test],Xp[test]]))
  for label,p in [('Pure ML CART forest',pred_b),('Physics-informed constrained regression',pred_p)]:
   mm=metrics(y[test],p[test]);fold_rows.append({'fold':f+1,'model':label,**mm})
 print('Spatial block CV completed',flush=True)
 return pred_b,pred_p,fold_rows


def main():
 app=QgsApplication([],False);app.initQgis()
 OUT.mkdir(parents=True,exist_ok=True);TABLE.mkdir(parents=True,exist_ok=True);MAPS.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 with (LAND/'Landsat_scene_manifest.csv').open(newline='',encoding='utf-8') as f:scenes=list(csv.DictReader(f))
 era=read_csv(P/'ERA5_Landsat_Overpass_Matched.csv');ids,rows,cols,x,y,ward=grid_data();urban_names,urban=load_urban(ids)
 Xb,Xp,target,rn_actual,meta=assemble(ids,rows,cols,x,y,urban_names,urban,scenes,era);pnames=BASE_NAMES+PHYS_NAMES
 # Persist the sampled and versioned model matrix with scene IDs and cell IDs.
 predictor_path=TABLE/'Nagpur_phase4_model_sample.csv'
 with predictor_path.open('w',newline='',encoding='utf-8') as f:
  fields=['item_id','season','grid_id','row_index','col_index',*BASE_NAMES,*PHYS_NAMES,'estimated_Rn_at_LST_Wm2','lst_c']
  w=csv.writer(f);w.writerow(fields)
  for b,p,rn,t,m in zip(Xb,Xp,rn_actual,target,meta):w.writerow([m[0],m[1],m[2],m[3],m[4],*b,*p,rn,t])
 # Spatial block cross-validation, including residual spatial autocorrelation.
 pbase,pphys,fold_rows=run_spatial_cv(Xb,np.column_stack([Xp]),target,meta,BASE_NAMES,PHYS_NAMES,(max(rows)+1,max(cols)+1))
 spatial=[]
 for name,pred in [('Pure ML CART forest',pbase),('Physics-informed constrained regression',pphys)]:
  spatial.append({'model':name,**metrics(target,pred),'residual_Morans_I_queen_3x3':aggregate_moran(pred-target,meta,(max(rows)+1,max(cols)+1))})
 # Temporal holdout: latest acquisition (all other dates train).
 # Use acquisition date, not scene ID ordering, for the temporal holdout.
 latest_scene=max(scenes,key=lambda s:s['date_utc'])['item_id']; ti=np.array([m[0]==latest_scene for m in meta]);tr=~ti
 rf=forest_fit(Xb[tr],target[tr],trees=18,depth=4,seed=SEED+91);pred=ensemble_predict(rf,Xb[ti])
 ph=physics_fit(np.column_stack([Xb[tr],Xp[tr]]),target[tr],pnames);pred2=physics_predict(ph,np.column_stack([Xb[ti],Xp[ti]]))
 temporal=[{'holdout':'latest scene '+latest_scene,'model':'Pure ML CART forest',**metrics(target[ti],pred),
            'residual_Morans_I_queen_3x3':aggregate_moran(pred-target[ti], [m for m,ok in zip(meta,ti) if ok],(max(rows)+1,max(cols)+1))},
           {'holdout':'latest scene '+latest_scene,'model':'Physics-informed constrained regression',**metrics(target[ti],pred2),
            'residual_Morans_I_queen_3x3':aggregate_moran(pred2-target[ti], [m for m,ok in zip(meta,ti) if ok],(max(rows)+1,max(cols)+1))}]
 # Season transfer (train one season, test the other).
 for a,b in [('hot','cool'),('cool','hot')]:
  tr=np.array([m[1]==a for m in meta]);te=~tr
  forest=forest_fit(Xb[tr],target[tr],trees=18,depth=4,seed=SEED+(1 if a=='cool' else 0));p1=ensemble_predict(forest,Xb[te])
  lm=physics_fit(np.column_stack([Xb[tr],Xp[tr]]),target[tr],pnames);p2=physics_predict(lm,np.column_stack([Xb[te],Xp[te]]))
  metam=[m for m,ok in zip(meta,te) if ok]
  temporal += [{'holdout':f'train {a}; test {b}','model':'Pure ML CART forest',**metrics(target[te],p1),'residual_Morans_I_queen_3x3':aggregate_moran(p1-target[te],metam,(max(rows)+1,max(cols)+1))},
     {'holdout':f'train {a}; test {b}','model':'Physics-informed constrained regression',**metrics(target[te],p2),'residual_Morans_I_queen_3x3':aggregate_moran(p2-target[te],metam,(max(rows)+1,max(cols)+1))}]
 # Physical perturbation test on the latest-scene holdout.
 test_idx=np.flatnonzero(ti)[:800];base_rows=Xb[test_idx].copy();phys_rows=Xp[test_idx].copy()
 ph=np.column_stack([base_rows,phys_rows])
 # Train/target only on remaining observations so perturb tests stay held out.
 tr_latest=~ti
 base_model=forest_fit(Xb[tr_latest],target[tr_latest],trees=18,depth=4,seed=SEED+92)
 pure_test=Xb[test_idx]
 def perturb_result(kind,amount):
  pb=pure_test.copy();px=ph.copy()
  col=0 if kind=='NDVI' else 3;pb[:,col]+=amount
  if kind=='NDVI':
   px[:,col]+=amount;px[:,len(BASE_NAMES)+PHYS_NAMES.index('evaporative_fraction_proxy')]=np.clip(.70*px[:,0]+.30*px[:,2]+.30,0,1)
   px[:,len(BASE_NAMES)+PHYS_NAMES.index('thermal_inertia_proxy')]=np.clip(.60*px[:,0]+.40*(1-np.clip(px[:,1],-1,1)),0,1)
   # Higher emissivity increases outgoing loss at the observed LST in this clear-sky daytime setting.
   old_eps=np.clip(.95+.04*np.clip(base_rows[:,0],0,1),.94,.99);new_eps=np.clip(.95+.04*np.clip(pb[:,0],0,1),.94,.99)
   ldown=np.array([float(env_longwave(float(v[12]),float(v[12])-5)) for v in base_rows])
   tref=base_rows[:,12]+273.15
   old_rn=(1-base_rows[:,3])*base_rows[:,15]+old_eps*ldown-old_eps*SIGMA*tref**4
   new_rn=(1-pb[:,3])*pb[:,15]+new_eps*ldown-new_eps*SIGMA*tref**4
   px[:,len(BASE_NAMES)+PHYS_NAMES.index('net_radiation_proxy_at_Tair_Wm2')]=new_rn
  else:
   pb[:,col]+=amount;px[:,col]+=amount
   # Recompute absorbed shortwave within the energy-balance feature.
   px[:,len(BASE_NAMES)+PHYS_NAMES.index('net_radiation_proxy_at_Tair_Wm2')]-=amount*base_rows[:,15]
  pred0=ensemble_predict(base_model,pure_test);pred1=ensemble_predict(base_model,pb)
  p0=physics_predict(phmodel,ph);p1=physics_predict(phmodel,px)
  return {'perturbation':kind,'amount':amount,'pure_ml_mean_delta_c':float(np.mean(pred1-pred0)),
    'pure_ml_share_nonincreasing':float(np.mean(pred1<=pred0+1e-8)),
    'physics_model_mean_delta_c':float(np.mean(p1-p0)),
    'physics_model_share_nonincreasing':float(np.mean(p1<=p0+1e-8))}
 def model_features(base,phys):return np.column_stack([base,phys])
 phmodel=physics_fit(np.column_stack([Xb[tr_latest],Xp[tr_latest]]),target[tr_latest],pnames)
 plausibility=[perturb_result('NDVI',.1),perturb_result('albedo',.05)]
 # Station context: compare nearest-pixel Landsat surface temperature to matched
 # CPCB air temperature; these variables differ and this is not a model validation score.
 cpcb=read_csv(P/'CPCB_Landsat_Overpass_Matched.csv');scene_manifest={s['item_id']:s for s in scenes}
 station_points={s['site_code']:(float(s['longitude']),float(s['latitude'])) for s in CFG['cpcb']['stations']}
 transform=QgsCoordinateTransform(QgsCoordinateReferenceSystem('EPSG:4326'),QgsCoordinateReferenceSystem('EPSG:32644'),QgsProject.instance())
 tree=cKDTree(np.column_stack([x,y]));lst_cache={}
 station_vals=[]
 for r in cpcb:
  item=r['item_id']
  try:air=float(r['air_temperature_c'])
  except (ValueError,TypeError):continue
  if item not in scene_manifest:continue
  if item not in lst_cache:lst_cache[item]=read_raster(LAND/item/f'{item}_LST_Celsius.tif')[0][rows,cols]
  lon,lat=station_points[r['site_code']];pt=QgsPointXY(lon,lat);pt=transform.transform(pt);_,idx=tree.query([pt.x(),pt.y()]);surface=float(lst_cache[item][idx])
  if math.isfinite(surface):station_vals.append({'item_id':item,'season':scene_manifest[item]['season'],'site_code':r['site_code'],
    'station':r['station'],'air_temperature_c':air,'nearest_Landsat_LST_c':surface,'LST_minus_air_c':surface-air})
 with (TABLE/'Nagpur_phase4_CPCB_context_check.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['item_id','season','site_code','station','air_temperature_c','nearest_Landsat_LST_c','LST_minus_air_c']);w.writeheader();w.writerows(station_vals)
 if station_vals:
  air=np.array([r['air_temperature_c'] for r in station_vals]);surf=np.array([r['nearest_Landsat_LST_c'] for r in station_vals])
  cpcb_meta={'n_station_scene_pairs':len(air),'pearson_r':float(np.corrcoef(air,surf)[0,1]),
    'MAE_LST_minus_air_C':float(np.mean(np.abs(surf-air))),'bias_LST_minus_air_C':float(np.mean(surf-air)),
    'interpretation':'context only: radiometric land-surface temperature is not station 2 m air temperature.'}
 else:cpcb_meta={'n_station_scene_pairs':0,'interpretation':'No complete CPCB pairs.'}
 with (TABLE/'Nagpur_phase4_metrics.json').open('w',encoding='utf-8') as f:json.dump({'spatial_block_cv':spatial,'spatial_block_cv_folds':fold_rows,
   'temporal_holdout':temporal,'physical_perturbation':plausibility,'CPCB_context_check':cpcb_meta,
   'sample_size':len(target),'source_scenes':len(scenes)},f,indent=2)
 with (TABLE/'Nagpur_phase4_spatial_cv_folds.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(fold_rows[0]));w.writeheader();w.writerows(fold_rows)
 with (TABLE/'Nagpur_phase4_temporal_holdout.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(temporal[0]));w.writeheader();w.writerows(temporal)
 date_count=len({scene['date_utc'][:10] for scene in scenes})
 report=make_report(spatial,temporal,plausibility,cpcb_meta,len(target),date_count)
 (REPORT/'Nagpur_PHASE4_QA.md').write_text(report,encoding='utf-8')
 print(f"Phase 4: sample {len(target):,}; spatial RF R2={spatial[0]['R2']:.3f}, physics R2={spatial[1]['R2']:.3f}",flush=True)
 print('Metrics:',TABLE/'Nagpur_phase4_metrics.json',flush=True)


def make_report(spatial,temporal,plausibility,cpcb,n,date_count):
 def table(rows,cols):
  out=['| '+' | '.join(cols)+' |','|'+ '|'.join(['---']*len(cols))+'|']
  for r in rows:out.append('| '+' | '.join(str(r.get(c,'')) if not isinstance(r.get(c),float) else f"{r[c]:.3f}" for c in cols)+' |')
  return out
 lines=['# Phase 4 physics-informed heat model — Nagpur','',
  '## Model design','',
  f'- Training matrix: {n:,} sampled 30 m cell-scene observations from {date_count} unique acquisition dates in the 2021–2025 seasonal sample. Target is Landsat daytime LST (°C).',
  '- Pure ML baseline: bootstrap CART random forest using satellite, urban-form, and overpass weather inputs.',
  '- Physics-informed model: constrained linear regression with radiative/evaporative/thermal proxy inputs. It adds net radiation, evaporative-fraction proxy, and thermal-inertia proxy; coefficients are constrained so NDVI/albedo/evaporative fraction/thermal inertia do not increase predicted LST and net radiation does not decrease it.',
  '- Net radiation: `Rn = (1 − α)K↓ + εL↓ − εσTs⁴`. `K↓` is matched ERA5 shortwave. `L↓` uses Brutsaert atmospheric-emissivity approximation from ERA5 air temperature/dewpoint. Surface emissivity is estimated from NDVI because Landsat emissivity assets were not collected. Estimated Rn evaluated at observed Landsat `Ts` is included in the sampled catalogue, but withheld from training to prevent target leakage; the model uses a potential-Rn proxy evaluated at ERA5 `Tair` instead. No second emissivity correction is applied to product-native Landsat ST.',
  '- `EF proxy` is a bounded combination of NDVI and MNDWI, not measured `LE/(H+LE)`. Thermal inertia is an NDVI/NDBI proxy, not a material-property measurement. These features do not close the full `Rn = H + LE + G` balance because H, LE, and G are not measured.','',
  '## Spatial-block cross-validation','',*table(spatial,['model','R2','RMSE_C','MAE_C','bias_C','residual_Morans_I_queen_3x3']),'',
  '## Temporal holdouts','',*table(temporal,['holdout','model','R2','RMSE_C','MAE_C','bias_C','residual_Morans_I_queen_3x3']),'',
  'Temporal results include a latest-scene holdout and two cross-season transfer directions. The cross-season task is difficult because hot and cool LST distributions differ; interpret negative R² as failed transfer, not model failure in its source season.','',
  '## Physical plausibility perturbations','',*table(plausibility,['perturbation','amount','pure_ml_mean_delta_c','pure_ml_share_nonincreasing','physics_model_mean_delta_c','physics_model_share_nonincreasing']),'',
  'Changes are mean predictions over a held-out sample, with all other predictors held fixed and dependent physics features recalculated. The physics-informed model has sign constraints; the pure forest is unconstrained, so report its non-increasing share and do not assume every cell responds monotonically.','',
  '## CPCB station context','',
  f"Nearest-cell LST and matched CPCB 2 m air-temperature context: {cpcb.get('n_station_scene_pairs',0)} pairs, Pearson r={cpcb.get('pearson_r',float('nan')):.3f}, MAE of LST minus air={cpcb.get('MAE_LST_minus_air_C',float('nan')):.2f}°C, bias={cpcb.get('bias_LST_minus_air_C',float('nan')):.2f}°C. This is not a direct LST validation because surface and air temperature are different variables.",'',
  '## Limitations and reproduction','',
  f'{date_count} selected clear-sky acquisition dates are available; ERA5 is one coarse city background point per date. Emissivity, EF, and thermal inertia are proxies; flux terms H, LE, and G are unavailable. OSM is a 2026 snapshot and height/SVF remain proxies. The CPCB comparison is contextual. Tree model is a custom compact CART forest because scikit-learn is unavailable in the QGIS runtime.','',
  'Run `python main.py phase4`. Script: `src/physics_model/phase4_physics_model.py`. Outputs: sampled feature matrix, fold/temporal metrics, CPCB context table, perturbation metrics, and this report.']
 return '\n'.join(lines)+'\n'


if __name__=='__main__':main()
