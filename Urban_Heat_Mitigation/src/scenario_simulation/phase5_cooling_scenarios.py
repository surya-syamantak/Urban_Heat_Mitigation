"""Simulate Ahmedabad cooling scenarios with the Phase 4 physics-feature model."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
import yaml
from osgeo import gdal
from qgis.core import QgsApplication, QgsVectorLayer
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/driver_analysis'));sys.path.insert(0,str(ROOT/'src/physics_model'));sys.path.insert(0,str(ROOT/'src/hotspot_analysis'))
from phase3_drivers import grid_data,read_raster,load_urban,median_fill
from phase4_physics_model import BASE_NAMES,PHYS_NAMES,physics_fit,physics_features,physics_predict,env_longwave
from phase2_hotspots import heat_index_celsius
CFG=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
P=ROOT/'data/processed';LAND=P/'Landsat';TABLE=ROOT/'outputs/tables';MAPS=ROOT/'outputs/maps';REPORT=ROOT/'outputs/reports'
NODATA=-9999.0;SIGMA=5.670374419e-8


def read_rows(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def save_raster(path,array,meta,description):
 w,h,gt,proj=meta;ds=gdal.GetDriverByName('GTiff').Create(str(path),w,h,1,gdal.GDT_Float32,options=['COMPRESS=DEFLATE','TILED=YES'])
 ds.SetGeoTransform(gt);ds.SetProjection(proj);b=ds.GetRasterBand(1);b.SetNoDataValue(NODATA);b.SetDescription(description);b.WriteArray(array);ds=None


def dist_to_source(source,shape):
 rr=distance_transform_edt(~source,sampling=(30.,30.)).astype(np.float32)
 rr[~shape]=np.nan
 return rr


def main():
 app=QgsApplication([],False);app.initQgis()
 TABLE.mkdir(parents=True,exist_ok=True);MAPS.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);(P/'phase5').mkdir(parents=True,exist_ok=True)
 ids,rows,cols,x,y,wards=grid_data();n=len(ids);urban_names,urban=load_urban(ids);urban=median_fill(urban)
 fields={k:i for i,k in enumerate(BASE_NAMES)}
 scene_manifest=read_rows(LAND/'Landsat_scene_manifest.csv'); hot_scenes=[s for s in scene_manifest if s['season']=='hot']
 # Model fitting uses the complete Phase 4 sample; counterfactuals use hot-season medians.
 train=[r for r in read_rows(TABLE/'Ahmedabad_phase4_model_sample.csv') if r['season']=='hot']
 Xb=np.array([[float(r[c]) for c in BASE_NAMES] for r in train],float)
 Xp=np.array([[float(r[c]) for c in PHYS_NAMES] for r in train],float)
 yt=np.array([float(r['lst_c']) for r in train],float)
 support_lo=np.percentile(Xb,1,axis=0);support_hi=np.percentile(Xb,99,axis=0)
 phys_lo=np.percentile(Xp,1,axis=0);phys_hi=np.percentile(Xp,99,axis=0)
 model=physics_fit(np.column_stack([Xb,Xp]),yt,BASE_NAMES+PHYS_NAMES)
 era=read_rows(P/'ERA5_Landsat_Overpass_Matched.csv');eh=[r for r in era if any(r['item_id']==s['item_id'] for s in hot_scenes)]
 climate={k:float(np.mean([float(r[k]) for r in eh])) for k in ['temperature_2m','relative_humidity_2m','wind_speed_10m','shortwave_radiation','dew_point_2m']}
 sezon=LAND/'seasonal_hot';lst_r,meta=read_raster(sezon/'Ahmedabad_hot_LST_C_median.tif')
 raster_names=['NDVI','NDBI','MNDWI','ALBEDO','LULC_rule_based']
 rasters={}
 for name in raster_names:rasters[name]=read_raster(sezon/f'Ahmedabad_hot_{name}_median.tif' if name!='LULC_rule_based' else sezon/'Ahmedabad_hot_LULC_rule_based.tif')[0]
 # Align urban-form table and seasonal rasters to canonical grid feature order.
 ndvi=rasters['NDVI'][rows,cols].astype(float);ndbi=rasters['NDBI'][rows,cols].astype(float)
 mndwi=rasters['MNDWI'][rows,cols].astype(float);albedo=rasters['ALBEDO'][rows,cols].astype(float)
 lulc=np.nan_to_num(rasters['LULC_rule_based'][rows,cols],nan=0).astype(int)
 lst=lst_r[rows,cols].astype(float)
 base=np.column_stack([ndvi,ndbi,mndwi,albedo,urban[:,0],urban[:,1],urban[:,2],urban[:,3],urban[:,4],urban[:,5],urban[:,6],urban[:,7],
   np.full(n,climate['temperature_2m']),np.full(n,climate['relative_humidity_2m']),np.full(n,climate['wind_speed_10m']),np.full(n,climate['shortwave_radiation'])])
 valid=np.isfinite(lst)&np.all(np.isfinite(base),axis=1)
 ldown=float(env_longwave(climate['temperature_2m'],climate['dew_point_2m']))
 phys,_,_=physics_features(base,lst,ldown)
 prediction=physics_predict(model,np.column_stack([base,phys]));prediction[~valid]=np.nan
 # Population density is GHSL 2020 persons/km2; use stored clipped-cell area.
 pop,_=read_raster(P/'phase2/Ahmedabad_GHSL2020_population_density_30m.tif');pop_density=pop[rows,cols]
 cent=QgsVectorLayer(str(ROOT/CFG['spatial']['centroid_ward_table']),'cellareas','ogr')
 area_by_id={int(f['grid_id']):float(f['area_km2']) for f in cent.getFeatures()};areas=np.array([area_by_id[int(i)] for i in ids])
 pop_est=np.maximum(0,np.nan_to_num(pop_density))*areas
 # Eligible areas defined from current hot LST, Landsat rule-LULC, and OSM form.
 coverage=base[:,fields['building_coverage_fraction']];dist_water=base[:,fields['distance_to_water_m']]
 building=(lulc==3)|(coverage>=.1)|(ndbi>.15)
 water_existing=(lulc==4)|(mndwi>.2)
 water_reference=valid&water_existing
 water_medians={key:float(np.nanmedian(base[water_reference,fields[key]])) for key in ['NDVI','NDBI','MNDWI','albedo']}
 valid_lst=lst[valid];hot_p75=float(np.nanpercentile(valid_lst,75));hot_p90=float(np.nanpercentile(valid_lst,90))
 sparse_open=valid&~building&~water_existing&(coverage<.1)
 green_eligible=sparse_open&(lst>=hot_p75)&(ndvi<.35)
 roof_eligible=valid&building&(lst>=hot_p75)
 cool_eligible=valid&building&(albedo<.6)
 # Limit new water footprint to 1% of the clipped city area, selected from
 # very hot low-built cells far from existing water.
 target_water=int(round(.01*valid.sum()));cand=np.flatnonzero(sparse_open&(lst>=hot_p75)&(dist_water>500))
 cand=cand[np.argsort(-lst[cand])[:target_water]]
 water_eligible=np.zeros(n,bool);water_eligible[cand]=True
 # Recompute a hot-priority spatial mask and intervention distance surfaces.
 h,w=meta[1],meta[0];valid_grid=np.zeros((h,w),bool);valid_grid[rows[valid],cols[valid]]=True
 green_source=np.zeros((h,w),bool);green_source[rows[green_eligible],cols[green_eligible]]=True
 water_add=np.zeros((h,w),bool);water_add[rows[water_eligible],cols[water_eligible]]=True
 dgreen=dist_to_source(green_source,valid_grid);dwater=dist_to_source(water_add,valid_grid)
 # Baseline current mapped distances remain where new intervention is not closer.
 dgreen_cell=dgreen[rows,cols];dwater_cell=dwater[rows,cols]
 # Cap the modeled neighborhood influence at Ahmedabad-specific observed
 # saturation radii (vegetation 130 m; water 150 m).
 base_distance_green=np.where(dgreen_cell<=130.,np.minimum(base[:,fields['distance_to_green_m']],dgreen_cell),base[:,fields['distance_to_green_m']])
 base_distance_water=np.where(dwater_cell<=150.,np.minimum(base[:,fields['distance_to_water_m']],dwater_cell),base[:,fields['distance_to_water_m']])

 scenario_defs=[]
 # Helper returns feature counterfactual; all changes are bounded to plausible
 # input ranges and apply only within stated eligibility masks.
 def simulate(name,treated,kind,changes):
  b=base.copy()
  treated=treated&valid
  modified=treated.copy()
  if kind=='green':
   green_distance_change=base_distance_green<base[:,fields['distance_to_green_m']]
   b[:,fields['distance_to_green_m']]=base_distance_green;modified|=green_distance_change&valid
   b[treated,fields['NDVI']]=np.clip(b[treated,fields['NDVI']]+.10,-.2,.95)
   b[treated,fields['NDBI']]-=.05
   b[treated,fields['MNDWI']]=np.clip(b[treated,fields['MNDWI']]+.03,-1,1)
  elif kind=='roof':
   b[treated,fields['NDVI']]=np.clip(b[treated,fields['NDVI']]+.08,-.2,.95)
   b[treated,fields['albedo']]=np.clip(b[treated,fields['albedo']]+.04,0,.8)
   b[treated,fields['MNDWI']]=np.clip(b[treated,fields['MNDWI']]+.03,-1,1)
  elif kind=='cool':
   b[treated,fields['albedo']]=np.maximum(b[treated,fields['albedo']],.60)
  elif kind=='water':
   water_distance_change=base_distance_water<base[:,fields['distance_to_water_m']]
   b[:,fields['distance_to_water_m']]=base_distance_water;modified|=water_distance_change&valid
   for key,value in water_medians.items():b[treated,fields[key]]=value
   b[treated,fields['building_density_per_ha']]=0.;b[treated,fields['building_coverage_fraction']]=0.
   b[treated,fields['road_density_km_per_km2']]=0.;b[:,fields['distance_to_water_m']]=np.minimum(base[:,fields['distance_to_water_m']],dwater_cell)
  elif kind=='combined':
   b[green_eligible,fields['NDVI']]=np.clip(b[green_eligible,fields['NDVI']]+.10,-.2,.95);b[green_eligible,fields['NDBI']]-=.05
   b[green_eligible,fields['MNDWI']]=np.clip(b[green_eligible,fields['MNDWI']]+.03,-1,1)
   b[roof_eligible,fields['NDVI']]=np.clip(b[roof_eligible,fields['NDVI']]+.08,-.2,.95);b[roof_eligible,fields['albedo']]=np.clip(b[roof_eligible,fields['albedo']]+.04,0,.8)
   b[cool_eligible,fields['albedo']]=np.maximum(b[cool_eligible,fields['albedo']],.60)
   for key,value in water_medians.items():b[water_eligible,fields[key]]=value
   b[water_eligible,fields['building_density_per_ha']]=0.;b[water_eligible,fields['building_coverage_fraction']]=0.;b[water_eligible,fields['road_density_km_per_km2']]=0.
   b[:,fields['distance_to_green_m']]=base_distance_green
   b[:,fields['distance_to_water_m']]=base_distance_water
   modified|=(b[:,fields['distance_to_green_m']]<base[:,fields['distance_to_green_m']])
   modified|=(b[:,fields['distance_to_water_m']]<base[:,fields['distance_to_water_m']])
  # Bound counterfactual features to the hot-season training support to avoid
  # unvalidated extrapolation beyond observed material/index distributions.
  b[modified]=np.minimum(np.maximum(b[modified],support_lo),support_hi)
  ph,_,_=physics_features(b,lst,ldown);ph[modified]=np.minimum(np.maximum(ph[modified],phys_lo),phys_hi)
  pp=physics_predict(model,np.column_stack([b,ph]));pp[~valid]=np.nan
  reduction=prediction-pp;reduction[~valid]=np.nan
  return b,pp,reduction,treated

 scenarios=[
  ('urban_greening_tree_cover',green_eligible,'green','On hot sparse open cells, raise NDVI by 0.10, reduce NDBI by 0.05, add 0.03 MNDWI; proximity to new greenery updated within 130 m.'),
  ('green_roofs',roof_eligible,'roof','On hot built cells, raise NDVI by 0.08, albedo by 0.04, and MNDWI by 0.03.'),
  ('cool_roofs_pavements',cool_eligible,'cool','On built cells, raise albedo to at least 0.60.'),
  ('water_bodies',water_eligible,'water','Convert at most 1% of valid cells: hottest low-built cells >500 m from current water; use median mapped-water reflectance; update distance-to-water within 150 m.'),
  ('combined_strategies',green_eligible|roof_eligible|cool_eligible|water_eligible,'combined','Apply greening, green roofs, cool roofs/pavements, and new water footprint together; distance effects capped at 130 m for vegetation and 150 m for water.')]
 results=[]; reductions={};treated_masks={};scenario_predictions={}
 for name,mask,kind,desc in scenarios:
  b,pp,red,treated=simulate(name,mask,kind,desc);reductions[name]=red;treated_masks[name]=treated;scenario_predictions[name]=pp
  area_t=float(areas[treated].sum());benefit=valid&(red>=.1);benefit_area=float(areas[benefit].sum());benefit_pop=float(pop_est[benefit].sum())
  cooling=red[benefit]
  # Convert surface cooling to an air-temperature/Heat-Index change using the
  # empirical station-pixel slope as a scenario sensitivity, not a causal law.
  cpcb=read_rows(TABLE/'Ahmedabad_phase4_CPCB_context_check.csv')
  pairs=[r for r in cpcb if r['season']=='hot']
  surf=np.array([float(r['nearest_Landsat_LST_c']) for r in pairs]);air=np.array([float(r['air_temperature_c']) for r in pairs])
  air_slope=float(np.polyfit(surf,air,1)[0]) if len(surf)>2 else 0.0
  cpcb_raw=read_rows(P/'CPCB_Landsat_Overpass_Matched.csv');rhvals=[]
  for r in cpcb_raw:
   if any(r['item_id']==s['item_id'] for s in hot_scenes) and r.get('relative_humidity_percent'):
    try:rhvals.append(float(r['relative_humidity_percent']))
    except ValueError:pass
  rhbase=float(np.mean(rhvals)) if rhvals else climate['relative_humidity_2m']
  tair=climate['temperature_2m']; basehi=heat_index_celsius(tair,rhbase)
  hi_delta=np.array([basehi-heat_index_celsius(tair-air_slope*max(0,float(v)),rhbase) for v in np.nan_to_num(red,nan=0)])
  hi_benefit=valid&(hi_delta>=.05)
  results.append({'scenario':name,'intervention_area_km2':area_t,'valid_cells_treated':int(treated.sum()),
   'area_cooling_at_least_0_1C_km2':benefit_area,'population_benefiting_at_least_0_1C_GHSL2020':benefit_pop,
   'mean_cooling_on_benefiting_area_C':float(np.mean(cooling)) if len(cooling) else 0.,
   'maximum_cooling_C':float(np.nanmax(red[valid])),'mean_cooling_on_treated_area_C':float(np.nanmean(red[treated])) if treated.any() else 0.,
   'mean_heat_index_reduction_on_benefiting_area_C':float(np.mean(hi_delta[benefit])) if benefit.any() else 0.,
   'mean_heat_index_reduction_C_all_valid':float(np.mean(hi_delta[valid])),
   'population_with_heat_index_reduction_at_least_0_05C_GHSL2020':float(pop_est[hi_benefit].sum()),
   'comfort_bridge_air_temp_per_LST_slope':air_slope,'eligibility_and_input_change':desc})
  coolmap=np.full((h,w),NODATA,np.float32);coolmap[rows[valid],cols[valid]]=red[valid].astype(np.float32)
  outname=f'Ahmedabad_phase5_{name}_cooling_C.tif';save_raster(P/'phase5'/outname,coolmap,meta,'Predicted cooling (baseline LST minus scenario LST), degrees C')
  render_map(coolmap,meta,MAPS/f'Ahmedabad_phase5_{name}_cooling.png',f'{name.replace("_"," ").title()} — predicted daytime LST reduction')
  print(name, 'intervention km2',round(area_t,2),'cooling area',round(benefit_area,2),'max C',round(float(np.nanmax(red[valid])),2),flush=True)
 # Compare combined prediction with sum of independent effects cell by cell.
 add_names=['urban_greening_tree_cover','green_roofs','cool_roofs_pavements','water_bodies']
 additive=np.nansum(np.stack([reductions[k] for k in add_names]),axis=0)
 combo=reductions['combined_strategies'];joint=valid&np.isfinite(combo)
 inter={'mean_combined_minus_sum_individual_C':float(np.nanmean(combo[joint]-additive[joint])),
  'mean_absolute_interaction_C':float(np.nanmean(np.abs(combo[joint]-additive[joint]))),
  'share_cells_combined_cooling_less_than_sum':float(np.mean(combo[joint]<additive[joint])),
  'note':'This is model response nonlinearity and overlap under the stated input changes, not an empirical observed interaction.'}
 outcsv=TABLE/'Ahmedabad_phase5_scenario_comparison.csv'
 with outcsv.open('w',newline='',encoding='utf-8') as f:
  wr=csv.DictWriter(f,fieldnames=list(results[0]));wr.writeheader();wr.writerows(results)
 (TABLE/'Ahmedabad_phase5_interaction.json').write_text(json.dumps(inter,indent=2),encoding='utf-8')
 metaout={'scenario_model':f"Phase 4 constrained physics-feature regression, retrained on sampled scene-cell rows from all {len(scene_manifest)} selected Landsat scenes",
  'season':'hot pre-monsoon seasonal composite; predictions are for clear-sky daytime LST',
  'scenario_results':results,'interaction_test':inter,
  'comfort_metric':'NWS Heat Index counterfactual computed using city-average hot-season CPCB RH and an empirical OLS slope converting station-nearest LST change to air-temperature change; sensitivity only, not a validated microclimate model.',
  'published_comparison':{'Ahmedabad cooling radii':'local remote-sensing study reports saturation radii of 130 m for dense vegetation and 150 m for water bodies; scenario distance effects are capped to these radii.',
   'urban tree cooling':'global Landsat park study reports mean summer well-treed-park cooling around 3.4 C versus surrounding urban fabric; model scenario is a 30 m intervention response, not a park-to-neighborhood contrast.',
   'cool roofs':'field studies report roof-surface reductions around 5.6 C and peaks above 10 C; model reports mixed 30 m cell LST changes and should be smaller/not directly comparable.'},
  'limitations':['Phase 4 spatial CV R2 is high but latest-scene and cross-season transfer were weak; scenario results are conditional on fitted model and should not be treated as validated forecasts.',
    'Pure CART baseline failed the higher-albedo plausibility test, so scenario outputs use the sign-constrained physics-feature model.',
    'Surface emissivity and downwelling longwave are approximated; EF and thermal inertia are proxies; H, LE and G are not measured.',
    'Built-up/roof eligibility uses Landsat rule-based LULC, NDBI and current OSM building coverage; no parcel/roof inventory exists.',
    'Added water footprint is capped at 1% of valid AOI grid cells; intervention footprints are heuristic, not costed or site-verified.',
    'CPCB LST/air association is used only as a rough comfort sensitivity bridge; surface LST is not air temperature or UTCI/WBGT.'],
  'outputs':{'scenario_table':str(outcsv.relative_to(ROOT)),'maps':[str((MAPS/f'Ahmedabad_phase5_{r["scenario"]}_cooling.png').relative_to(ROOT)) for r in results]}}
 (REPORT/'Ahmedabad_phase5_metadata.json').write_text(json.dumps(metaout,indent=2),encoding='utf-8')
 (REPORT/'Ahmedabad_PHASE5_QA.md').write_text(make_report(results,inter,metaout),encoding='utf-8')
 print('Wrote scenario comparison:',outcsv,flush=True)


def render_map(array,meta,path,title):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 w,h,gt,_=meta;x0,dx,_,y0,_,dy=gt;extent=(x0,x0+w*dx,y0+h*dy,y0)
 fig,ax=plt.subplots(figsize=(9,8),layout='constrained');arr=np.ma.masked_where(array==NODATA,array)
 im=ax.imshow(arr,extent=extent,origin='upper',cmap='YlOrRd',vmin=0,vmax=max(.5,float(np.nanpercentile(array[array!=NODATA],98))))
 fig.colorbar(im,ax=ax,label='Predicted LST reduction (°C)');ax.set_title(title);ax.set_aspect('equal');ax.set_xlabel('Easting (m), EPSG:32643');ax.set_ylabel('Northing (m), EPSG:32643')
 fig.savefig(path,dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)


def make_report(results,interaction,meta):
 lines=['# Phase 5 cooling scenario analysis — Ahmedabad','',
  'Scenarios were simulated with the sign-constrained Phase 4 physics-feature model on the hot-season composite. Maps show model-predicted clear-sky daytime LST reduction (baseline prediction minus scenario prediction).','',
  '| Scenario | Treated area (km²) | Area cooling ≥0.1°C (km²) | Population benefiting | Mean cooling on benefited area (°C) | Maximum cooling (°C) | Mean HI reduction (°C) |',
  '|---|---:|---:|---:|---:|---:|---:|']
 for r in results:
  lines.append(f"| {r['scenario']} | {r['intervention_area_km2']:.2f} | {r['area_cooling_at_least_0_1C_km2']:.2f} | {r['population_benefiting_at_least_0_1C_GHSL2020']:.0f} | {r['mean_cooling_on_benefiting_area_C']:.3f} | {r['maximum_cooling_C']:.3f} | {r['mean_heat_index_reduction_on_benefiting_area_C']:.3f} |")
 lines += ['', '## Non-additivity test','',
  f"Combined cooling minus the sum of the four separate predictions averaged {interaction['mean_combined_minus_sum_individual_C']:.3f}°C; mean absolute difference was {interaction['mean_absolute_interaction_C']:.3f}°C, and {interaction['share_cells_combined_cooling_less_than_sum']:.1%} of valid cells had combined cooling below the sum. This is model nonlinearity plus overlapping treatment changes, not a measured causal interaction.",'',
  '## Comparison with published results','',
  f"The modeled tree/greening mean cooling on benefiting cells was {results[0]['mean_cooling_on_benefiting_area_C']:.2f}°C, similar in magnitude to global Landsat observations of well-treed park cool-island intensity (about 3.4°C in summer). Those studies compare whole parks with surrounding urban areas; this simulation changes inputs on selected 30 m cells, so it is only a magnitude check. An Ahmedabad remote-sensing study reports cooling influence saturation around 130 m for dense vegetation and 150 m for water; the scenario’s distance effects are restricted to those radii. ([global park study](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL112887), [Ahmedabad blue/green space study](https://www.ertjournal.org/index.php/pub/article/view/462))",'',
  f"The cool-roof/pavement scenario’s mean grid-cell cooling was {results[2]['mean_cooling_on_benefiting_area_C']:.2f}°C. Roof-scale field experiments report roughly 5.6°C mean surface reduction and summer differences above 10°C; roof membrane temperature is not a mixed 30 m pixel, so the quantities are not directly comparable. ([cool-roof field study](https://www.mdpi.com/2071-1030/14/8/4843))",'',
  'SOLWEIG/InVEST was not run: this project lacks a validated 3D urban geometry and neighborhood-scale meteorology, so an additional comfort/cooling-capacity model would add unsupported precision.','',
  '## Comfort estimate and physical interpretation','',
  'Scenario Heat Index changes use a rough sensitivity bridge: city-average hot-season CPCB humidity and an empirical slope from nearest-pixel Landsat LST to CPCB air temperature. This gives a first-order comfort proxy only. It is not a neighborhood air-temperature, UTCI, or WBGT model. Do not interpret the comfort column as a validated health-impact prediction.','',
  'The scenario model uses a net-radiation proxy evaluated at ERA5 air temperature to avoid target leakage. Estimated Rn evaluated at Landsat surface temperature was not used as a predictor. Longwave and emissivity are estimated; EF and thermal inertia are proxies. The model does not solve H, LE, and G separately.','',
  '## Eligibility and limitations','',
  'Urban greening targets hot, sparse open cells; green roofs target hot built cells; cool roofs/pavements target built cells with albedo below 0.6 and raise it to 0.6; new water converts the hottest sparse cells over 500 m from mapped water, capped at 1% of valid cells. Combined scenarios apply the same rules together. Intervention sites, roof inventory, planting feasibility, costs, and maintenance are not modeled.','',
  'Phase 4 cross-season transfer was weak and the pure ML forest failed the albedo plausibility check, so scenario outputs use the constrained physics-feature model. Its spatial-block validation does not validate intervention effects; all results are conditional on the model and heuristic input changes.','',
  'Reproduce with `python main.py phase5`. Outputs: `outputs/tables/Ahmedabad_phase5_scenario_comparison.csv`, interaction JSON, cooling rasters in `data/processed/phase5/`, and the maps listed in the phase5 metadata.']
 return '\n'.join(lines)+'\n'


if __name__=='__main__':main()
