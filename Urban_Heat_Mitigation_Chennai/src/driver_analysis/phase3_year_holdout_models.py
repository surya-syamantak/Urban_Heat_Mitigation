"""Compare RF, XGBoost, and CatBoost with whole-acquisition-year holdouts."""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
try:
 from catboost import CatBoostRegressor
except ImportError as exc:
 raise SystemExit('CatBoost is missing from this project Python environment. Run: python -m pip install -r requirements.txt') from exc

ROOT=Path(__file__).resolve().parents[2]
CITY=__import__('yaml').safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))['project']['city']
TABLE=ROOT/'outputs/tables'; REPORT=ROOT/'outputs/reports'; FIG=ROOT/'outputs/figures'
FEATURES=['NDVI','NDBI','MNDWI','albedo','building_equivalents_per_ha','building_coverage_fraction',
 'mean_tagged_building_height_m','height_tag_area_coverage_fraction','road_density_km_per_km2',
 'distance_to_green_m','distance_to_water_m','svf_proxy','impervious_fraction','LULC_vegetation',
 'LULC_built','LULC_water','NDVI_x_building_density']
SEED=42

def scores(y,p):
 return {'n':int(len(y)),'R2':float(r2_score(y,p)),'RMSE_C':float(np.sqrt(mean_squared_error(y,p))),
  'MAE_C':float(mean_absolute_error(y,p)),'bias_C':float(np.mean(p-y))}

def estimator(kind):
 if kind=='Random Forest':
  return RandomForestRegressor(n_estimators=160,max_depth=18,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=SEED)
 if kind=='XGBoost':
  return XGBRegressor(n_estimators=240,max_depth=4,learning_rate=.04,subsample=.8,colsample_bytree=.8,
   reg_lambda=8.,min_child_weight=5,objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=SEED,verbosity=0)
 return CatBoostRegressor(iterations=240,depth=6,learning_rate=.05,l2_leaf_reg=5,loss_function='RMSE',
  random_seed=SEED,verbose=False,thread_count=4,allow_writing_files=False,task_type='CPU')

def main():
 TABLE.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
 path=TABLE/f'{CITY}_phase3_per_scene_predictor_table.csv'
 if not path.exists(): raise FileNotFoundError(f'{path} missing. Run python main.py phase3 to build it from Phase 1 scene rasters.')
 df=pd.read_csv(path,low_memory=False)
 required={'year','season','date','item_id','grid_id','lst_median_c',*FEATURES}
 missing=sorted(required-set(df.columns))
 if missing: raise ValueError(f'Per-scene table is missing required columns: {missing}')
 df=df.dropna(subset=['year','season','lst_median_c']).copy(); df['year']=df['year'].astype(int)
 years=[int(value) for value in sorted(df.year.unique())]
 if len(years)<3: raise ValueError(f'At least 3 acquisition years are required; found {years}')
 model_names=['Random Forest','XGBoost','CatBoost']; fold_rows=[]; pred_rows=[]; pooled={}
 for season in ('hot','cool'):
  s=df.loc[df.season.astype(str).str.lower()==season].reset_index(drop=True)
  if s.empty: raise ValueError(f'No per-scene samples available for {season} season.')
  X=s[FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(float); y=pd.to_numeric(s.lst_median_c,errors='coerce').to_numpy(float)
  year=s.year.to_numpy(int); dates=s.date.astype(str).to_numpy(); item=s.item_id.astype(str).to_numpy(); gids=s.grid_id.to_numpy()
  for model_name in model_names:
   pooled_pred=np.full(len(s),np.nan)
   for held in years:
    test=np.flatnonzero(year==held); train=np.flatnonzero(year!=held)
    if not len(test): continue
    if not len(train): raise ValueError(f'No training records remain when holding out {held}.')
    imp=SimpleImputer(strategy='median'); xtr=imp.fit_transform(X[train]); xte=imp.transform(X[test])
    model=estimator(model_name); model.fit(xtr,y[train]); p=np.asarray(model.predict(xte),dtype=float).reshape(-1)
    pooled_pred[test]=p; fold_rows.append({'season':season,'model':model_name,'held_out_year':int(held),**scores(y[test],p)})
    pred_rows.extend({'season':season,'model':model_name,'held_out_year':int(held),'date':dates[i],'item_id':item[i],
     'grid_id':int(gids[i]),'observed_lst_c':float(y[i]),'predicted_lst_c':float(pj)} for i,pj in zip(test,p))
   ok=np.isfinite(pooled_pred)
   if not ok.any(): raise RuntimeError(f'No out-of-fold predictions created for {season}, {model_name}.')
   pooled[season,model_name]=scores(y[ok],pooled_pred[ok])
   print(f'{season.title()} {model_name} year-held-out: R2={pooled[season,model_name]["R2"]:.3f}, RMSE={pooled[season,model_name]["RMSE_C"]:.2f} C, MAE={pooled[season,model_name]["MAE_C"]:.2f} C (n={ok.sum():,})',flush=True)
 fold_df=pd.DataFrame(fold_rows); fold_df.to_csv(TABLE/f'{CITY}_phase3_year_holdout_metrics.csv',index=False)
 pd.DataFrame(pred_rows).to_csv(TABLE/f'{CITY}_phase3_year_holdout_predictions.csv',index=False)
 report={'evaluation':'leave-one-acquisition-year-out; separate model per season; predictions pooled over held-out years',
  'years':years,'n_records':int(len(df)),'features':FEATURES,'metrics_by_fold':fold_rows,
  'pooled_out_of_fold_metrics':{season:{model:pooled[season,model] for model in model_names} for season in ('hot','cool')},
  'models':{'Random Forest':'160 trees, max_depth=18, min_samples_leaf=3, max_features=0.8',
   'XGBoost':'240 trees, depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_lambda=8',
   'CatBoost':'240 iterations, depth=6, learning_rate=0.05, L2=5, CPU'},
  'leakage_controls':'Entire year is held out in each fold. Median imputation is fit on training rows only. Parameters are fixed before year holdout and are not tuned on test-year records.',
  'limitations':'Samples are spatially autocorrelated; year holdout tests temporal transfer but does not remove within-year spatial dependence. LULC/OSM covariates are static or seasonal.'}
 (TABLE/f'{CITY}_phase3_year_holdout_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 qa=REPORT/f'{CITY}_PHASE3_QA.md'; text=qa.read_text(encoding='utf-8') if qa.exists() else '# Phase 3 driver analysis\n'
 heading='\n## Year-held-out Random Forest, XGBoost, and CatBoost comparison\n'
 if heading in text: text=text.split(heading)[0]
 text+=heading+'\nA second, year-aware evaluation uses individual Landsat scene samples with year and season retained. Each season is modeled separately. Every fold holds out one entire acquisition year, and the reported summary pools only out-of-fold predictions. Imputation is learned from training years only; model parameters are fixed in advance, so held-out years are not used for model selection. Phase 1 scene rasters are reused; Phase 1 and Phase 2 workflows are unchanged.\n\n'
 text+='| Season | Model | Pooled held-out R² | RMSE (°C) | MAE (°C) | Bias (°C) | n |\n|---|---|---:|---:|---:|---:|---:|\n'
 for season in ('hot','cool'):
  for name in model_names:
   m=pooled[season,name]; text+=f"| {season.title()} | {name} | {m['R2']:.3f} | {m['RMSE_C']:.3f} | {m['MAE_C']:.3f} | {m['bias_C']:.3f} | {m['n']:,} |\n"
 text+='\nSee `outputs/tables/'+CITY+'_phase3_year_holdout_metrics.csv` for fold scores, the JSON for pooled scores and settings, and the predictions CSV for out-of-fold records. This is a temporal transfer comparison; it does not establish causal effects or eliminate spatial dependence.\n'
 qa.write_text(text,encoding='utf-8')
 meta_path=REPORT/f'{CITY}_phase3_metadata.json'; meta=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
 meta['year_held_out_comparison']={'models':model_names,'metrics':'outputs/tables/'+CITY+'_phase3_year_holdout_metrics.json',
  'predictions':'outputs/tables/'+CITY+'_phase3_year_holdout_predictions.csv','validation':'leave-one-acquisition-year-out, stratified by season'}
 meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
 print(f'Wrote year-held-out fold metrics and predictions under {TABLE}',flush=True)

if __name__=='__main__': main()
