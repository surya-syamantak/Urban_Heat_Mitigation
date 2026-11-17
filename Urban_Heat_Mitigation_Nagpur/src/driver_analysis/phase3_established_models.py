"""Fit Random Forest, XGBoost, and CatBoost with spatial block tuning and TreeSHAP.
Runs in the project .venv; QGIS Python prepares the predictor table and legacy analyses.
"""
from __future__ import annotations
import csv, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
try:
 import sklearn, xgboost, shap
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.impute import SimpleImputer
 from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
 from sklearn.model_selection import GroupKFold
 from xgboost import XGBRegressor
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
except ImportError as exc:
 raise SystemExit('Phase 3 native models require scikit-learn, xgboost, shap, and matplotlib in the project .venv. From the project root run .venv\\Scripts\\python.exe -m pip install -r requirements.txt.') from exc
try:
 from catboost import CatBoostRegressor
 import catboost
 CATBOOST_AVAILABLE=True
except ImportError:
 CatBoostRegressor=None
 catboost=None
 CATBOOST_AVAILABLE=False

ROOT=Path(__file__).resolve().parents[2]
TABLE=ROOT/'outputs/tables'; FIG=ROOT/'outputs/figures'; REPORT=ROOT/'outputs/reports'
SEED=42; SAMPLE_N=16000; BLOCK_M=3000; CV_SPLITS=10; EXPLAIN_N=250
FEATURES=['NDVI','NDBI','MNDWI','albedo','building_equivalents_per_ha','building_coverage_fraction',
 'mean_tagged_building_height_m','height_tag_area_coverage_fraction','road_density_km_per_km2',
 'distance_to_green_m','distance_to_water_m','svf_proxy','impervious_fraction','LULC_vegetation',
 'LULC_built','LULC_water','NDVI_x_building_density']
MODEL_SPECS=[('rf','Random Forest (scikit-learn)'),('xgb','XGBoost')]
if CATBOOST_AVAILABLE: MODEL_SPECS.append(('cat','CatBoost'))
MODEL_NAMES=[name for _,name in MODEL_SPECS]

def metrics(y,p):
 return {'n':int(len(y)),'R2':float(r2_score(y,p)),'RMSE_C':float(np.sqrt(mean_squared_error(y,p))),
  'MAE_C':float(mean_absolute_error(y,p)),'bias_C':float(np.mean(p-y))}

def make_estimator(kind,params):
 if kind=='rf':return RandomForestRegressor(**params,random_state=SEED,n_jobs=4)
 if kind=='xgb':return XGBRegressor(**params,objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=SEED,verbosity=0)
 if kind=='cat' and CatBoostRegressor is not None:
  return CatBoostRegressor(**params,loss_function='RMSE',random_seed=SEED,thread_count=4,
                          verbose=False,allow_writing_files=False)
 raise ValueError(f'Unsupported or unavailable estimator: {kind}')

def fit_predict(kind,params,xtrain,ytrain,xtest):
 imputer=SimpleImputer(strategy='median')
 xt=imputer.fit_transform(xtrain);xv=imputer.transform(xtest)
 model=make_estimator(kind,params);model.fit(xt,ytrain)
 return imputer,model,xv,model.predict(xv)

def candidate_sets():
 return {
  'rf':[
   {'n_estimators':100,'max_depth':12,'min_samples_leaf':2,'max_features':0.7},
   {'n_estimators':140,'max_depth':20,'min_samples_leaf':4,'max_features':'sqrt'}],
  'xgb':[
   {'n_estimators':140,'max_depth':3,'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,'reg_lambda':5.0,'min_child_weight':5},
   {'n_estimators':220,'max_depth':4,'learning_rate':0.03,'subsample':0.8,'colsample_bytree':0.8,'reg_lambda':10.0,'min_child_weight':10}],
  'cat':[
   {'iterations':180,'depth':5,'learning_rate':0.05,'l2_leaf_reg':5.0},
   {'iterations':260,'depth':6,'learning_rate':0.035,'l2_leaf_reg':8.0}]
 }

def model_tag(model_name):
 if model_name.startswith('Random Forest'):return 'random_forest'
 if model_name=='XGBoost':return 'xgboost'
 return 'catboost'

def load_season(frame,season):
 d=frame.loc[frame.season==season].dropna(subset=['lst_median_c','easting_m','northing_m']).copy()
 if len(d)>SAMPLE_N:d=d.sample(n=SAMPLE_N,random_state=SEED).sort_values('grid_id').reset_index(drop=True)
 else:d=d.reset_index(drop=True)
 X=d[FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
 y=pd.to_numeric(d['lst_median_c'],errors='coerce').to_numpy(dtype=float)
 east=pd.to_numeric(d['easting_m'],errors='coerce').to_numpy(dtype=float)
 north=pd.to_numeric(d['northing_m'],errors='coerce').to_numpy(dtype=float)
 bx=np.floor((east-np.nanmin(east))/BLOCK_M).astype(int);by=np.floor((north-np.nanmin(north))/BLOCK_M).astype(int)
 groups=np.array([f'{a}_{b}' for a,b in zip(bx,by)])
 return d,X,y,groups

def tune_spatial(kind,X,y,groups):
 unique=np.unique(groups)
 if len(unique)<CV_SPLITS:raise RuntimeError(f'Only {len(unique)} spatial blocks; need {CV_SPLITS}.')
 cv=GroupKFold(n_splits=CV_SPLITS);results=[];best=None
 for ci,params in enumerate(candidate_sets()[kind],1):
  pred=np.full(len(y),np.nan);folds=[]
  for fold,(tr,va) in enumerate(cv.split(X,y,groups),1):
   imp,model,xv,p=fit_predict(kind,params,X[tr],y[tr],X[va]);pred[va]=p
   folds.append({'fold':fold,'n_train':len(tr),'n_validation':len(va),**metrics(y[va],p)})
  score=metrics(y,pred);entry={'candidate':ci,'params':params,'spatial_cv':score,'folds':folds,'oof_prediction':pred}
  results.append(entry)
  if best is None or score['RMSE_C']<best['spatial_cv']['RMSE_C']:best=entry
 return best,results,len(unique)

def shap_array(explainer,values):
 out=explainer.shap_values(values,check_additivity=False) if hasattr(explainer,'model') else explainer.shap_values(values)
 if isinstance(out,list):out=out[0]
 a=np.asarray(out)
 if a.ndim==3:a=a[:,:,0]
 return a

def save_plots(shap_by_key,features_by_key,importance):
 FIG.mkdir(parents=True,exist_ok=True)
 # Seasonal comparison of native mean absolute TreeSHAP values.
 fig,axs=plt.subplots(1,2,figsize=(15,8),sharex=True)
 for ax,season in zip(axs,['hot','cool']):
  subset=[r for r in importance if r['season']==season]
  ranking=sorted(subset,key=lambda r:float(r['mean_abs_shap_c']),reverse=True)
  top=[r['driver'] for r in ranking[:10]]
  y=np.arange(len(top));offset=.18
  colors=['#2166ac','#b2182b','#1b7837']
  shifts=np.linspace(-offset,offset,len(MODEL_NAMES)) if len(MODEL_NAMES)>1 else [0]
  model_styles=[(name,colors[i],shifts[i]) for i,name in enumerate(MODEL_NAMES)]
  for model,color,shift in model_styles:
   look={r['driver']:float(r['mean_abs_shap_c']) for r in subset if r['model']==model}
   ax.barh(y+shift,[look.get(f,0) for f in top],height=.34,label=model,color=color,alpha=.88)
  ax.set_yticks(y,top);ax.invert_yaxis();ax.set_title('Hot pre-monsoon' if season=='hot' else 'Cool winter')
  ax.set_xlabel('Mean absolute native TreeSHAP value (deg C)');ax.grid(axis='x',alpha=.25);ax.legend(fontsize=8)
 fig.suptitle('Phase 3 feature importance from spatially tuned tree models')
 fig.tight_layout();fig.savefig(FIG/'Nagpur_phase3_native_feature_importance.png',dpi=220,bbox_inches='tight');plt.close(fig)
 # Native SHAP beeswarm summary, one per model and season.
 for (season,model_name),(sv,df) in shap_by_key.items():
  tag=model_tag(model_name)
  plt.figure(figsize=(10,8))
  shap.summary_plot(sv,df[FEATURES],max_display=12,show=False,plot_size=None)
  plt.title(f'{season.title()} season - {model_name} native SHAP summary')
  plt.tight_layout();plt.savefig(FIG/f'Nagpur_phase3_native_shap_summary_{season}_{tag}.png',dpi=200,bbox_inches='tight');plt.close()
 # Native SHAP dependence values, with vegetation and building-density interaction coloring.
 for model_name in MODEL_NAMES:
  tag=model_tag(model_name)
  fig,axs=plt.subplots(2,2,figsize=(13,10))
  for r,season in enumerate(['hot','cool']):
   sv,df=shap_by_key[(season,model_name)]
   for c,(feature,interaction) in enumerate([('NDVI','building_equivalents_per_ha'),('building_equivalents_per_ha','NDVI')]):
    ax=axs[r,c];color=df[interaction].to_numpy(dtype=float)
    if interaction=='building_equivalents_per_ha':color=np.log1p(np.maximum(color,0));clabel='log(1 + building equivalents/ha)'
    else:clabel='NDVI'
    sc=ax.scatter(df[feature],sv[:,FEATURES.index(feature)],c=color,cmap='viridis',s=24,alpha=.78,edgecolors='none')
    ax.axhline(0,color='#555',lw=.8);ax.set_xlabel(feature);ax.set_ylabel('Native SHAP contribution to LST (deg C)')
    ax.set_title(f"{season.title()}: {feature}");ax.grid(alpha=.2);fig.colorbar(sc,ax=ax,label=clabel)
  fig.suptitle(f'Native TreeSHAP dependence - {model_name}');fig.tight_layout(rect=(0,0,1,.96))
  fig.savefig(FIG/f'Nagpur_phase3_native_shap_dependence_{tag}.png',dpi=210,bbox_inches='tight');plt.close(fig)

def main():
 TABLE.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 predictor=TABLE/'Nagpur_phase3_predictor_table.csv'
 usecols=['season','grid_id','ward','lst_median_c',*FEATURES,'easting_m','northing_m']
 frame=pd.read_csv(predictor,usecols=usecols,low_memory=False)
 seasonal={s:load_season(frame,s) for s in ['hot','cool']}
 if not CATBOOST_AVAILABLE:
  print('CatBoost is not installed in this .venv; continuing with Random Forest and XGBoost. Install project requirements to enable CatBoost.',flush=True)
 cv_results={};best_params={};samples={}
 for season,(df,X,y,groups) in seasonal.items():
  samples[season]=(df,X,y,groups);cv_results[season]={};best_params[season]={}
  print(f'{season}: tuning on {len(y):,} samples across {len(np.unique(groups))} spatial blocks',flush=True)
  for kind,name in MODEL_SPECS:
   best,results,nblocks=tune_spatial(kind,X,y,groups);best_params[season][kind]=best['params']
   cv_results[season][name]={'best_params':best['params'],'candidate_scores':[{k:v for k,v in z.items() if k!='oof_prediction'} for z in results],
      'spatial_cv':best['spatial_cv'],'folds':best['folds'],'spatial_blocks':nblocks}
   print(f"  {name}: spatial CV R2={best['spatial_cv']['R2']:.3f}, RMSE={best['spatial_cv']['RMSE_C']:.2f} C; params={best['params']}",flush=True)
 # Cross-season transfer checks (seasonal composites are not independent year-long validation).
 temporal=[]
 for train_season in ['hot','cool']:
  test_season='cool' if train_season=='hot' else 'hot'
  train_df,Xtr,ytr,_=samples[train_season];test_df,Xte,yte,_=samples[test_season]
  for kind,name in MODEL_SPECS:
   imp,model,xv,pred=fit_predict(kind,best_params[train_season][kind],Xtr,ytr,Xte)
   temporal.append({'train_season':train_season,'test_season':test_season,'model':name,**metrics(yte,pred)})
 # Fit seasonal best estimators and compute native TreeSHAP on held-out sampled rows.
 shap_by_key={};importance=[];long_rows=[];model_specs={}
 for season,(df,X,y,groups) in samples.items():
  rng=np.random.default_rng(SEED+(season=='cool')*101);exp_ix=rng.choice(len(y),min(EXPLAIN_N,len(y)),replace=False)
  for kind,name in MODEL_SPECS:
   imputer=SimpleImputer(strategy='median');Xi=imputer.fit_transform(X);xexp=Xi[exp_ix]
   model=make_estimator(kind,best_params[season][kind]);model.fit(Xi,y)
   explainer=shap.TreeExplainer(model);sv=shap_array(explainer,xexp)
   dfexp=df.iloc[exp_ix].reset_index(drop=True).copy();shap_by_key[(season,name)]=(sv,dfexp)
   mae=np.mean(np.abs(sv),axis=0);mean=np.mean(sv,axis=0);total=float(mae.sum()) or 1.
   for j,feature in enumerate(FEATURES):importance.append({'season':season,'model':name,'driver':feature,'mean_abs_shap_c':float(mae[j]),'mean_shap_c':float(mean[j]),'share_of_total_percent':float(100*mae[j]/total)})
   model_specs[(season,name)]={'model':model,'imputer':imputer,'params':best_params[season][kind]}
   pred=model.predict(xexp)
   for i in range(len(exp_ix)):
    for j,feature in enumerate(FEATURES):long_rows.append({'season':season,'model':name,'grid_id':int(dfexp.loc[i,'grid_id']),'ward':dfexp.loc[i,'ward'],'observed_lst_c':float(y[exp_ix[i]]),'predicted_lst_c':float(pred[i]),'feature':feature,'feature_value':float(dfexp.loc[i,feature]) if pd.notna(dfexp.loc[i,feature]) else float(Xi[exp_ix[i],j]),'shap_value_c':float(sv[i,j])})
    
 # Save tables.
 versions={'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__,'shap':shap.__version__}
 if CATBOOST_AVAILABLE: versions['catboost']=catboost.__version__
 with (TABLE/'Nagpur_phase3_spatial_cv_metrics.json').open('w',encoding='utf-8') as f:json.dump({'modeling_sample_per_season':SAMPLE_N,'spatial_block_size_m':BLOCK_M,'cv_splits':CV_SPLITS,'tuning_metric':'pooled out-of-fold RMSE','estimators':cv_results,'cross_season_transfer':temporal,'versions':versions},f,indent=2)
 importance.sort(key=lambda r:(r['season'],r['model'],-r['mean_abs_shap_c']))
 with (TABLE/'Nagpur_phase3_native_ranked_drivers.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['season','model','rank','driver','mean_abs_shap_c','mean_shap_c','share_of_total_percent']);w.writeheader()
  for season in ['hot','cool']:
   for model in MODEL_NAMES:
    q=[r for r in importance if r['season']==season and r['model']==model]
    for i,row in enumerate(sorted(q,key=lambda z:-z['mean_abs_shap_c']),1):w.writerow({'season':season,'model':model,'rank':i,**{k:v for k,v in row.items() if k not in ('season','model')}})
 with (TABLE/'Nagpur_phase3_native_shap_values.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['season','model','grid_id','ward','observed_lst_c','predicted_lst_c','feature','feature_value','shap_value_c']);w.writeheader();w.writerows(long_rows)
 save_plots(shap_by_key,FEATURES,importance)
 # Update the existing Phase 3 metadata and QA report with the established model results.
 mp=REPORT/'Nagpur_phase3_metadata.json'
 if mp.exists():
  meta=json.loads(mp.read_text(encoding='utf-8'))
 else:meta={}
 meta['established_library_models']={'estimators':[f'{kind}: {name}' for kind,name in MODEL_SPECS],'tuning':f'{CV_SPLITS}-fold GroupKFold by {BLOCK_M/1000:g} km spatial blocks; two documented configurations per model and season, selected by pooled out-of-fold RMSE','spatial_cv_metrics':'outputs/tables/Nagpur_phase3_spatial_cv_metrics.json','native_shap':'TreeExplainer for fitted tree models; explanation sample is up to 250 cells per season/model','limitations':'Cross-season check uses seasonal composites and is not an independent multi-year temporal validation.'}
 meta['native_figures']=[str(p.relative_to(ROOT)).replace('\\','/') for p in sorted(FIG.glob('Nagpur_phase3_native_*.png'))]
 mp.write_text(json.dumps(meta,indent=2),encoding='utf-8')
 report=REPORT/'Nagpur_PHASE3_QA.md';txt=report.read_text(encoding='utf-8') if report.exists() else '# Phase 3 driver analysis\n'
 start='\n## Established-library models, spatial tuning, and native SHAP\n'
 if start in txt:txt=txt.split(start)[0]
 txt+=start+f'\nThe Phase 3 workflow fits {", ".join(MODEL_NAMES)} in the project `.venv`. For each hot/cool seasonal composite, it samples up to {SAMPLE_N:,} grid cells, groups them into {BLOCK_M/1000:g} km spatial blocks, evaluates two parameter configurations per model with {CV_SPLITS}-fold `GroupKFold`, and selects by pooled out-of-fold RMSE. It reports spatial-CV metrics and train-hot/test-cool plus train-cool/test-hot transfer checks.\n\n'
 txt+='### Spatial-CV results\n\n| Season | Model | R² | RMSE (°C) | MAE (°C) | Bias (°C) | Best parameters |\n|---|---|---:|---:|---:|---:|---|\n'
 for season in ['hot','cool']:
  for name in MODEL_NAMES:
   m=cv_results[season][name]['spatial_cv'];params=json.dumps(cv_results[season][name]['best_params'],sort_keys=True)
   txt+=f"| {season.title()} | {name} | {m['R2']:.3f} | {m['RMSE_C']:.3f} | {m['MAE_C']:.3f} | {m['bias_C']:.3f} | `{params}` |\n"
 txt+='\n### Cross-season transfer stress test\n\n| Train season | Test season | Model | R² | RMSE (°C) | MAE (°C) | Bias (°C) |\n|---|---|---|---:|---:|---:|---:|\n'
 for m in temporal:
  txt+=f"| {m['train_season'].title()} | {m['test_season'].title()} | {m['model']} | {m['R2']:.3f} | {m['RMSE_C']:.3f} | {m['MAE_C']:.3f} | {m['bias_C']:.3f} |\n"
 txt+='\nThe cross-season checks are held-out stress tests and were not used to select hyperparameters.\n\n'
 txt+='Native SHAP `TreeExplainer` values are calculated for up to 250 sampled cells per season/model. See `outputs/tables/Nagpur_phase3_native_ranked_drivers.csv`, `outputs/tables/Nagpur_phase3_native_shap_values.csv`, and `outputs/tables/Nagpur_phase3_spatial_cv_metrics.json`; figures are in `outputs/figures/Nagpur_phase3_native_*.png`. Earlier custom CART models and approximate path attributions remain as legacy comparison outputs.\n\n'
 txt+='Results are observational associations with seasonal clear-sky Landsat LST. The composites now pool selected scenes from the configured multi-year window; spatial blocking reduces leakage from neighboring cells, but the pooled predictor table still does not provide an independent year holdout. Cross-season scores are a transfer stress test. SHAP explains fitted-model predictions and is not causal attribution.\n'
 report.write_text(txt,encoding='utf-8')
 print('Wrote spatial-CV metrics, native SHAP tables, and figures.',flush=True)
if __name__=='__main__':main()
