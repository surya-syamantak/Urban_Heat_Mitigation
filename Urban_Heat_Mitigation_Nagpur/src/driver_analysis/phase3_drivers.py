"""Phase 3 seasonal land-surface-temperature driver analysis.

Uses NumPy/SciPy/QGIS only so it can run in the project's QGIS Python runtime.
The forest and gradient-boosted tree implementations are compact CART ensembles;
permutation-path Shapley values are calculated on a held-out sample. Spatial error
GLS and GWR use explicit queen-neighborhood and adaptive-neighbor definitions.
"""
from __future__ import annotations

import csv, json, math
from pathlib import Path

import numpy as np
import yaml
from osgeo import gdal
from qgis.core import QgsVectorLayer
from scipy.ndimage import convolve
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
P = ROOT / 'data/processed'
LAND = P / 'Landsat'
URBAN_CSV = P / 'urban_form/Nagpur_urban_form_30m.csv'
GRID = ROOT / CFG['spatial']['grid']
CENTROID = ROOT / CFG['spatial']['centroid_ward_table']
ERA = P / 'ERA5_Landsat_Overpass_Matched.csv'
OUT = P / 'phase3'
TABLE = ROOT / 'outputs/tables'
MAPS = ROOT / 'outputs/maps'
REPORT = ROOT / 'outputs/reports'
SEED = int(CFG['project']['random_seed'])
NODATA = -9999.0


def grid_data():
    gl = QgsVectorLayer(str(GRID), 'grid', 'ogr')
    cl = QgsVectorLayer(str(CENTROID), 'centroid', 'ogr')
    if not gl.isValid() or not cl.isValid(): raise RuntimeError('Could not open grid/ward lookup')
    ward = {int(f['grid_id']): str(f['Name']) for f in cl.getFeatures()}
    feats = list(gl.getFeatures())
    ids = np.array([int(f['grid_id']) for f in feats], np.int64)
    rows = np.array([int(f['row_index']) for f in feats], np.int32)
    cols = np.array([int(f['col_index']) for f in feats], np.int32)
    x = np.array([(float(f['left'])+float(f['right']))/2 for f in feats])
    y = np.array([(float(f['top'])+float(f['bottom']))/2 for f in feats])
    wards = np.array([ward.get(int(i),'') for i in ids], object)
    if np.any(wards==''): raise RuntimeError('Unmatched grid cell ward label')
    return ids, rows, cols, x, y, wards


def read_raster(path):
    ds = gdal.Open(str(path))
    if ds is None: raise FileNotFoundError(path)
    a = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None: a[a==nd] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a, (ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform(), ds.GetProjection())


def raster_values(a, rows, cols): return a[rows, cols].astype(np.float64)


def load_urban(ids):
    result = {}
    with URBAN_CSV.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try: result[int(row['grid_id'])] = row
            except (ValueError,KeyError): pass
    cols = ['building_equivalents_per_ha','building_coverage_fraction','mean_tagged_building_height_m',
            'height_tag_area_coverage_fraction','road_density_km_per_km2','distance_to_green_m',
            'distance_to_water_m','svf_proxy']
    arrays=[]
    for key in cols:
        a=np.array([float(result[int(i)].get(key) or 'nan') if int(i) in result else np.nan for i in ids])
        arrays.append(a)
    return cols, np.column_stack(arrays)


def read_era():
    with ERA.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    return rows


def median_fill(x):
    x=x.copy()
    for j in range(x.shape[1]):
        v=x[:,j]; ok=np.isfinite(v)
        med=float(np.median(v[ok])) if ok.any() else 0.0
        v[~ok]=med
        x[:,j]=v
    return x


def build_season(season,ids,rows,cols,xy,ward,urban_names,urban,era_rows):
    folder=LAND/f'seasonal_{season}'
    paths={
      'LST':folder/f'Nagpur_{season}_LST_C_median.tif',
      'NDVI':folder/f'Nagpur_{season}_NDVI_median.tif',
      'NDBI':folder/f'Nagpur_{season}_NDBI_median.tif',
      'MNDWI':folder/f'Nagpur_{season}_MNDWI_median.tif',
      'Albedo':folder/f'Nagpur_{season}_ALBEDO_median.tif',
      'LULC':folder/f'Nagpur_{season}_LULC_rule_based.tif'}
    vals={}; ref=None; rasters={}
    for name,path in paths.items():
        ras,meta=read_raster(path); rasters[name]=ras; vals[name]=raster_values(ras,rows,cols)
        if ref is None: ref=meta
        elif meta[:3]!=ref[:3]: raise RuntimeError(f'Raster lattice mismatch for {path}')
    era=[r for r in era_rows if any(s['date_utc'][:10]==r['acquisition_utc'][:10] for s in scenes if s['season']==season)]
    climate_names=['temperature_2m','relative_humidity_2m','wind_speed_10m','shortwave_radiation']
    climate={k:float(np.mean([float(r[k]) for r in era if r.get(k)])) for k in climate_names}
    # Predictors available at grid scale. ERA5 is preserved in the table as
    # seasonal background context; its single city grid point is not a spatial driver.
    categorical=vals['LULC']
    urban=median_fill(urban)
    # Height coverage is retained so missing OSM height is visible to models.
    use_names=['NDVI','NDBI','MNDWI','Albedo',*urban_names,'impervious_fraction',
               'LULC_vegetation','LULC_built','LULC_water','NDVI_x_building_density']
    # Fraction proxy blends mapped built-up class with fractional OSM building cover.
    imp=np.clip(0.5*(categorical==3).astype(float)+0.5*np.nan_to_num(urban[:,1],nan=0),0,1)
    use=np.column_stack([vals['NDVI'],vals['NDBI'],vals['MNDWI'],vals['Albedo'],urban,imp,
         (categorical==2).astype(float),(categorical==3).astype(float),(categorical==4).astype(float),
         vals['NDVI']*np.log1p(np.maximum(urban[:,0],0))])
    target=vals['LST']
    valid=np.isfinite(target)&np.all(np.isfinite(use),axis=1)
    if valid.sum()<10000: raise RuntimeError(f'Too few valid {season} grid cells: {valid.sum()}')
    return dict(season=season, y=target, X=use, names=use_names, valid=valid,
                rasters=rasters, meta=ref, climate=climate)


def split_sample(valid, cap=18000):
    ix=np.flatnonzero(valid)
    rng=np.random.default_rng(SEED+len(ix))
    if len(ix)>cap: ix=rng.choice(ix,cap,replace=False)
    else: rng.shuffle(ix)
    n=len(ix); cut=int(n*.8)
    return ix[:cut],ix[cut:]


def standardize(x):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-10]=1
    return (x-mu)/sd,mu,sd


def fit_ols(x,y):
    xs,mu,sd=standardize(x)
    design=np.column_stack([np.ones(len(xs)),xs])
    beta=np.linalg.lstsq(design,y,rcond=None)[0]
    residual=y-design@beta
    dof=max(1,len(y)-design.shape[1]); s2=float(residual@residual/dof)
    cov=s2*np.linalg.pinv(design.T@design)
    return beta, np.sqrt(np.maximum(0,np.diag(cov))), mu,sd


def forest_fit(x,y,trees=24,depth=4,seed=SEED):
    rng=np.random.default_rng(seed); models=[]
    for t in range(trees):
        ix=rng.integers(0,len(y),size=min(len(y),5000)); X=x[ix]; Y=y[ix]
        def grow(a,b,d):
            pred=float(np.mean(b))
            if d==depth or len(b)<80: return (pred,None,None,None,None)
            best=0.; choice=None
            features=rng.choice(X.shape[1],max(1,int(np.sqrt(X.shape[1]))),replace=False)
            for j in features:
                lo,hi=np.quantile(a[:,j],[.15,.85])
                if hi<=lo: continue
                for threshold in rng.uniform(lo,hi,8):
                    left=a[:,j]<=threshold; nl=int(left.sum()); nr=len(b)-nl
                    if nl<30 or nr<30: continue
                    bl=b[left]; br=b[~left]
                    gain=float((bl.sum()**2/nl)+(br.sum()**2/nr)-b.sum()**2/len(b))
                    if gain>best: best=gain; choice=(j,float(threshold),left)
            if choice is None:return(pred,None,None,None,None)
            j,th,l=choice
            return (pred,j,th,grow(a[l],b[l],d+1),grow(a[~l],b[~l],d+1))
        models.append(grow(X,Y,0))
    return models


def tree_predict(tree,x):
    pred=np.empty(len(x),float)
    def descend(node,indices):
        _,j,t,left,right=node
        if j is None: pred[indices]=node[0]; return
        mask=x[indices,j]<=t
        if mask.any():descend(left,indices[mask])
        if (~mask).any():descend(right,indices[~mask])
    descend(tree,np.arange(len(x)));return pred


def ensemble_predict(models,x): return np.mean([tree_predict(t,x) for t in models],axis=0)


def boost_fit(x,y,rounds=35,depth=2,lr=.08,seed=SEED):
    base=float(y.mean());pred=np.full(len(y),base); trees=[]; rng=np.random.default_rng(seed)
    for it in range(rounds):
        residual=y-pred; sample=rng.choice(len(y),min(len(y),9000),replace=False)
        # Fit a shallow randomized regression tree to negative gradients.
        tree=forest_fit(x[sample],residual[sample],trees=1,depth=depth,seed=seed+it)[0]
        update=tree_predict(tree,x); pred+=lr*update;trees.append((lr,tree))
    return base,trees


def boost_predict(models,x):
    base,trees=models
    return base+np.sum([lr*tree_predict(t,x) for lr,t in trees],axis=0)


def tree_split_counts(node,out):
    _,j,_,left,right=node
    if j is None:return
    out[j]+=1;tree_split_counts(left,out);tree_split_counts(right,out)


def path_shap(predict,x,background,features,permutations=48):
    # Monte Carlo interventional Shapley estimates; each permutation telescopes
    # from the mean-background prediction to the explained observation.
    rng=np.random.default_rng(SEED+991); out=np.zeros((len(x),x.shape[1]),float)
    base=np.median(background,axis=0)
    for i,row in enumerate(x):
        for _ in range(permutations):
            order=rng.permutation(x.shape[1]); z=base.copy(); before=float(predict(z[None,:])[0])
            for j in order:
                z[j]=row[j]; after=float(predict(z[None,:])[0]);out[i,j]+=(after-before)/permutations;before=after
    return out


def spatial_error(x,y,rows,cols,shape):
    # Feasible GLS: estimate error autoregression from Moran's residual ratio,
    # then refit the transformed regression. This is a sparse-operator spatial
    # error approximation, not a full maximum-likelihood spatial-lag estimator.
    valid=np.isfinite(y); h,w=shape
    def surface(v):
        z=np.zeros((h,w),float);z[rows[valid],cols[valid]]=v[valid];return z
    def lag(v):
        z=surface(v); m=surface(np.where(valid,1.0,np.nan)); kernel=np.ones((3,3));kernel[1,1]=0
        den=convolve(m,kernel,mode='constant',cval=0); num=convolve(z,kernel,mode='constant',cval=0)
        out=np.full(len(y),np.nan); ok=den[rows[valid],cols[valid]]>0
        out[valid]=num[rows[valid],cols[valid]]/np.maximum(den[rows[valid],cols[valid]],1)
        return out
    xs=np.column_stack([np.ones(valid.sum()),x[valid]])
    yy=y[valid]; b=np.linalg.lstsq(xs,yy,rcond=None)[0]; e=yy-xs@b
    efull=np.full(len(y),np.nan);efull[valid]=e; we=lag(efull)[valid]
    lam=float(np.clip(np.mean(e*we)/max(np.mean(e*e),1e-9),-.8,.8))
    wy=lag(y); wx=np.column_stack([lag(x[:,j]) for j in range(x.shape[1])])
    yt=y-lam*wy; xt=x-lam*wx
    ok=np.isfinite(yt)&np.all(np.isfinite(xt),axis=1)
    b=np.linalg.lstsq(np.column_stack([np.ones(ok.sum()),xt[ok]]),yt[ok],rcond=None)[0]
    return b,lam


def gwr_coefficients(x,y,xy,valid,names,limit=180):
    # Adaptive bi-square kernel, 240 neighbors, centers sampled across AOI;
    # coefficients are mapped to the nearest sampled center for a compact map.
    ix=np.flatnonzero(valid); rng=np.random.default_rng(SEED+73)
    centers=np.sort(rng.choice(ix,min(limit,len(ix)),replace=False))
    train_tree=cKDTree(xy[ix]); center_xy=xy[centers]
    k=min(240,len(ix)); dist,nearest=train_tree.query(center_xy,k=k)
    local=[]
    for ci,inds,d in zip(centers,nearest,dist):
        ids=ix[np.asarray(inds)]; bandwidth=max(float(np.max(d)),1.)
        wt=np.maximum(1-(d/bandwidth)**2,0)**2
        z=np.column_stack([np.ones(k),x[ids]]); sw=np.sqrt(wt)
        b=np.linalg.lstsq(z*sw[:,None],y[ids]*sw,rcond=None)[0];local.append(b)
    loc=np.asarray(local)
    _,nearest_center=cKDTree(center_xy).query(xy,k=1)
    return loc,nearest_center


def main():
    global scenes
    OUT.mkdir(parents=True,exist_ok=True);TABLE.mkdir(parents=True,exist_ok=True);MAPS.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    with (LAND/'Landsat_scene_manifest.csv').open(newline='',encoding='utf-8') as f: scenes=list(csv.DictReader(f))
    ids,rows,cols,xcoord,ycoord,ward=grid_data(); xy=np.column_stack([xcoord,ycoord])
    urban_names,urban=load_urban(ids); era_rows=read_era()
    all_rank=[]; all_response=[]; interaction_rows=[]; all_shap_samples=[]; summaries=[]; climate_by_season={}
    predictor_path=TABLE/'Nagpur_phase3_predictor_table.csv'
    field=['season','grid_id','ward','lst_median_c','NDVI','NDBI','MNDWI','albedo',*urban_names,
           'impervious_fraction','LULC_vegetation','LULC_built','LULC_water','NDVI_x_building_density',
           'era5_temperature_2m','era5_relative_humidity_2m','era5_wind_speed_10m','era5_shortwave_radiation',
           'easting_m','northing_m']
    with predictor_path.open('w',newline='',encoding='utf-8') as outcsv:
      writer=csv.DictWriter(outcsv,fieldnames=field);writer.writeheader()
      for season in ['hot','cool']:
        d=build_season(season,ids,rows,cols,xy,ward,urban_names,urban,era_rows); climate=d['climate'];climate_by_season[season]=climate
        X=d['X']; y=d['y']; valid=d['valid']; names=d['names']; ras=d['rasters']; meta=d['meta']
        lulc=ras['LULC'][rows,cols]
        for i in range(len(ids)):
          if not valid[i]: continue
          vals=X[i]; rowout={'season':season,'grid_id':int(ids[i]),'ward':ward[i],'lst_median_c':float(y[i]),
            'NDVI':vals[0],'NDBI':vals[1],'MNDWI':vals[2],'albedo':vals[3]}
          rowout.update({n:float(vals[4+j]) for j,n in enumerate(urban_names)})
          rowout.update({'impervious_fraction':float(vals[12]),'LULC_vegetation':float(vals[13]),
             'LULC_built':float(vals[14]),'LULC_water':float(vals[15]),'NDVI_x_building_density':float(vals[16]),**
             {f'era5_{k}':v for k,v in climate.items()}});rowout.update({'easting_m':float(xcoord[i]),'northing_m':float(ycoord[i])});writer.writerow(rowout)
        train,test=split_sample(valid)
        Xtr,mu,sd=standardize(X[train]); Xte=(X[test]-mu)/sd; ytr=y[train];yte=y[test]
        ols,se,_,_=fit_ols(X[train],ytr)
        # robust metric on a held-out random 20% sample
        olsp=np.column_stack([np.ones(len(Xte)),Xte])@fit_ols(X[train],ytr)[0]
        rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
        r2=lambda a,b:float(1-np.sum((a-b)**2)/max(np.sum((a-a.mean())**2),1e-9))
        forest=forest_fit(Xtr,ytr,trees=24,depth=4,seed=SEED+(season=='cool'))
        fp=ensemble_predict(forest,Xte)
        boosted=boost_fit(Xtr,ytr,rounds=28,depth=2,lr=.08,seed=SEED+int(season=='cool')*100)
        bp=boost_predict(boosted,Xte)
        split_counts=np.zeros(len(names),float)
        for _,tree in boosted[1]:tree_split_counts(tree,split_counts)
        # Tree Shapley approximation from random feature permutations on a holdout sample.
        explain=Xte[:min(60,len(Xte))]; background=Xtr[np.linspace(0,len(Xtr)-1,min(500,len(Xtr))).astype(int)]
        shp=path_shap(lambda z:ensemble_predict(forest,z),explain,background,features=names,permutations=8)
        importance=np.mean(np.abs(shp),axis=0); signed=np.mean(shp,axis=0)
        for row_i in range(len(explain)):
          sample_i=int(test[row_i]); sample={'season':season,'sample_index':sample_i,'observed_lst_c':float(y[sample_i])}
          for col_i,feature in enumerate(names):
            all_shap_samples.append({'season':season,'sample_index':sample_i,'observed_lst_c':float(y[sample_i]),
              'feature':feature,'feature_value':float(explain[row_i,col_i]*sd[col_i]+mu[col_i]),
              'approx_path_contribution_c':float(shp[row_i,col_i])})
        # GWR on six core continuous drivers to avoid local collinearity from class dummies.
        gwr_cols=[0,1,2,3,4,5,6,8,9,10,11,16]
        gnames=[names[j] for j in gwr_cols]
        gx=np.zeros((len(X),len(gwr_cols)),float)
        gxv,gmu,gsd=standardize(X[valid][:,gwr_cols]);gx[valid]=gxv
        loc,nearest=gwr_coefficients(gx,y,xy,valid,gnames)
        local_beta=loc[nearest]
        # OLS table: standardized coefficients plus original-scale NDVI effect.
        beta,stderr,meanx,stdx=fit_ols(X[train],ytr)
        # standardized coefficient (from scaled design)
        for j,n in enumerate(names):
          all_rank.append({'season':season,'driver':n,'ols_standardized_beta':float(beta[j+1]),
             'ols_se':float(stderr[j+1]),'forest_mean_abs_shap_c':float(importance[j]),
             'forest_mean_shap_c':float(signed[j]),'boost_split_count':int(split_counts[j])})
        # GWR slopes summarize spatial variation in scaled driver units.
        for j,n in enumerate(gnames):
          arr=loc[:,j+1]; all_response.append({'season':season,'driver':n,'gwr_coefficient_median':float(np.median(arr)),
            'gwr_coefficient_p10':float(np.percentile(arr,10)),'gwr_coefficient_p90':float(np.percentile(arr,90)),
            'gwr_positive_center_share':float(np.mean(arr>0))})
        # Spatial error regression on whole raster (valid mask represented in master-grid coordinates).
        h,w=meta[1],meta[0]
        feature_surface=np.full((len(ids),X.shape[1]),np.nan);feature_surface[:]=X
        sb,lam=spatial_error(X,y,rows,cols,(h,w))
        for j,n in enumerate(names):
          next(r for r in all_rank if r['season']==season and r['driver']==n)['spatial_error_coefficient']=float(sb[j+1])
        # Per-cell local driver dominance: max absolute standardized local GWR slope.
        dom=np.argmax(np.abs(local_beta[:,1:]),axis=1)
        dmap=np.full((h,w),-9999,np.float32);valid_idxs=np.flatnonzero(valid)
        dmap[rows[valid_idxs],cols[valid_idxs]]=dom[valid_idxs].astype(np.float32)+1
        # Write GeoTIFF dominance classes and local NDVI slope.
        from osgeo import gdal
        def save_map(array,path,desc,dtype=gdal.GDT_Float32,nodata=-9999):
          ds=gdal.GetDriverByName('GTiff').Create(str(path),w,h,1,dtype,options=['COMPRESS=DEFLATE','TILED=YES'])
          ds.SetGeoTransform(meta[2]);ds.SetProjection(meta[3]);b=ds.GetRasterBand(1);b.SetNoDataValue(nodata);b.SetDescription(desc);b.WriteArray(array);ds=None
        save_map(dmap,OUT/f'Nagpur_{season}_local_driver_dominance.tif','Nearest adaptive GWR center: dominant absolute local slope')
        ndvi_idx=gnames.index('NDVI'); sl=np.full((h,w),-9999,np.float32);sl[rows[valid_idxs],cols[valid_idxs]]=local_beta[valid_idxs,ndvi_idx+1].astype(np.float32)
        save_map(sl,OUT/f'Nagpur_{season}_local_NDVI_slope.tif','Local GWR NDVI coefficient; C per standardized NDVI')
        # Model metrics + coefficients; spatial-error residual explanatory fit.
        metrics={'season':season,'n_valid_cells':int(valid.sum()),'n_train':int(len(train)),'n_test':int(len(test)),
          'OLS_RMSE_C':rmse(yte,olsp),'OLS_R2':r2(yte,olsp),'RandomForest_RMSE_C':rmse(yte,fp),'RandomForest_R2':r2(yte,fp),
          'GradientBoostedTrees_RMSE_C':rmse(yte,bp),'GradientBoostedTrees_R2':r2(yte,bp),
          'spatial_error_rho_moran_ratio':lam,'spatial_error_n':int(valid.sum()),'GWR_centers':int(len(loc)),
          'GWR_adaptive_neighbors':240,'ERA5_background_context':climate}
        summaries.append(metrics)
        # Response curves: tree partial dependence over 10-90th percentile, on holdout.
        for feature in ['NDVI','building_equivalents_per_ha','building_coverage_fraction','distance_to_water_m','Albedo']:
          j=names.index(feature); grid=np.quantile(Xte[:,j],[.1,.2,.3,.4,.5,.6,.7,.8,.9]); xx=np.repeat(np.median(Xte,axis=0)[None,:],len(grid),axis=0);xx[:,j]=grid
          pdp=ensemble_predict(forest,xx)
          for g,pred in zip(grid,pdp): all_response.append({'season':season,'driver':feature,'response_curve_x':float(g*sd[j]+mu[j]),
             'predicted_lst_c':float(pred),'gwr_coefficient_median':None,'gwr_coefficient_p10':None,'gwr_coefficient_p90':None,'gwr_positive_center_share':None})
        # Vegetation response at sparse and dense built-form levels. Recompute
        # the NDVI×density interaction when moving NDVI along the curve.
        ndvi_j=names.index('NDVI'); density_j=names.index('building_equivalents_per_ha')
        # Most cells carry no mapped OSM building polygons; compare the median
        # empty cell with the upper tail so the dense class is actually populated.
        density_values=np.quantile(X[valid,density_j],[.5,.95])
        ndvi_values=np.quantile(X[valid,ndvi_j],np.linspace(.1,.9,9))
        interaction_summary={}
        for label,density_value in zip(['low_density_p50','high_density_p95'],density_values):
          raw=np.repeat(np.median(X[train],axis=0)[None,:],len(ndvi_values),axis=0)
          raw[:,ndvi_j]=ndvi_values;raw[:,density_j]=density_value
          raw[:,names.index('NDVI_x_building_density')]=ndvi_values*np.log1p(max(0.,density_value))
          curve=ensemble_predict(forest,(raw-mu)/sd)
          slope=float(np.polyfit(ndvi_values,curve,1)[0]*.1)
          interaction_summary[label+'_lst_change_c_per_0_1_ndvi']=slope
          int_j=names.index('NDVI_x_building_density')
          ols_slope=float(.1*beta[ndvi_j+1]/stdx[ndvi_j] + .1*beta[int_j+1]*np.log1p(max(0.,density_value))/stdx[int_j])
          interaction_summary['OLS_'+label+'_lst_change_c_per_0_1_ndvi']=ols_slope
          interaction_rows.append({'season':season,'density_group':label,'density_p50_or_p95_building_eq_per_ha':float(density_value),
            'forest_partial_dependence_lst_change_c_per_0_1_ndvi':slope,'OLS_interaction_lst_change_c_per_0_1_ndvi':ols_slope})
          for ndvi_val,pred in zip(ndvi_values,curve):
            all_response.append({'season':season,'driver':'NDVI','response_curve_x':float(ndvi_val),
              'predicted_lst_c':float(pred),'interaction_stratum':label,'ndvi_slope_c_per_0_1':slope,
              'gwr_coefficient_median':None,'gwr_coefficient_p10':None,'gwr_coefficient_p90':None,'gwr_positive_center_share':None})
        metrics['vegetation_density_interaction']=interaction_summary
        print(f"{season}: valid={valid.sum():,}; OLS R2={metrics['OLS_R2']:.3f}; RF R2={metrics['RandomForest_R2']:.3f}; GBT R2={metrics['GradientBoostedTrees_R2']:.3f}",flush=True)
    # Write consolidated tables.
    rankpath=TABLE/'Nagpur_phase3_ranked_drivers.csv'
    with rankpath.open('w',newline='',encoding='utf-8') as f:
      fields=['season','rank_by_mean_abs_shap','driver','ols_standardized_beta','ols_se','spatial_error_coefficient','forest_mean_abs_shap_c','forest_mean_shap_c','boost_split_count']
      wri=csv.DictWriter(f,fieldnames=fields);wri.writeheader()
      for season in ['hot','cool']:
        vals=[r for r in all_rank if r['season']==season]; vals.sort(key=lambda r:-r['forest_mean_abs_shap_c'])
        for i,r in enumerate(vals,1):wri.writerow({'rank_by_mean_abs_shap':i,**r})
    resp=TABLE/'Nagpur_phase3_response_curves_and_gwr.csv'
    with resp.open('w',newline='',encoding='utf-8') as f:
      fields=['season','driver','response_curve_x','predicted_lst_c','interaction_stratum','ndvi_slope_c_per_0_1','gwr_coefficient_median','gwr_coefficient_p10','gwr_coefficient_p90','gwr_positive_center_share']
      wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(all_response)
    shap_path=TABLE/'Nagpur_phase3_path_attributions.csv'
    with shap_path.open('w',newline='',encoding='utf-8') as f:
      wr=csv.DictWriter(f,fieldnames=['season','sample_index','observed_lst_c','feature','feature_value','approx_path_contribution_c']);wr.writeheader();wr.writerows(all_shap_samples)
    met=TABLE/'Nagpur_phase3_model_metrics.json';met.write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    interpath=TABLE/'Nagpur_phase3_vegetation_density_interaction.csv'
    with interpath.open('w',newline='',encoding='utf-8') as f:
      wr=csv.DictWriter(f,fieldnames=['season','density_group','density_p50_or_p95_building_eq_per_ha',
        'forest_partial_dependence_lst_change_c_per_0_1_ndvi','OLS_interaction_lst_change_c_per_0_1_ndvi']);wr.writeheader();wr.writerows(interaction_rows)
    # Dominance and coefficient figures by season.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    for season in ['hot','cool']:
      arr,_=read_raster(OUT/f'Nagpur_{season}_local_driver_dominance.tif')
      fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
      driver_colors=plt.get_cmap('tab20').colors[:len(gnames)]
      im=ax.imshow(np.ma.masked_invalid(arr),cmap=ListedColormap(driver_colors),vmin=.5,vmax=len(gnames)+.5,origin='upper',interpolation='nearest')
      cbar=fig.colorbar(im,ax=ax,ticks=np.arange(1,len(gnames)+1),shrink=.82)
      cbar.ax.set_yticklabels(gnames,fontsize=7)
      ax.set_title(f'Nagpur {season} season: local driver dominance\nAdaptive GWR local coefficient magnitude')
      ax.set_axis_off();fig.savefig(MAPS/f'Nagpur_{season}_local_driver_dominance.png',dpi=180,bbox_inches='tight');plt.close(fig)
    metadata={'modeling_period':f"{CFG['study']['start_date']} to {CFG['study']['end_date']}",'seasonal_scenes':{season:len({r['date_utc'][:10] for r in scenes if r['season']==season}) for season in ['hot','cool']},'seasonal_item_scenes':{season:sum(r['season']==season for r in scenes) for season in ['hot','cool']},
      'grid_cells':int(len(ids)),'predictor_table':str(predictor_path.relative_to(ROOT)),
      'predictors':field[4:-6],
      'path_attribution_samples':str(shap_path.relative_to(ROOT)),
      'figures':['outputs/figures/Nagpur_phase3_feature_importance.png','outputs/figures/Nagpur_phase3_shap_summary.png','outputs/figures/Nagpur_phase3_shap_dependence.png','outputs/figures/Nagpur_phase3_partial_dependence.png'],'omitted_predictor_note':'ERA5 values are one city-grid background series per overpass, so they are catalogued in the predictor table but not used to explain within-city spatial patterns. Impervious fraction blends built-up LULC with OSM building coverage; OSM height and SVF are incomplete-tag proxies.',
      'spatial_model':'one-step feasible-GLS spatial error model using 3x3 queen neighbors; rho estimated by residual lag ratio and clipped to [-0.8,0.8]; coefficient estimates are approximate, not full ML.',
      'gwr':'adaptive bi-square kernel, 240 nearest valid cells, 180 spatially distributed centers; continuous-driver subset; dominance raster uses nearest center. Dominance class numbers correspond to '+', '.join(f'{i+1}={n}' for i,n in enumerate(gnames))+'.',
      'legacy_custom_baseline':'24 bootstrap CART trees, max depth 4; custom gradient-boosted shallow CART residual trees; 8-permutation path attribution on up to 60 held-out cells. These are retained as exploratory comparators; the phase3 workflow subsequently fits scikit-learn RandomForestRegressor, XGBoost, and CatBoost models with spatial-block tuning and native SHAP in the project Python environment.',
      'notes':'Seasonal LST composites pool clear-sky daytime acquisitions across the configured multi-year period; associations are observational, not causal. The predictor table includes all valid cells; tree/GWR fitting uses deterministic samples for practical runtime.',
      'metrics':summaries}
    (REPORT/'Nagpur_PHASE3_QA.md').write_text(report_text(metadata,all_rank,all_response),encoding='utf-8')
    (REPORT/'Nagpur_phase3_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Wrote {predictor_path}, {rankpath}, {resp}',flush=True)


def report_text(meta,rank,response):
  lines=['# Phase 3 driver analysis — Nagpur','',
    '## What was produced','',
    '- Full grid-cell seasonal predictor table (valid LST cells only), with Landsat LST/NDVI/NDBI/MNDWI/albedo/rule-based LULC, OSM urban-form metrics, and ERA5 background context.',
    '- Seasonal OLS, one-step feasible spatial-error GLS, CART forest, gradient-boosted tree, and adaptive GWR results.',
    '- Predictor ranking table with standardized OLS coefficients and held-out permutation-path Shapley importance.',
    '- Response curves and local GWR driver dominance / NDVI slope maps for both seasons.','',
    '## Legacy custom-model random-split metrics','', '| Season | Model | Test R² | Test RMSE (°C) |','|---|---:|---:|---:|']
  for s in meta['metrics']:
    for model,key in [('OLS','OLS'),('Random forest','RandomForest'),('Gradient boosted trees','GradientBoostedTrees')]:
      lines.append(f"| {s['season']} | {model} | {s[key+'_R2']:.3f} | {s[key+'_RMSE_C']:.3f} |")
  lines += ['', '## Seasonal driver ranking and vegetation interaction','']
  for season in ['hot','cool']:
    top=sorted([r for r in rank if r['season']==season],key=lambda r:-r['forest_mean_abs_shap_c'])[:5]
    names=', '.join(f"{r['driver']} ({r['forest_mean_abs_shap_c']:.3f}°C mean |path contribution|)" for r in top)
    m=next(x for x in meta['metrics'] if x['season']==season)['vegetation_density_interaction']
    lines.append(f"- **{season.title()}** top tree-attribution drivers: {names}. NDVI change per +0.1 NDVI (low/high density, OSM building equivalents p50/p95): forest {m['low_density_p50_lst_change_c_per_0_1_ndvi']:+.3f}/{m['high_density_p95_lst_change_c_per_0_1_ndvi']:+.3f}°C; OLS interaction {m['OLS_low_density_p50_lst_change_c_per_0_1_ndvi']:+.3f}/{m['OLS_high_density_p95_lst_change_c_per_0_1_ndvi']:+.3f}°C.")
  lines += ['', '## Important interpretation','',
    'These models describe associations with clear-sky Landsat daytime surface temperature; they do not establish causal cooling. Hot and cool observations are drawn from the multi-year seasonal sample recorded in the Phase 1 manifest. ERA5 is a city-scale background point repeated across grid cells, so it is included in the table but cannot explain within-city spatial differences. The seasonal LULC product is a threshold-based proxy. Impervious fraction blends built-up LULC with OSM building coverage, OSM is a current snapshot, and OSM height/SVF are incomplete-tag proxies.', '',
    'The spatial-error estimate uses one-step feasible GLS with a 3×3 queen-neighbor operator and a residual lag-ratio estimate clipped to ±0.8; it is not full maximum-likelihood spatial econometrics. A fitted value at the cap indicates the spatial parameter is not well pinned down by this approximation. GWR uses a 240-cell adaptive bi-square kernel at 180 sample centers, then assigns each cell the nearest center coefficients for mapping. Legacy CART Shapley contributions use an 8-permutation path estimate on up to 60 held-out cells; the subsequent project-Python step reports native TreeSHAP for Random Forest, XGBoost, and CatBoost.', '',
    '## Reproduction','', 'Run `python main.py phase3` from the project root. The modeling code is `src/driver_analysis/phase3_drivers.py`.']
  return '\n'.join(lines)+'\n'


if __name__=='__main__': main()
