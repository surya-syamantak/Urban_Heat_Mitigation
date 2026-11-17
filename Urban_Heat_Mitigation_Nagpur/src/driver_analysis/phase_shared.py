"""Small shared grid and raster utilities for QGIS-based Phases 4–6."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import yaml
from osgeo import gdal
from qgis.core import QgsVectorLayer
ROOT=Path(__file__).resolve().parents[2]
CFG=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))
GRID=ROOT/CFG["spatial"]["grid"]
CENTROID=ROOT/CFG["spatial"]["centroid_ward_table"]
URBAN_CSV=ROOT/"data/processed/urban_form/Nagpur_urban_form_30m.csv"

def grid_data():
    gl=QgsVectorLayer(str(GRID),"grid","ogr");cl=QgsVectorLayer(str(CENTROID),"centroid","ogr")
    if not gl.isValid() or not cl.isValid():raise RuntimeError("Could not open grid/ward lookup")
    ward={int(f["grid_id"]):str(f["Name"]) for f in cl.getFeatures()}
    feats=list(gl.getFeatures())
    ids=np.array([int(f["grid_id"]) for f in feats],np.int64)
    rows=np.array([int(f["row_index"]) for f in feats],np.int32)
    cols=np.array([int(f["col_index"]) for f in feats],np.int32)
    x=np.array([(float(f["left"])+float(f["right"]))/2 for f in feats])
    y=np.array([(float(f["top"])+float(f["bottom"]))/2 for f in feats])
    wards=np.array([ward.get(int(i),"") for i in ids],object)
    if np.any(wards==""):raise RuntimeError("Unmatched grid cell ward label")
    return ids,rows,cols,x,y,wards

def read_raster(path):
    ds=gdal.Open(str(path))
    if ds is None:raise FileNotFoundError(path)
    a=ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nd=ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None:a[a==nd]=np.nan
    a[~np.isfinite(a)]=np.nan
    meta=(ds.RasterXSize,ds.RasterYSize,ds.GetGeoTransform(),ds.GetProjection())
    ds=None
    return a,meta

def load_urban(ids):
    result={}
    with URBAN_CSV.open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:result[int(row["grid_id"])]=row
            except (ValueError,KeyError):pass
    columns=["building_equivalents_per_ha","building_coverage_fraction","mean_tagged_building_height_m",
      "height_tag_area_coverage_fraction","road_density_km_per_km2","distance_to_green_m","distance_to_water_m","svf_proxy"]
    arrays=[]
    for key in columns:
        arrays.append(np.array([float(result[int(i)].get(key) or "nan") if int(i) in result else np.nan for i in ids]))
    return columns,np.column_stack(arrays)

def median_fill(x):
    x=x.copy()
    for j in range(x.shape[1]):
        values=x[:,j];ok=np.isfinite(values);med=float(np.median(values[ok])) if ok.any() else 0.0
        values[~ok]=med;x[:,j]=values
    return x

def forest_fit(x,y,trees=24,depth=4,seed=42):
    rng=np.random.default_rng(seed);models=[]
    for _ in range(trees):
        ix=rng.integers(0,len(y),size=min(len(y),5000));X=x[ix];Y=y[ix]
        def grow(a,b,d):
            pred=float(np.mean(b))
            if d==depth or len(b)<80:return(pred,None,None,None,None)
            best=0.;choice=None
            features=rng.choice(X.shape[1],max(1,int(np.sqrt(X.shape[1]))),replace=False)
            for j in features:
                lo,hi=np.quantile(a[:,j],[.15,.85])
                if hi<=lo:continue
                for threshold in rng.uniform(lo,hi,8):
                    left=a[:,j]<=threshold;nl=int(left.sum());nr=len(b)-nl
                    if nl<30 or nr<30:continue
                    bl=b[left];br=b[~left]
                    gain=float(bl.sum()**2/nl+br.sum()**2/nr-b.sum()**2/len(b))
                    if gain>best:best=gain;choice=(j,float(threshold),left)
            if choice is None:return(pred,None,None,None,None)
            j,t,left=choice
            return(pred,j,t,grow(a[left],b[left],d+1),grow(a[~left],b[~left],d+1))
        models.append(grow(X,Y,0))
    return models

def tree_predict(tree,x):
    pred=np.empty(len(x),float)
    def descend(node,indices):
        _,j,t,left,right=node
        if j is None:pred[indices]=node[0];return
        mask=x[indices,j]<=t
        if mask.any():descend(left,indices[mask])
        if (~mask).any():descend(right,indices[~mask])
    descend(tree,np.arange(len(x)));return pred

def ensemble_predict(models,x):return np.mean([tree_predict(tree,x) for tree in models],axis=0)
