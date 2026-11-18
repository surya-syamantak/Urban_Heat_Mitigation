"""Phase 6: budgeted cell-level cooling portfolio optimization for Ahmedabad.
Uses Phase 5 counterfactual rasters as a fast, independent-cell surrogate. Costs are
relative planning units because no verified Ahmedabad intervention cost inventory exists.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import yaml
from osgeo import gdal
from qgis.core import QgsApplication, QgsVectorLayer
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/driver_analysis'))
from phase3_drivers import grid_data,read_raster,load_urban,median_fill
CFG=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
P=ROOT/'data/processed'; TAB=ROOT/'outputs/tables'; MAP=ROOT/'outputs/maps'; REP=ROOT/'outputs/reports'
SEED=int(CFG['project'].get('random_seed',42)); NODATA=-9999.
TYPES=['urban_greening_tree_cover','green_roofs','cool_roofs_pavements','water_bodies']
COST_RATE={'urban_greening_tree_cover':2.,'green_roofs':5.,'cool_roofs_pavements':1.,'water_bodies':8.}


def dominates(a,b): return (a[0]>=b[0] and a[1]<=b[1] and (a[0]>b[0] or a[1]<b[1]))
def front(pop, vals):
    n=len(pop); fronts=[]; S=[[] for _ in range(n)]; count=np.zeros(n,int); f=[]
    for i in range(n):
        for j in range(i+1,n):
            if dominates(vals[i],vals[j]): S[i].append(j);count[j]+=1
            elif dominates(vals[j],vals[i]): S[j].append(i);count[i]+=1
        if count[i]==0:f.append(i)
    while f:
        fronts.append(f); nxt=[]
        for i in f:
            for j in S[i]:
                count[j]-=1
                if count[j]==0:nxt.append(j)
        f=nxt
    return fronts

def crowding(indices, vals):
    if len(indices)<3:return np.full(len(indices),np.inf)
    d=np.zeros(len(indices))
    for k in range(2):
        order=np.argsort([vals[i][k] for i in indices]);d[order[0]]=d[order[-1]]=np.inf
        lo=vals[indices[order[0]]][k];hi=vals[indices[order[-1]]][k]
        if hi==lo:continue
        for z in range(1,len(order)-1):d[order[z]]+=(vals[indices[order[z+1]]][k]-vals[indices[order[z-1]]][k])/(hi-lo)
    return d

def render_plan(grid,meta,path,rows,cols,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch, Polygon
    w,h,gt,proj=meta
    r0=max(0,int(rows.min())-2);r1=min(h,int(rows.max())+3)
    c0=max(0,int(cols.min())-2);c1=min(w,int(cols.max())+3)
    view=grid[r0:r1,c0:c1]
    shown=np.ma.masked_equal(view,NODATA)
    colors=['#e7e8e3','#218c4b','#48a6cf','#e8a900','#7b3294']
    cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,5.5,1),cmap.N)
    fig,ax=plt.subplots(figsize=(10,10),facecolor='white')
    extent=(gt[0]+gt[1]*c0,gt[0]+gt[1]*c1,gt[3]+gt[5]*r1,gt[3]+gt[5]*r0)
    ax.imshow(shown,origin='upper',extent=extent,cmap=cmap,norm=norm,interpolation='nearest',zorder=1)
    # Add ward outlines as geographic context.
    wl=QgsVectorLayer(str(ROOT/CFG['spatial']['wards']),'wards','ogr')
    if wl.isValid():
        for feat in wl.getFeatures():
            geom=feat.geometry()
            polygons=geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]
            for poly in polygons:
                if not poly:continue
                pts=[(pt.x(),pt.y()) for pt in poly[0]]
                if len(pts)>2:ax.add_patch(Polygon(pts,closed=True,facecolor='none',edgecolor='#555555',linewidth=.35,zorder=3))
    handles=[Patch(facecolor=colors[0],edgecolor='#888',label='No intervention'),
             Patch(color=colors[1],label='Greening'),Patch(color=colors[2],label='Green roofs'),
             Patch(color=colors[3],label='Cool roofs / pavements'),Patch(color=colors[4],label='Water')]
    ax.legend(handles=handles,loc='lower left',framealpha=.95,ncol=1)
    ax.set_xlim(extent[0],extent[1]);ax.set_ylim(extent[2],extent[3]);ax.set_aspect('equal')
    ax.set_xlabel('Easting (UTM zone 43N, m)');ax.set_ylabel('Northing (UTM zone 43N, m)')
    rec=summary['recommended_budget_max_cooling_plan']
    ax.set_title('Ahmedabad Phase 6 recommended intervention plan\n%d cells | %d wards | mean direct cooling %.2f C | cost %.0f relative units' % (rec['selected_cells'],rec['wards_reached'],rec['mean_direct_cooling_C'],rec['cost_relative_units']),fontsize=13)
    ax.grid(color='white',linewidth=.25,alpha=.6,zorder=2);fig.tight_layout();fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig)


def main():
    app=QgsApplication([],False);app.initQgis();TAB.mkdir(parents=True,exist_ok=True);(P/'phase6').mkdir(parents=True,exist_ok=True);MAP.mkdir(parents=True,exist_ok=True);REP.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    ids,rows,cols,x,y,wards=grid_data();n=len(ids);un,urban=load_urban(ids);urban=median_fill(urban)
    fields={k:i for i,k in enumerate(un)}
    hotdir=P/'Landsat'/'seasonal_hot';lst_r,meta=read_raster(hotdir/'Ahmedabad_hot_LST_C_median.tif')
    def seasonal(name):return read_raster(hotdir/f'Ahmedabad_hot_{name}_median.tif')[0][rows,cols].astype(float)
    lst=lst_r[rows,cols].astype(float);ndvi=seasonal('NDVI');ndbi=seasonal('NDBI');mndwi=seasonal('MNDWI');albedo=seasonal('ALBEDO');lulc=np.nan_to_num(read_raster(hotdir/'Ahmedabad_hot_LULC_rule_based.tif')[0][rows,cols],nan=0).astype(int)
    names=['urban_greening_tree_cover','green_roofs','cool_roofs_pavements','water_bodies']; cool={k:read_raster(P/'phase5'/f'Ahmedabad_phase5_{k}_cooling_C.tif')[0][rows,cols].astype(float) for k in names}
    popr,_=read_raster(P/'phase2'/'Ahmedabad_GHSL2020_population_density_30m.tif');pdens=np.maximum(0,np.nan_to_num(popr[rows,cols]))
    cl=QgsVectorLayer(str(ROOT/CFG['spatial']['centroid_ward_table']),'cellarea','ogr'); amap={int(f['grid_id']):float(f['area_km2']) for f in cl.getFeatures()};area=np.array([amap[int(i)] for i in ids]);people=pdens*area
    cover=urban[:,fields['building_coverage_fraction']];dwater=urban[:,fields['distance_to_water_m']]
    building=(lulc==3)|(cover>=.1)|(ndbi>.15);water=(lulc==4)|(mndwi>.2)
    hot=lst>=np.nanpercentile(lst[np.isfinite(lst)],75);valid=np.isfinite(lst)&np.isfinite(people)
    elig={
      names[0]:valid&hot&~building&~water&(cover<.1)&(ndvi<.35),
      names[1]:valid&hot&building,
      names[2]:valid&building&(albedo<.6),
      names[3]:valid&hot&~building&~water&(cover<.1)&(dwater>500)}
    # Match Phase 5 cap for new-water candidate cells.
    wi=np.flatnonzero(elig[names[3]]); wi=wi[np.argsort(-lst[wi])[:int(round(.01*valid.sum()))]];wm=np.zeros(n,bool);wm[wi]=True;elig[names[3]]=wm
    # Candidate screen: retain at most 8 high-value cells per ward and type.
    # Direct treated-cell cooling only is used; neighborhood spillovers are excluded
    # here to avoid attributing one shared buffer benefit to multiple interventions.
    records=[]
    for ti,t in enumerate(names,1):
        dtemp=np.maximum(0,np.nan_to_num(cool[t],nan=0))
        for ward in np.unique(wards):
            ix=np.flatnonzero(elig[t]&(wards==ward)&(dtemp>.02)&(people>0))
            if not len(ix):continue
            score=dtemp[ix]*people[ix]
            ix=ix[np.argsort(-score)[:8]]
            for i in ix:
                cost=COST_RATE[t]*area[i]*1000
                records.append((int(i),ti,t,float(dtemp[i]),float(people[i]),float(cost),float(dtemp[i]*people[i]),str(ward)))
    # Resolve alternative types at a cell by highest benefit/cost when genes collide.
    # budget in relative cost units; one selected intervention per cell.
    budget=10000.
    cand=np.array(records,dtype=object)
    if not len(cand):raise RuntimeError('No eligible optimization candidates found.')
    cell=cand[:,0].astype(int);ct=cand[:,1].astype(int);dt=cand[:,3].astype(float);pp=cand[:,4].astype(float);cost=cand[:,5].astype(float);gain=cand[:,6].astype(float);ward=cand[:,7].astype(str)
    ratio=gain/np.maximum(cost,1e-8)
    def canonical(g):
        ix=np.flatnonzero(g)
        if not len(ix):return np.zeros(len(g),bool)
        # at most one action per grid cell
        best={}
        for j in ix:
            key=cell[j]
            if key not in best or ratio[j]>ratio[best[key]]:best[key]=j
        out=np.zeros(len(g),bool);out[list(best.values())]=True
        # repair budget by removing the least productive actions
        chosen=np.flatnonzero(out)
        if cost[chosen].sum()>budget:
            for j in chosen[np.argsort(ratio[chosen])]:
                out[j]=False
                if cost[out].sum()<=budget:break
        return out
    def evaluate(g):
        ix=np.flatnonzero(canonical(g));return float(gain[ix].sum()),float(cost[ix].sum())
    # Greedy marginal-benefit baseline under the same budget.
    order=np.argsort(-ratio);greedy=np.zeros(len(cand),bool);used=set();spent=0.
    for j in order:
        if cell[j] in used or spent+cost[j]>budget:continue
        greedy[j]=True;used.add(cell[j]);spent+=cost[j]
    greedy_value,greedy_cost=evaluate(greedy)
    # Compact binary NSGA-II implementation (maximize exposure-weighted benefit; minimize cost).
    popn=72;gens=100;pop=[]
    for _ in range(popn):
        p=rng.random(len(cand))<rng.uniform(.005,.11);pop.append(canonical(p))
    def tournament(vals,fronts):
        a,b=rng.integers(0,len(pop),2)
        rank={i:r for r,f in enumerate(fronts) for i in f}
        if rank[a]!=rank[b]:return a if rank[a]<rank[b] else b
        f=fronts[rank[a]];d=crowding(f,vals); lookup={idx:d[j] for j,idx in enumerate(f)}
        return a if lookup[a]>=lookup[b] else b
    for _ in range(gens):
        vals=[evaluate(g) for g in pop];fs=front(pop,vals);children=[]
        while len(children)<popn:
            a=pop[tournament(vals,fs)].copy();b=pop[tournament(vals,fs)].copy()
            if rng.random()<.85:
                mask=rng.random(len(cand))<.5;child=np.where(mask,a,b)
            else:child=a.copy()
            mut=rng.random(len(cand))<min(.03,2/max(1,len(cand)));child[mut]=~child[mut]
            children.append(canonical(child))
        combo=pop+children;v=[evaluate(g) for g in combo];fs=front(combo,v);new=[]
        for f in fs:
            if len(new)+len(f)<=popn:new.extend(f)
            else:
                d=crowding(f,v);new.extend([f[i] for i in np.argsort(-d)[:popn-len(new)]]);break
        pop=[combo[i] for i in new]
    pop.append(greedy.copy());vals=[evaluate(g) for g in pop];pf=front(pop,vals)[0]
    uniq={}
    for j in pf:
        b,c=vals[j]
        if c<=budget and (round(c,5) not in uniq or b>uniq[round(c,5)][0]):uniq[round(c,5)]=(b,c,pop[j])
    plist=sorted(uniq.values(),key=lambda z:z[1])
    # Select the highest exposure-weighted cooling plan on the front within budget.
    ri=int(np.argmax([p[0] for p in plist]))
    rec=canonical(plist[ri][2]);best_value,best_cost=evaluate(rec);sel=np.flatnonzero(rec)
    # predictive uncertainty proxy: +/- spatial-CV RMSE (not a causal or temporal CI).
    metrics=json.loads((TAB/'Ahmedabad_phase4_metrics.json').read_text(encoding='utf-8'));rmse=float(next(r['RMSE_C'] for r in metrics['spatial_block_cv'] if r['model']=='Physics-informed constrained regression'))
    popsel=float(pp[sel].sum());mean=float(best_value/popsel) if popsel else 0.
    cool_low=max(0.,mean-rmse);cool_high=mean+rmse
    # Write Pareto front and cell/ward allocation.
    with (TAB/'Ahmedabad_phase6_pareto_front.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['cost_relative_units','exposure_weighted_cooling_person_C','population_treated_GHSL2020','mean_direct_cell_cooling_C','selected_cells'])
        for b,c,g in plist:
            ix=np.flatnonzero(canonical(g));psum=float(pp[ix].sum());w.writerow([c,b,psum,b/psum if psum else 0,len(ix)])
    with (TAB/'Ahmedabad_phase6_recommended_cells.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['grid_id','ward','intervention','easting_utm43n_m','northing_utm43n_m','cell_area_km2','population_GHSL2020_est','predicted_direct_LST_reduction_C','exposure_weighted_benefit_person_C','relative_cost_units'])
        for j in sel:
            i=cell[j];w.writerow([ids[i],ward[j],names[ct[j]-1],x[i],y[i],area[i],pp[j],dt[j],gain[j],cost[j]])
    wardagg={}
    for j in sel:
        k=(ward[j],names[ct[j]-1]);z=wardagg.setdefault(k,[0,0.,0.,0.]);z[0]+=1;z[1]+=gain[j];z[2]+=cost[j];z[3]+=pp[j]
    with (TAB/'Ahmedabad_phase6_ward_allocation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['ward','intervention','selected_cells','exposure_weighted_benefit_person_C','cost_relative_units','population_GHSL2020_est']);
        for (a,b),v in sorted(wardagg.items()):w.writerow([a,b,*v[:3],v[3]])
    # Spatial plan raster: 0 untreated; codes 1..4 are intervention types.
    plan=np.full((meta[1],meta[0]),NODATA,np.float32);plan[rows[valid],cols[valid]]=0
    for j in sel:plan[rows[cell[j]],cols[cell[j]]]=ct[j]
    ds=gdal.GetDriverByName('GTiff').Create(str(P/'phase6'/'Ahmedabad_phase6_recommended_intervention.tif'),meta[0],meta[1],1,gdal.GDT_Float32,options=['COMPRESS=DEFLATE','TILED=YES']);ds.SetGeoTransform(meta[2]);ds.SetProjection(meta[3]);b=ds.GetRasterBand(1);b.SetNoDataValue(NODATA);b.SetDescription('0 untreated; 1 greening; 2 green roof; 3 cool roof/pavement; 4 water');b.WriteArray(plan);ds=None
    summary={'optimizer':'custom NSGA-II binary portfolio, 72 population x 100 generations; greedy marginal benefit/cost baseline','candidate_screen':'Up to 8 eligible high marginal-benefit cells per ward and type; direct treated-cell cooling only; one type per cell','budget_relative_units':budget,'cost_assumptions_relative_units_per_km2':COST_RATE,'greedy_baseline':{'cost_relative_units':greedy_cost,'exposure_weighted_cooling_person_C':greedy_value,'selected_cells':int(greedy.sum())},'recommended_budget_max_cooling_plan':{'cost_relative_units':best_cost,'exposure_weighted_cooling_person_C':best_value,'selection_rule':'highest exposure-weighted cooling on the nondominated front within the stated budget','selected_cells':int(len(sel)),'estimated_population_on_treated_cells_GHSL2020':popsel,'mean_direct_cooling_C':mean,'spatial_CV_RMSE_proxy_C':rmse,'cooling_sensitivity_range_mean_C':[cool_low,cool_high],'exposure_weighted_sensitivity_range_person_C':[max(0,best_value-rmse*popsel),best_value+rmse*popsel],'wards_reached':int(len(set(ward[sel]))),'interpretation':'Sensitivity interval applies one spatial-CV RMSE to average direct treated-cell cooling. It is not a confidence interval and does not represent failed temporal transfer.'},'pareto_solutions':len(plist),'uncertainty_note':'The Phase 4 model has weak temporal and cross-season validation. Spatial CV RMSE is used only as a broad sensitivity scale; temporal errors are much larger. Reported benefits are conditional and may not transfer across seasons.','limitations':['Costs are relative weights, not currency or procurement estimates.','Candidate screening is limited to up to eight high-value grid cells per ward and type; it is not parcel-complete.','Benefits use independent direct-cell Phase 5 scenario response and omit spillovers to prevent double counting.','Exposure uses GHSL 2020 population estimates.','Intervention effects are not causally validated; temporal holdout performance is poor.']}
    (TAB/'Ahmedabad_phase6_optimization_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    render_plan(plan,meta,MAP/'Ahmedabad_phase6_recommended_plan.png',rows[valid],cols[valid],summary)
    print(json.dumps(summary['recommended_budget_max_cooling_plan'],indent=2))
    print('Phase 6 candidates',len(cand),'Pareto',len(plist),'greedy benefit',round(greedy_value,2),'plan benefit',round(best_value,2),flush=True)
if __name__=='__main__':main()

