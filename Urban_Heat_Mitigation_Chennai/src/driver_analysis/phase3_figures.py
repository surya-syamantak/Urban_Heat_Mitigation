"""Plot Phase 3 drivers using the project’s custom forest and path attributions."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
ROOT=Path(__file__).resolve().parents[2]
TABLE=ROOT/'outputs/tables';FIG=ROOT/'outputs/figures'

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def as_float(row,key):
    try:return float(row[key])
    except (ValueError,TypeError,KeyError):return np.nan

def make_importance(rank):
    fig,axs=plt.subplots(1,2,figsize=(13,8),sharex=True)
    for ax,season,color in zip(axs,['hot','cool'],['#d95f02','#1b9e77']):
        rr=[r for r in rank if r['season']==season]
        rr=sorted(rr,key=lambda r:float(r['forest_mean_abs_shap_c']))[-12:]
        rr=rr[::-1];names=[r['driver'] for r in rr];vals=[float(r['forest_mean_abs_shap_c']) for r in rr]
        ax.barh(np.arange(len(rr)),vals,color=color,alpha=.88)
        ax.set_yticks(np.arange(len(rr)),names);ax.invert_yaxis();ax.grid(axis='x',alpha=.25)
        ax.set_title('Hot pre-monsoon' if season=='hot' else 'Cool winter');ax.set_xlabel('Mean absolute approximate path contribution (°C)')
    fig.suptitle('Phase 3 feature importance — custom CART forest',fontsize=15)
    fig.text(.5,.01,'Held-out permutation-path Shapley approximation; 8 feature orders x up to 60 samples. Importance describes model attribution, not causation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(FIG/'Chennai_phase3_feature_importance.png',dpi=220,bbox_inches='tight');plt.close(fig)

def make_summary(rank,samples):
    fig,axs=plt.subplots(1,2,figsize=(16,9))
    cmap=plt.get_cmap('coolwarm')
    for ax,season in zip(axs,['hot','cool']):
        rr=sorted([r for r in rank if r['season']==season],key=lambda r:float(r['forest_mean_abs_shap_c']),reverse=True)[:12]
        names=[r['driver'] for r in rr]
        sr=[r for r in samples if r['season']==season and r['feature'] in names]
        rng=np.random.default_rng(42+(season=='cool'))
        for yi,name in enumerate(names):
            vals=[r for r in sr if r['feature']==name]
            sv=np.array([as_float(r,'approx_path_contribution_c') for r in vals]);fv=np.array([as_float(r,'feature_value') for r in vals])
            ok=np.isfinite(sv)&np.isfinite(fv);sv=sv[ok];fv=fv[ok]
            lo,hi=np.percentile(fv,[5,95]) if len(fv) else (0,1)
            scaled=np.clip((fv-lo)/max(hi-lo,1e-12),0,1)
            # Small deterministic jitter makes overlapping held-out attributions visible.
            y=yi+rng.uniform(-.24,.24,len(sv))
            ax.scatter(sv,y,c=scaled,cmap=cmap,vmin=0,vmax=1,s=28,alpha=.82,linewidths=0)
        ax.axvline(0,color='#444',lw=.8);ax.set_yticks(range(len(names)),names);ax.invert_yaxis()
        ax.set_xlabel('Approximate path contribution to predicted LST (°C)');ax.set_title('Hot pre-monsoon' if season=='hot' else 'Cool winter');ax.grid(axis='x',alpha=.22)
    # Reserve a dedicated colorbar axis outside both plot panels.
    fig.tight_layout(rect=(0,.04,.86,.94))
    cax=fig.add_axes([.895,.18,.018,.68])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cax)
    cb.set_label('Feature value (low to high within feature; 5th–95th percentile scaled)')
    fig.suptitle('Phase 3 approximate Shapley summary — custom CART forest',fontsize=15)
    fig.text(.5,.01,'Each dot is one held-out sample (up to 60 per season). This is an 8-permutation path approximation, not XGBoost TreeSHAP. Attribution is associational, not causal.',ha='center',fontsize=9)
    fig.savefig(FIG/'Chennai_phase3_shap_summary.png',dpi=220,bbox_inches='tight');plt.close(fig)

def make_dependence(samples):
    fig,axs=plt.subplots(2,2,figsize=(13,10))
    for row,season in enumerate(['hot','cool']):
        sr=[r for r in samples if r['season']==season]
        for col,(feature,interaction) in enumerate([('NDVI','building_equivalents_per_ha'),('building_equivalents_per_ha','NDVI')]):
            subset=[r for r in sr if r['feature']==feature]
            vals=np.array([as_float(r,'feature_value') for r in subset]);shap=np.array([as_float(r,'approx_path_contribution_c') for r in subset])
            # Attribution rows are long-form: lookup the interacting feature for the same sample.
            lookup={}
            for r in sr:
                if r['feature']==interaction:lookup[int(r['sample_index'])]=as_float(r,'feature_value')
            color=np.array([lookup.get(int(r['sample_index']),np.nan) for r in subset])
            ok=np.isfinite(vals)&np.isfinite(shap)&np.isfinite(color)
            vals,shap,color=vals[ok],shap[ok],color[ok]
            if interaction=='building_equivalents_per_ha':color=np.log1p(np.maximum(color,0));label='log(1 + building equivalents/ha)'
            else:label='NDVI'
            ax=axs[row,col];sc=ax.scatter(vals,shap,c=color,cmap='viridis',s=35,alpha=.8,edgecolors='none')
            ax.axhline(0,color='#555',lw=.8);ax.set_xlabel(feature.replace('_',' '));ax.set_ylabel('Approximate contribution to predicted LST (°C)')
            ax.set_title(('Hot' if season=='hot' else 'Cool')+' season: '+feature+' attribution');ax.grid(alpha=.2)
            cb=fig.colorbar(sc,ax=ax);cb.set_label(label)
    fig.suptitle('Phase 3 approximate Shapley dependence plots',fontsize=15)
    fig.text(.5,.01,'Dot color shows the interacting feature. Contributions come from the custom CART forest path approximation; patterns are not causal effects.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(FIG/'Chennai_phase3_shap_dependence.png',dpi=220,bbox_inches='tight');plt.close(fig)

def make_partial_dependence(response):
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for row,season in enumerate(['hot','cool']):
        sr=[r for r in response if r['season']==season]
        ax=axs[row,0]
        for stratum,color,label in [('', '#333333','Average reference'),('low_density_p50','#1b9e77','Low building density'),('high_density_p95','#d95f02','High building density')]:
            cur=[r for r in sr if r['driver']=='NDVI' and (r.get('interaction_stratum') or '')==stratum and r.get('predicted_lst_c')]
            cur=sorted(cur,key=lambda r:float(r['response_curve_x']))
            if cur:ax.plot([float(r['response_curve_x']) for r in cur],[float(r['predicted_lst_c']) for r in cur],marker='o',ms=3,color=color,label=label)
        ax.set_title(('Hot' if season=='hot' else 'Cool')+' season: NDVI');ax.set_xlabel('NDVI');ax.set_ylabel('Predicted daytime LST (°C)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax=axs[row,1];cur=[r for r in sr if r['driver']=='building_equivalents_per_ha' and not r.get('interaction_stratum') and r.get('predicted_lst_c')];cur=sorted(cur,key=lambda r:float(r['response_curve_x']))
        if cur:ax.plot([float(r['response_curve_x']) for r in cur],[float(r['predicted_lst_c']) for r in cur],marker='o',ms=4,color='#7570b3')
        ax.set_title(('Hot' if season=='hot' else 'Cool')+' season: building density');ax.set_xlabel('Building equivalents per hectare');ax.set_ylabel('Predicted daytime LST (°C)');ax.grid(alpha=.2)
    fig.suptitle('Phase 3 partial-dependence response curves',fontsize=15)
    fig.text(.5,.01,'Custom CART forest predictions with other predictors held at reference values. These are model associations, not intervention effects.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(FIG/'Chennai_phase3_partial_dependence.png',dpi=220,bbox_inches='tight');plt.close(fig)

def main():
    FIG.mkdir(parents=True,exist_ok=True)
    rank=read_csv(TABLE/'Chennai_phase3_ranked_drivers.csv')
    samples=read_csv(TABLE/'Chennai_phase3_path_attributions.csv')
    response=read_csv(TABLE/'Chennai_phase3_response_curves_and_gwr.csv')
    make_importance(rank);make_summary(rank,samples);make_dependence(samples);make_partial_dependence(response)
    print('Wrote Phase 3 figures to',FIG)
if __name__=='__main__':main()
