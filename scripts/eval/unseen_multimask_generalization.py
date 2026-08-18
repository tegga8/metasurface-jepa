#!/usr/bin/env python3
"""Milestone B: unseen validation, multiple disjoint sample pools and masks.
No training is performed. Only val_set.mat is instantiated.
"""
from __future__ import annotations
import argparse, csv, gc, json, math, sys
from pathlib import Path
import numpy as np
import torch, yaml
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/'src'))
from data.dataset import MetaDiTDataset
from assembly import build_model, load_into_model
from train.engine import FixedValidation, build_deterministic_reference, healthy_references

CANDIDATES = {
    'jepa_vicreg':'phase_00_jepa_vicreg_best_healthy.pt',
    'jepa_vicreg2':'phase_01_jepa_vicreg2_best_healthy.pt',
    'jepa_barlow':'phase_02_jepa_barlow_best_healthy.pt',
    'lejepa':'phase_03_lejepa_best_healthy.pt',
}

def path(x):
    p=Path(x); return p if p.is_absolute() else REPO/p

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,list): return [clean(v) for v in x]
    if isinstance(x,np.floating): x=float(x)
    if isinstance(x,float) and not math.isfinite(x): return None
    return x

def build(cfg,device):
    mc,w=cfg['model'],cfg['weights']
    m=build_model(mc,str(path(w['spectrum'])),device=device,
                  init_from_metadit=mc.get('init_from_metadit',True),
                  metadit_weights=str(path(w['metadit'])))
    m.eval(); return m

def make_batches(ds,indices,batch_size,device):
    loader=DataLoader(Subset(ds,indices),batch_size=batch_size,shuffle=False,
                      num_workers=0,pin_memory=device.type=='cuda')
    return [(g.to(device,non_blocking=True),s.to(device,non_blocking=True)) for g,s in loader]

def disjoint_pools(n_total,n_pools,n_per_pool,seed):
    need=n_pools*n_per_pool
    if need>n_total: raise RuntimeError(f'Need {need} distinct validation samples, only {n_total} exist.')
    perm=np.random.RandomState(seed).permutation(n_total)
    pools=[perm[i*n_per_pool:(i+1)*n_per_pool].tolist() for i in range(n_pools)]
    flat=[i for p in pools for i in p]
    assert len(flat)==len(set(flat)), 'Pool overlap detected.'
    return pools

def evaluate(model,reference,batches,cfg,ratio,mask_seed,device):
    collapse=cfg.get('adaptive_training',{}).get('collapse',{})
    fixed=FixedValidation(batches=batches,ratio=ratio,grid=16,
        min_side=cfg['mask'].get('min_side',3),
        k_range=tuple(cfg['mask'].get('k_range',[1,4])),
        mask_seed=mask_seed,device=device,collapse_cfg=collapse)
    refs=healthy_references(reference,fixed,proj_source=model)
    metrics,health=fixed.evaluate(model,refs['raw'],refs['proj'],goal_mode='real')
    _,null_cos,null_gap=fixed.null_gap(model)
    raw,proj,pred=health['raw'],health['proj'],health['pred']
    return {
      'cos_err':float(metrics['cos_err_r0.5']),'null_cos_err':float(null_cos),'null_gap':float(null_gap),
      'health_status':health['status'],'collapse_votes':int(health['signals'].get('votes',0)),
      'raw_eff_rank':raw.get('eff_rank_unnorm'),'raw_rank_frac':raw.get('eff_rank_frac'),
      'raw_pairwise_p05':raw['pairwise_cos'].get('p05'),'raw_pairwise_mean':raw['pairwise_cos'].get('mean'),
      'raw_same_token_cos':raw.get('same_token_cos'),'proj_eff_rank':proj.get('eff_rank_unnorm'),
      'proj_rank_frac':proj.get('eff_rank_frac'),'pred_eff_rank':pred.get('eff_rank_unnorm'),
      'pred_pairwise_p05':pred['pairwise_cos'].get('p05'),'pred_pairwise_mean':pred['pairwise_cos'].get('mean'),
      'pred_mean_std':pred.get('mean_std'),'actual_mask_ratio':fixed.mask_statistics.get('actual_mask_ratio_mean')
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/milestone_b_adaptive.yaml')
    p.add_argument('--checkpoint-dir',default='checkpoints/milestone_b/adaptive')
    p.add_argument('--out-dir',default='checkpoints/milestone_b/unseen_multimask_eval')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--sample-pools',type=int,default=2)
    p.add_argument('--samples-per-pool',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--mask-ratios',type=float,nargs='+',default=[.25,.50,.75])
    p.add_argument('--mask-seeds',type=int,nargs='+',default=[1101,2202,3303,4404])
    p.add_argument('--sample-seed',type=int,default=71001)
    p.add_argument('--refs-seed',type=int,default=2026)
    a=p.parse_args(); device=torch.device(a.device)
    with open(path(a.config),encoding='utf-8') as f: cfg=yaml.safe_load(f)
    out=path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    val_path=path(cfg['data']['val_split']); train_path=path(cfg['data']['train_split'])
    ds=MetaDiTDataset(val_path)
    pools=disjoint_pools(len(ds),a.sample_pools,a.samples_per_pool,a.sample_seed)
    print(f'DEVICE: {device}')
    print(f'VALIDATION ONLY: {val_path}')
    print(f'TRAIN SPLIT NOT USED: {train_path}')
    print(f'Validation size: {len(ds)}')
    print(f'Disjoint pools: {a.sample_pools} x {a.samples_per_pool} = {a.sample_pools*a.samples_per_pool} distinct samples')
    print(f'Mask ratios: {a.mask_ratios}; independent mask seeds: {a.mask_seeds}')
    print(f'Mask conditions/sample pool: {len(a.mask_ratios)*len(a.mask_seeds)}')
    print(f'Per model sample-mask evaluations: {a.sample_pools*a.samples_per_pool*len(a.mask_ratios)*len(a.mask_seeds):,}')
    ref=build_deterministic_reference(lambda:build(cfg,device),seed=a.refs_seed); ref.eval()
    models={}
    for name,file in CANDIDATES.items():
        ck=path(a.checkpoint_dir)/file
        if not ck.exists(): raise FileNotFoundError(ck)
        m=build(cfg,device); obj=torch.load(ck,map_location='cpu',weights_only=False); load_into_model(m,obj['model'],device); m.eval(); models[name]=m
    pool_batches=[make_batches(ds,idx,a.batch_size,device) for idx in pools]
    rows=[]
    total=len(models)*len(pool_batches)*len(a.mask_ratios)*len(a.mask_seeds)
    c=0
    for pi,batches in enumerate(pool_batches):
      for ratio in a.mask_ratios:
       for mseed in a.mask_seeds:
        for name,m in models.items():
          c+=1; r=evaluate(m,ref,batches,cfg,ratio,mseed,device)
          rows.append(clean({'condition':c,'model':name,'pool':pi,'n_samples':a.samples_per_pool,
                             'mask_ratio':ratio,'mask_seed':mseed,**r}))
          print(f'[{c}/{total}] pool={pi} mask={ratio:.2f} seed={mseed} {name}: cos={r["cos_err"]:.7f} null={r["null_cos_err"]:.7f} gap={r["null_gap"]:.7f} status={r["health_status"]} votes={r["collapse_votes"]}')
    summary={}
    for name in models:
      q=[r for r in rows if r['model']==name]; x=np.array([r['cos_err'] for r in q]); g=np.array([r['null_gap'] for r in q])
      summary[name]={'conditions':len(q),'cos_err_mean':float(x.mean()),'cos_err_std':float(x.std()),'cos_err_min':float(x.min()),'cos_err_max':float(x.max()),'null_gap_mean':float(g.mean()),'null_gap_std':float(g.std()),'healthy_fraction':float(np.mean([r['health_status']=='HEALTHY' for r in q])),'collapse_votes':int(sum(r['collapse_votes'] for r in q))}
    payload={'experiment':'unseen_multimask_generalization','validation_split':str(val_path),'training_split':str(train_path),'training_split_used':False,'validation_dataset_size':len(ds),'disjoint_pools':a.sample_pools,'samples_per_pool':a.samples_per_pool,'total_distinct_validation_samples':a.sample_pools*a.samples_per_pool,'mask_ratios':a.mask_ratios,'mask_seeds':a.mask_seeds,'total_conditions':total,'per_model_sample_mask_evaluations':a.sample_pools*a.samples_per_pool*len(a.mask_ratios)*len(a.mask_seeds),'summary':summary,'conditions':rows}
    jp=out/'unseen_multimask_results.json'; cp=out/'unseen_multimask_results.csv'
    jp.write_text(json.dumps(clean(payload),indent=2,allow_nan=False),encoding='utf-8')
    fields=list(rows[0]);
    with cp.open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('\n'+'='*110); print('UNSEEN MULTI-MASK SUMMARY'); print('='*110)
    print(f"{'MODEL':16s}{'COS_MEAN':>12s}{'COS_STD':>12s}{'NULL_GAP':>12s}{'HEALTHY':>10s}{'VOTES':>8s}")
    for name,s in sorted(summary.items(),key=lambda kv:kv[1]['cos_err_mean']):
      print(f"{name:16s}{s['cos_err_mean']:12.7f}{s['cos_err_std']:12.7f}{s['null_gap_mean']:12.7f}{100*s['healthy_fraction']:9.1f}%{s['collapse_votes']:8d}")
    print(f'\nJSON -> {jp}\nCSV  -> {cp}\nNO TRAINING WAS PERFORMED.')
    del models,ref,ds,pool_batches; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()

if __name__=='__main__': main()
