#!/usr/bin/env python3
"""Post-hoc fair comparison of the four Milestone-B candidate checkpoints.

Uses the repository's existing FixedValidation and health-diagnostic path.
Does not train.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import yaml
from data.dataset import MetaDiTDataset
from assembly import build_model, load_into_model
from train.engine import build_deterministic_reference, fixed_validation_from_loader, healthy_references

CANDIDATES = {
    "jepa_vicreg": "phase_00_jepa_vicreg_best_healthy.pt",
    "jepa_vicreg2": "phase_01_jepa_vicreg2_best_healthy.pt",
    "jepa_barlow": "phase_02_jepa_barlow_best_healthy.pt",
    "lejepa": "phase_03_lejepa_best_healthy.pt",
}

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints/milestone_b/adaptive")
    p.add_argument("--out-dir", default="checkpoints/milestone_b/posthoc_candidate_eval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--mask-seed", type=int, default=12345)
    p.add_argument("--refs-seed", type=int, default=2026)
    return p.parse_args()

def rpath(x):
    p=Path(x); return p if p.is_absolute() else REPO_ROOT/p

def build(cfg, device):
    mc=cfg["model"]; w=cfg["weights"]
    m=build_model(mc, str(rpath(w["spectrum"])), device=device,
                  init_from_metadit=mc.get("init_from_metadit", True),
                  metadit_weights=str(rpath(w["metadit"])))
    m.eval(); return m

def finite(x):
    if isinstance(x,float): return x if math.isfinite(x) else None
    if isinstance(x,np.floating):
        x=float(x); return x if math.isfinite(x) else None
    return x

def sanitize(x):
    if isinstance(x,dict): return {str(k):sanitize(v) for k,v in x.items()}
    if isinstance(x,list): return [sanitize(v) for v in x]
    return finite(x)

def row_for(name, ckpt_path, obj, metrics, health, null_cos, gap):
    raw,proj,pred=health["raw"],health["proj"],health["pred"]
    goal,attn=health["goal"],health["attention"]
    sig=health["signals"]
    am=obj.get("adaptive_meta",{}); best=obj.get("best",{})
    r={
      "objective":name,"checkpoint":str(ckpt_path),
      "checkpoint_step":obj.get("step"),"phase":am.get("phase"),"phase_step":am.get("phase_step"),
      "saved_representation_status":am.get("representation_status"),
      "saved_best_cos_err":best.get("primary"),
      "eval_cos_err_r0.5":metrics["cos_err_r0.5"],"eval_null_cos_err_r0.5":null_cos,"eval_null_gap":gap,
      "health_status":health["status"],"collapse_votes":sig.get("votes",0),
      "raw_eff_rank":raw.get("eff_rank_unnorm"),"raw_eff_rank_frac":raw.get("eff_rank_frac"),
      "raw_participation":raw.get("participation"),"raw_top_eig_frac":raw.get("top_eig_frac"),
      "raw_token_std":raw.get("token_std"),"raw_pairwise_p05":raw["pairwise_cos"].get("p05"),
      "raw_pairwise_mean":raw["pairwise_cos"].get("mean"),"raw_same_token_cos":raw.get("same_token_cos"),
      "proj_eff_rank":proj.get("eff_rank_unnorm"),"proj_eff_rank_frac":proj.get("eff_rank_frac"),
      "proj_participation":proj.get("participation"),"proj_top_eig_frac":proj.get("top_eig_frac"),
      "proj_token_std":proj.get("token_std"),"proj_pairwise_p05":proj["pairwise_cos"].get("p05"),
      "proj_pairwise_mean":proj["pairwise_cos"].get("mean"),"proj_same_token_cos":proj.get("same_token_cos"),
      "pred_mean_std":pred.get("mean_std"),"pred_eff_rank":pred.get("eff_rank_unnorm"),"pred_eff_rank_frac":pred.get("eff_rank_frac"),
      "goal_token_pairwise_cosine_mean":goal.get("goal_token_pairwise_cosine_mean"),
      "goal_token_effective_rank":goal.get("goal_token_effective_rank"),
      "goal_attention_entropy_mean":attn.get("goal_attention_entropy_mean"),
      "goal_attention_peak_mass":attn.get("goal_attention_peak_mass"),
      "goal_attention_overlap_mean":attn.get("goal_attention_overlap_mean"),
    }
    for k,v in sig.get("signals",{}).items(): r[f"collapse_signal_{k}"]=bool(v)
    for k,v in sig.get("margins",{}).items(): r[f"health_margin_{k}"]=v
    return sanitize(r)

def main():
    a=args()
    if a.n_samples<2: raise ValueError("--n-samples must be >= 2")
    if a.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    dev=torch.device(a.device)
    cfg=yaml.safe_load(rpath(a.config).read_text(encoding="utf-8"))
    out=rpath(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    val_ds=MetaDiTDataset(rpath(cfg["data"]["val_split"]))
    fixed=fixed_validation_from_loader(val_ds,a.n_samples,a.batch_size,dev,ratio=a.mask_ratio,mask_seed=a.mask_seed,
                                       collapse_cfg=cfg.get("adaptive_training",{}).get("collapse",{}))
    ref=build_deterministic_reference(lambda: build(cfg,dev),seed=a.refs_seed); ref.eval()
    print(f"[posthoc] fixed n={fixed.mask_statistics['n_samples']} batches={fixed.mask_statistics['n_batches']} ratio={fixed.mask_statistics['requested_mask_ratio']}")
    rows=[]
    for name,fn in CANDIDATES.items():
        path=rpath(a.checkpoint_dir)/fn
        if not path.exists(): raise FileNotFoundError(path)
        print(f"\n[posthoc] {name}: {path}")
        model=build(cfg,dev)
        obj=torch.load(path,map_location="cpu",weights_only=False)
        load_into_model(model,obj["model"],dev); model.eval()
        refs=healthy_references(ref,fixed,proj_source=model)
        metrics,health=fixed.evaluate(model,refs["raw"],refs["proj"],goal_mode="real")
        real,null,gap=fixed.null_gap(model)
        if abs(real-metrics["cos_err_r0.5"])>1e-10:
            raise RuntimeError(f"{name}: evaluate/null_gap mismatch: {real} vs {metrics['cos_err_r0.5']}")
        rows.append(row_for(name,path,obj,metrics,health,float(null),float(gap)))
        print(f"  cos_err={metrics['cos_err_r0.5']:.9g} null={null:.9g} gap={gap:.9g} status={health['status']} votes={health['signals'].get('votes',0)}")
        del model,obj,refs
        if dev.type=="cuda": torch.cuda.empty_cache()
    rows.sort(key=lambda r:(0 if r["health_status"]=="HEALTHY" else 1,float(r["eval_cos_err_r0.5"])))
    payload={"config":str(rpath(a.config)),"fixed_validation":sanitize(fixed.mask_statistics),"mask_seed":a.mask_seed,"refs_seed":a.refs_seed,"results":rows}
    jp=out/"candidate_comparison.json"; cp=out/"candidate_comparison.csv"
    jp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with cp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("\n"+"="*110); print("MILESTONE B POST-HOC CANDIDATE COMPARISON"); print("="*110)
    print(f"{'objective':16s} {'cos_err':>12s} {'null_gap':>12s} {'eff_rank':>10s} {'rank_frac':>10s} {'p05':>10s} {'same_tok':>10s} {'status':>10s} {'votes':>5s}")
    for r in rows:
        print(f"{r['objective']:16s} {r['eval_cos_err_r0.5']:12.6g} {r['eval_null_gap']:12.6g} {r['raw_eff_rank']:10.5g} {r['raw_eff_rank_frac']:10.5g} {r['raw_pairwise_p05']:10.5g} {r['raw_same_token_cos']:10.5g} {r['health_status']:>10s} {r['collapse_votes']:5d}")
    print(f"\nJSON -> {jp}\nCSV  -> {cp}\nNo training was performed.")

if __name__=="__main__": main()
