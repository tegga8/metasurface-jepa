"""Run MetaDiT's released model on exactly the inputs our JEPA was given.

Same-task, same-input comparison. The condition tensor is built with the SAME
validated Meep->CST convention used for our decode, so both models receive a
bit-identical [2,301] target spectrum for each novel geometry. Nothing here is
re-derived per model.

Pipeline follows external/metadit/generate.py::generate_one_batch exactly:
  resolution 32, 500 diffusion steps, cfg_scale 4.0, CFG by duplicating the
  batch and appending a null condition of 0.5, clip_denoised=False, then the
  conditional half is taken back out.
Decoding to 64x64 uses metric.py::restore_structure unmodified.

NOTE worth recording: restore_structure builds the 64x64 output by mirroring a
32x32 quadrant (fliplr + flipud), so every MetaDiT generation is EXACTLY
mirror-symmetric BY CONSTRUCTION. That is why 300/300 sampled geometries in
MetaDiT's own released corpus are bit-identical under fliplr/flipud -- the
symmetry is a property of the output parameterisation, not of the physics. Our
JEPA decoder has no such constraint and scores 0.88-0.90.

Usage: python _metadit_decode.py <b40_meep.json> <convention.json> <out_dir> [--limit N]
"""

import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "external", "metadit"))
sys.path.insert(0, REPO)

from _fit_meep_convention import APPLY  # noqa: E402

RESOLUTION = 32
CFG_SCALE = 4.0
TIME_STEPS = 500
SEED = 0


def _load_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {what} {path}: {exc}")


# MetaDiT's modules live in external/metadit/ and are put on sys.path just
# above; they are resolved at runtime, not statically. The same import pattern
# appears in scripts/eval/reproduce_metadit_baseline.py.
from model.dit import DIT_MODEL  # type: ignore[import-not-found]  # noqa: E402
from diffusion import create_diffusion  # type: ignore[import-not-found]  # noqa: E402
from metric import restore_structure  # type: ignore[import-not-found]  # noqa: E402


def main():
    meep_path, conv_path, out_dir = sys.argv[1:4]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        sys.exit(f"could not create {out_dir}: {exc}")

    conv = _load_json(conv_path, "convention")
    entries = [e for e in _load_json(meep_path, "Meep output")["designs"]
               if e.get("passivity_ok")]
    if limit:
        entries = entries[:limit]
    print(f"conditions: {len(entries)} (passivity-clean novel geometries)")

    def to_target(e):
        t = APPLY[conv["kind"]](e, conv)
        return np.stack([t.real, t.imag]).astype(np.float32)

    cond = torch.from_numpy(np.stack([to_target(e) for e in entries]))
    print(f"condition tensor {tuple(cond.shape)}  <- identical to our JEPA input")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    diffusion = create_diffusion(str(TIME_STEPS), learn_sigma=False)
    model = DIT_MODEL["metadit_s"](diffusion=diffusion, condition_channel=301)
    model.load_state_dict(
        torch.load(os.path.join(REPO, "data/metadit/weights/metadit-small.bin"),
                   map_location="cpu"), strict=True)
    model.to(device).eval()
    print(f"metadit_s loaded strict=True, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    torch.manual_seed(SEED)
    B = cond.shape[0]
    c = cond.to(device)
    z = torch.randn(B, 3, RESOLUTION, RESOLUTION, device=device)
    z = torch.cat([z, z], 0)
    null = torch.ones_like(c) * 0.5
    c = torch.cat([c, null], 0)
    print(f"sampling: {TIME_STEPS} steps, cfg_scale={CFG_SCALE}, "
          f"batch={2 * B} (CFG), seed={SEED}")
    t0 = time.time()
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=dict(y=c, cfg_scale=CFG_SCALE), progress=False, device=device)
    samples, _ = samples.chunk(2, dim=0)
    elapsed = time.time() - t0
    print(f"sampling done in {elapsed:.1f}s "
          f"({elapsed / max(B, 1):.2f}s per item)")

    pats, names = [], []
    for j, e in enumerate(entries):
        gen = samples[j].detach().cpu()
        full = restore_structure(gen.clone())
        # channel 0 is the geometry; threshold to binary occupancy as metric.py
        # does when scoring, then mirror-consistent 64x64.
        occ = (full[0] > full[0].mean()).to(torch.uint8).numpy()
        pats.append(occ)
        names.append(e.get("name") or str(e["idx"]))
    pats = np.stack(pats)

    np.savez(os.path.join(out_dir, "metadit_decoded.npz"), patterns=pats,
             names=np.array(names),
             scalars=np.tile(np.array([2.75002396, 0.82856321, 4.48338421]),
                             (len(names), 1)))
    print(f"saved {len(pats)} generations to {out_dir}/metadit_decoded.npz")
    print(f"mean occupancy {pats.mean():.3f} "
          f"(our novel originals 0.426, MetaDiT corpus 0.428)")


if __name__ == "__main__":
    main()