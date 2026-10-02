# Results ledger (append-only)

One block per phase: the **before** (baseline) numbers and the **after** numbers,
same battery, same split / config / seed. Never overwrite a block — append.
Numbers come from `checkpoints/benchmark/*.json` + the eval-scenario output.

See `ROADMAP.md` §"Standard phase protocol" and `BASELINE.md`.

---

## Baseline — current architecture (Phase 1)

- date: _TBD_   commit: _TBD_   config: `configs/unified.yaml`   checkpoint: `checkpoints/unified/latest.pt`
- artifact: `checkpoints/benchmark/baseline_scenarioA.json`

| quantity | value | notes |
|---|---|---|
| tests | _TBD_ passed / _TBD_ skipped / _TBD_ failed | |
| MAE (ours, Scenario A, test) | _TBD_ | MetaDiT-S 0.0801 |
| AAE (ours) | _TBD_ | 48.2495 |
| AAE&2 / AAE&4 | _TBD_ | analogue; paper 58.80 / 68.73 |
| AVG1 MAE / AAE | _TBD_ | 0.5860 / 352.7424 |
| NN MAE / AAE | _TBD_ | |
| surrogate floor MAE | _TBD_ | ≈0.0084 |
| A / B / C win rate | _TBD_ | gate ≥ 0.75 |
| scalar-dep. one-known / two-known | _TBD_ | **expected FAIL ≈0.50** |
| guidance gap (hard) | _TBD_ | |
| physics gradient share | _TBD_ | **expected ≈66 %** |
| masked-fill seam / roughness / locality | _TBD_ | |

---

## Phase 2 — representation-first training schedule

- date: _TBD_   commit: _TBD_   change: _
- before: (copy Baseline)
- after: _
- delta / gate: _

---

## Phase 3 — representation hygiene

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _

---

## Phase 4 — multi-target objective (spectrum + scalar in the target side)

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _
- shortcut probe (null vs real at fixed occupancy): _TBD_

---

## Phase 5 — scalar capacity & bounds

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _

---

## Phase 6 — encoder sizing

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _
