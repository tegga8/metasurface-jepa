# Benchmark provenance

`test10_random/` holds the results of the ten held-out items used for the matched-input
comparison and for the reproduction check:

- `final_scen.json` - base model, regimes A/B/C, plus the MetaDiT-S pure-inverse entry.
- `imp1_final.json` - symmetry variant, regimes A/B/C.
- `imp1_s0|s1|s2.json` - the variant per seed.
- `metadit_test10_geoms.npz` - designs, occupancies and target spectra for those items.

Note that the `occ` array in the npz is the dataset occupancy, not the decoded design;
the decoded design is channel 0 of `geoms`. The figures live in `docs/paper/figs/`.
