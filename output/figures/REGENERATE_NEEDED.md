# Figures pending regeneration

The following figures were not re-rendered after the fix because they
depend on the full Monte Carlo pipeline (1M paths and Tier-B bootstrap),
which can only be run on your local machine, not in the dev sandbox:

- `06_path_sanity_check.png`               — needs Champion paths tensor
- `appendix/A17_validation_stress.png`     — needs final stress table from full run
- `appendix/A19_bootstrap_param.png`       — needs Tier-B bootstrap samples

Run `make all` (or `python main.py`) once on your machine and these will
be produced automatically alongside the rest. All three have been
visually verified to compile correctly with synthetic test data — only
the input data needs to be real.

The 22 other figures (8 main + 14 appendix), all 12 tables, and
`numbers.json` were either already correct or have been re-rendered
in-place with the fixed code.
