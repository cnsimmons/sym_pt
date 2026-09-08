# sym_pt pipeline — run order

Cross-sectional OTC resection study. Run top to bottom; each stage reads the
stage above it.

Lines marked **[unverified]** were not checked directly and need confirming
before anyone relies on them.

---

## 0 · Config

Two parameter files exist and they are **not** interchangeable.

| file | used by | `skip_subs` | `skip_codes` | `should_skip()` | `pre_surgery_sessions` |
|---|---|---|---|---|---|
| `params.py` | `D_liu/verified/*` | `['017']` | `set()` | yes | yes |
| `sym_pt_params.py` | `A_preprocessing/*`, `tfce_votc_contrasts.py`, `calc_peak_coords_mni.py`, `extract_selective_voxel_counts.py` | `['017']` | `set()` | yes | yes |

Both now carry the same exclusions, but only after an edit in Sept 2026 —
`sym_pt_params.py` previously had `skip_subs = []` and no `should_skip` or
`pre_surgery_sessions` at all. Scripts importing it therefore excluded nobody
and could not drop pre-surgical sessions. **Consolidating these two files into
one is outstanding work.** `params.py` is the superset.

Subject table: `sub_info.csv` at the repo root, one row per subject-session.
This is the single source of truth for group, intact hemisphere, sex, and age.

---

## 1 · Preprocessing — `A_preprocessing/`

Imports `sym_pt_params.py`. Produces the FEAT output and registrations that
everything downstream reads from `/user_data/csimmon2/sym_pt/{sub}/{ses}/`.

Not rerun in normal analysis work.

---

## 2 · Searchmasks

`01_create_searchmasks.py` — Harvard-Oxford derived, 25% threshold, no
dilation. ROI index definitions are in the script.

**Stale-mask trap:** the script has skip-if-exists logic, so a definition
change does not propagate unless the old files are deleted first. Masks were
rewritten April 2026; anything extracted before that stamp should be treated
as suspect.

---

## 3 · Extraction — `D_liu/verified/`

Both scripts import `params.py` and call `should_skip()`, so the exclusion set
in `params.py` governs who enters the analysis files.

| script | output | runtime |
|---|---|---|
| `01_univariate_analyses.py` | `D_liu/univariate_v1.csv` | ~20 min |
| `04_multivariate_analyses.py` | `D_liu/rsa_v1.csv` | ~60 min |
| `04b_multivariate_scramble.py` | `D_liu/rsa_v2_scramble.csv` | **[unverified]** |

`univariate_v1.csv` columns of interest: `peak_z` (peak selectivity),
`volume` (count of suprathreshold voxels — this is the selective-voxel-count
measure), `sum_selec_norm` (retired), `peak_x_native` / `peak_y_native`.

`rsa_v1.csv` columns of interest: `liu_distinctiveness` (preferred-category
similarity), `fisher_r` with a `pair` column (between-category similarity).

Note `univariate_v1.csv` holds more subjects than `rsa_v1.csv` (74 vs 65 as of
Sept 2026) because it includes sessions the RSA stage does not. Not an error.

---

## 4 · Harmonization — `F_harmonization/`

ComBat is **primary, not a sensitivity check**. All group analyses run on
harmonized features. Batch = scanner; preserved covariates = group, age, sex;
winsorized at the 5th/95th percentile within group × hemisphere; run
separately per hemisphere.

| script | reads | writes |
|---|---|---|
| `combat_05b_harmonize_univariate_sqrt.py` | `univariate_v1.csv` | `univariate_v1_harmonized_sqrt.csv` |
| `combat_06_harmonize_rsa.py` | `rsa_v1.csv` | `rsa_v1_harmonized.csv` |
| `combat_03_tfce_harmonized.py` | voxelwise maps | harmonized TFCE inputs **[unverified]** |
| `combat_06b_harmonize_rsa_scramble.py` | `rsa_v2_scramble.csv` | **[unverified]** |

`combat_05` (no `b`) is the earlier non-sqrt version. **Use `05b`.**

**The sqrt scale.** `SQRT_MEASURES` in `05b` lists which measures are
sqrt-transformed before ComBat and left on that scale; currently
`['sum_selec_norm', 'volume']`. They are not back-transformed, because
squaring folds negative sqrt-scale values to positive. Run tests on the sqrt
values; square medians only for descriptive reporting. Cohen's d is in sqrt
units.

**Residual negatives.** Even with sqrt, ComBat leaves ~40 `volume` cells
below zero (about 3% of cells). Any voxel-count analysis has to decide what
to do with those.

WTA is **not** harmonized: the four category proportions are compositional,
and harmonizing each category map independently distorts the voxel-wise
argmax.

---

## 5 · Voxelwise branches

| script | purpose | output |
|---|---|---|
| `D_liu/tfce_votc_contrasts.py` | builds category-vs-all-others inputs for randomise | **[unverified]** |
| `D_liu/verified/02_tfce_analyses_dontuse_useharmony.py` | superseded — name says so | — |
| `D_liu/verified/03_wta_analyses.py` | preference mapping / WTA | `wta_percentages.csv` **[unverified — file not found at repo root]** |
| `D_liu/wta_parcel_composition.py` | WTA within parcels | **[unverified]** |

`wta_percentages.csv` is cited by the handoff docs as living at the repo root
but was not there in Sept 2026. Locate before use.

---

## 6 · Statistics — `D_liu/verified/05_stats_harmony.py`

The live stats script. `05_stats.py` and the `.BAK_*` files are earlier
versions.

```bash
cd D_liu/verified
python 05_stats_harmony.py                     # uses harmonized defaults
python 05_stats_harmony.py --tag _harmonized   # writes to a suffixed filename
```

Outputs `D_liu/stats_results.csv` and `D_liu/tfce_clusters.csv`, or with a
suffix if `--tag` is given.

**Defaults matter.** `UNIVAR_CSV` and `RSA_CSV` at the top of the file set
what a bare run reads. These were changed in Sept 2026 to point at the
harmonized files; before that, a bare run silently used unharmonized input.

**Exclusions** are set by `EXCLUDE` near the top:

```python
EXCLUDE = ['sub-017',                          # polymicrogyria
           'sub-091', 'sub-095', 'sub-096',    # age cap <= 23
           'sub-027', 'sub-084']               # control exclusions
EXCLUDE_SES = [('sub-108', 2)]
```

The same list must be mirrored in `06_fig.py` and `05_stats.py`, which keep
their own copies. **Centralising this is outstanding work.**

Resulting cohort: 12 LH-intact, 12 RH-intact, 36 controls.

`stats_results_harmonized_corrected.csv` is **not produced by this script**.
It is a hand-built file (Aug 2026, after the sub-076 fix) that the grid
scripts and manuscript drafts cite. Nothing regenerates it.

---

## 7 · Grid — repo root

| script | rows | purpose |
|---|---|---|
| `marlene_grid.py` | 90 | permutation OLS, category × group by specification |
| `marlene_lmm.py` | 18 | LMM omnibus |
| `marlene_roi.py` | 216 | per-ROI and per-pair |
| `merge_marlene.py` | — | combines the three into the grid table |

`--csv` is required to write output; a bare run prints only.

These apply the age cap independently of `05_stats_harmony.py`. **Confirm the
two agree** — they disagreed until Sept 2026, which is why the grid reported
12/12 and 36 controls while the manuscript drafts carried 13/12 and 38.

---

## 8 · Whole-OTC RSM — `otc_rsm_rosenke.py`

Rosenke-method analysis on the whole OTC parcel, no spheres. **Has no
exclusion logic of its own** — subjects come straight from
`v.load_subjects()`. Any cohort restriction must be applied by editing the
script.

Outstanding: the odd/even run split, which is the one part of Rosenke's method
not implemented and the only route to the within-category reliability
diagonal.

---

## Known traps

- **`sessions[0]` vs `post_sessions[0]`** — scripts using `sessions[0]` silently
  take sub-021's pre-surgery session. Check any script that anchors on a
  session.
- **sub-076 is RH-intact** (`surgery_side=left`, `intact_hemi=right`). The live
  `sub_info.csv` is correct; older snapshots such as `sub_info4_30.csv` carry
  the old error.
- **sub-108 has no pre-surgical session** — both sessions are post. A stale
  `pre_surgery_sessions` entry of `{2}` will wrongly drop one.
- **Cope 13 for word ROIs** is the pipeline definition and stays. Substituting
  cope 9 was tested and rejected.
- **Two params files** — see section 0.
- **Three copies of `EXCLUDE`** — see section 6.
