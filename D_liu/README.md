# Patient OTC — Cross-Sectional & RSA Analysis Notebooks

Analysis notebooks for the patient-OTC manuscript (extends Liu et al., 2025, *Communications Biology*): category-selective OTC reorganization following pediatric unilateral OTC resection. This folder holds the **cross-sectional** analyses, **RSA/geometry** analyses, and the **WTA composition** statistics, plus the figure-generation code.

---

## Shared conventions

- **Sample:** 22 OTC patients (11 LH-intact / R-resected, 11 RH-intact / L-resected) + 38 controls, cross-sectional. nonOTC excluded; **sub-017 excluded** (polymicrogyria); `sub-108 ses-02` excluded where noted.
- **Session selection:** OTC patients → last (most recent) session; controls → first session. Applied consistently across notebooks.
- **Hemisphere handling:** split by intact hemisphere, never pooled. `hemi='l'` = LH-intact patients vs controls' left; `hemi='r'` = RH-intact vs controls' right.
- **Category order (figures):** word, face, house, object.
- **Colors:** controls gray (`#888888`/`#BBBBBB`); L-resected `#6B8AAD` (blue); R-resected `#E07A8B` (pink). Significance marked with red asterisks; no trend markers.
- **Stats defaults:** permutation 10,000 iterations; LMM omnibus (joint Wald χ²) as primary inferential test; Cohen's d + MSE reported; BH-FDR preferred over Bonferroni; per-category p-values uncorrected when categories are pre-specified.

## Key paths

| | |
|---|---|
| Processed data (`processed_dir`) | `/user_data/csimmon2/sym_pt/` |
| Git repo | `/user_data/csimmon2/git_repos/sym_pt/` |
| Subject CSV (`LIU_CSV`) | `…/sym_pt/D_liu/…` (+ addon) |
| Peak coords (`MNI_CSV`) | `…/sym_pt/group_results/peak_coords_mni.csv` |
| TFCE outputs | `…/sym_pt/group_results/tfce_votc_fdr/` |
| WTA export | `…/sym_pt/group_results/wta_export/` |
| **Figure outputs (`FIG_DIR`)** | `…/git_repos/sym_pt/C_results/figures/` |

---

## `cross-sectional.ipynb`
**Consolidated cross-sectional notebook — single source for the three locked measures.**

Recreates the Liu (2025) cross-sectional analyses across the three locked measures: **peak distance, sum-selectivity, distinctiveness**. Patient vs. control, per ROI × hemisphere.

**Sections:** Setup → Load → Helpers (`crawford`) → Fig 2C–H per-subject ROI peak scatter (native) → Fig 2I/J spatial-topography Crawford-t (ventral key) → peak coordinates MNI (controls vs patients) → per-ROI distance permutation test → Fig 3A/B example RDMs → Fig 3D per-ROI distinctiveness boxplot → Fig 3E per-ROI Crawford-t (FDR) → sum-selectivity violins → per-ROI sum-selectivity permutation test.

**Locked findings (per-ROI distance, 10k perm):**
- LH-intact `word_pSTG_liu`: p = .0067, Δ = −3.95 mm (patients *closer* to control centroid)
- RH-intact `house_TOS`: p = .0007, Δ = +6.73 mm
- `object_LOC`: trending in both groups

**Outputs:** `liu_exact_fig2CH.png`, `liu_exact_fig2IJ.png`, `peak_scatter_mni.png`, `peak_scatter_mni_primary.png`, `liu_exact_fig3D.png`, `liu_exact_fig3E.png`, `liu_exact_sum_selectivity*.png`; `pt_group_delta_sumsel_distance` CSV.

> Note: sum-selectivity violin has three variants in §11 (`[24]`, `[25]`, `[26]`); canonical one TBD before consolidation.

---

## `tfce_wta_figure.ipynb`
**Group-level TFCE + WTA brain maps (cross-sectional).**

Group-level patient-vs-control comparison of full-VOTC winner-take-all (WTA) maps using Liu contrasts and MNI z-stats — tests how voxel allegiance differs between groups *beyond* peak displacement and sum-selectivity. Controls (n≈38, both hemispheres); OTC patients (intact hemisphere only); one session per subject (first available post-surgery).

**Sections:** TFCE cluster renders → WTA category maps → within-cluster WTA composition → Crawford section.

**TFCE surviving clusters:** `object_L` (437 vox, peak MNI [−54,−70,−2]); `house_R` (480 vox, [30,−50,−10]); `word_R` (663 vox, [48,−64,−8]); **face did not survive** correction in either hemisphere.

**Within-cluster WTA composition:** `object_L` ctrl 60.7% vs pt 34.4%; `house_R` ctrl 36.4% vs pt 19.6%; `word_R` ctrl 11.4% vs pt 30.6%.

> Dependency: brain renders use `wb_command`/nilearn surface calls — require local `wb_view` + HCP surfaces; most likely to need path fixes when run elsewhere.

---

## `category_rsa_geometry.ipynb`
**Category RSA — patient vs. control, three measures coarse → fine.**

Same RDM spheres throughout. `fisher_r` stored as Fisher-z and modeled directly; `tanh`→r only for figures. OTC last session, controls first session; exclude sub-017 and (sub-108, ses-02). Split by intact hemisphere, never pooled.

**Sections:**
1. **Distinctiveness (collapsed)** — preferred category vs. the other three (mean correlation), one value per subject. 10k perm; `diff = pt − ctrl`. **Lower Fisher-z = more distinct** (preferred separates from the rest).
2. **Geometry — holistic (omnibus)** — does the full 6-value RDM differ by group? MixedLM, subject random intercept; 5-df joint Wald on `C(pair_sorted):C(group)`.
3. **Geometry — pairwise (post-hoc)** — which of the 6 pairs moved? 10k label-shuffle perm, Cohen's d, BH-FDR within ROI × hemi. Survivors under a *null* omnibus are exploratory only.
4. **Survivors** — pairwise survivors per stratum (cross-checked against omnibus).
5. **RDM panels + Procrustes (viz, not inferential)** — per ROI: control / LH-intact / RH-intact mean RDM (r-space); Procrustes disparity = misfit of each subject's 4-category config to control-mean (controls leave-one-out).

**Outputs:** `distinctiveness_bars.png`, `distinctiveness_violin.png`, `rdm_diff_{roi}_{hemi}.png`, `procrustes_{roi}_{hemi}.png`.

---

## `wta_formal_stats.ipynb`
**WTA composition — formal statistics. Four comparisons across full-hemisphere VOTC WTA composition.**

| # | Comparison | Model | Per-category test |
|---|---|---|---|
| 1 | LH ctrl vs LH pt | `pct ~ category * group + (1\|sid)` | Permutation |
| 2 | RH ctrl vs RH pt | `pct ~ category * group + (1\|sid)` | Permutation |
| 3 | LH pt vs RH pt | `pct ~ category * intact_hemi + (1\|sid)` | Permutation |
| — | LH ctrl vs RH ctrl | `pct ~ category * hemi + (1\|sid)` | Permutation (paired) |

**Sections:** Setup → WTA computation → long-format builder → helpers (LMM, permutation, Cohen's d) → analysis functions (omnibus χ²/p, MSE, per-category mean/diff/d/p) → reporter → **PRIMARY** (non-independent controls, both hemispheres; LMM handles via `(1|sid)`) → **SUPPLEMENTARY** (resampled random ctrl-hemisphere allocation per PI request; preserves independence; median across iterations) → summary tables.

**Key results:**
- **Model 2** (RH pt vs ctrl): strongest finding — all four categories significant (face↑, house↓, object↓, word↑; three Bonferroni-significant).
- **Model 1** (LH pt vs ctrl): omnibus p = .012; object↓ (p = .030), word↑ (p = .044) uncorrected; underpowered at n = 11.
- **Model 3** (LH-intact vs RH-intact pt): null (p = .289) — supports symmetric reorganization across resection sides.

**Outputs:** `wta_territory_violin.png` + summary tables.

---

## Dependencies
Python (statsmodels LMMs/REML, scipy, nilearn, pingouin, matplotlib/seaborn), FSL 6.0.3, `wb_command` (surface figures), conda `fmri` environment. Cross-sectional CSV + MNI peak-coords CSV are the primary tabular inputs; brain figures additionally need TFCE outputs and HCP surfaces.

## Status / planned consolidation
A single consolidated **figures notebook** is planned, pulling figure cells from `cross-sectional`, `tfce_wta_figure`, and `category_rsa_geometry` (stats cells stay in their home notebooks). The stats notebooks above are kept as-is.
