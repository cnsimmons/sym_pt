# Statistical Results — Provenance & Definitions

Last updated: 2026-06-23. This file documents every reported statistical output, the
cohort and preprocessing behind it, and the key analysis decisions, so results remain
interpretable and defensible (e.g. to reviewers) after the fact.

---

## Cohort (current)

22 OTC patients: **10 left-intact (right-resected) / 12 right-intact (left-resected)** + 38 controls.

- **sub-076 correction (2026-06-22):** previously coded left-intact; corrected to
  right-intact (left occipitotemporal resection) after operative-note and anatomical
  verification. Both `surgery_side` (right→left) and `intact_hemi` (left→right) were
  wrong in `sub_info.csv` and corrected. This shifted the cohort from 11/11 to 10/12.
  Verified the only mismatch in a full-cohort op-note↔CSV audit.
- Patients always split by intact hemisphere; never pooled across resection side.
- Exclusions: sub-017 (polymicrogyria); sub-098 / sub-101 / sub-109 (missing cope z-stats).

## Acquisition / harmonization

- Two scanners: Siemens Verio (7 patients) and Prisma (15 patients), plus controls.
  Scanner balance by intact hemisphere is comparable (LH 7 Prisma/4 Verio; RH 8/3),
  so site is not confounded with hemisphere.
- **ComBat (neuroCombat)** treated as a PREPROCESSING step, not a separate analysis:
  harmonized values are the reported results; unharmonized values are the
  without-correction comparison. batch = scanner; preserve group + age + sex;
  winsorize 5/95 within group; per-hemisphere fit. (Procedure follows Robert 2024,
  Granovetter 2026.) Site gap roughly halved, group gap preserved, on every measure.
- **Harmonized measures:** summed selectivity, distinctiveness, between-category
  geometry (continuous within-category measures), and the voxel-wise category z-maps
  entering TFCE.
- **WTA is NOT harmonized — by design.** It is thresholded and compositional
  (per-voxel argmax; category proportions sum to 100%). Independent per-category ComBat
  shifts categories relative to one another and distorts the winner assignment, with no
  closure-preserving precedent. WTA is reported raw and treated as descriptive (omnibus
  χ² is the inferential test; per-category FDR contrasts describe how the single
  compositional shift distributes — the four categories are not independent).

---

## Result files (all under D_liu/ unless noted)

| File | Cohort | Harmonized? | Notes |
|---|---|---|---|
| `stats_results.csv` | 10/12 (corrected) | raw | **AUTHORITATIVE current raw.** Sole source for raw reported values. |
| `stats_results_harmonized_corrected.csv` | 10/12 | harmonized (sel/dist/geom); WTA raw; TFCE harmonized | **AUTHORITATIVE current harmonized.** WTA rows are raw by design. |
| `stats_results.BAK_2026-06-22.csv` | 11/11 (pre-076) | raw | Pre-correction snapshot. Comparison only. |
| `stats_results_harmonized.csv` | 11/11 (pre-076) | harmonized | Pre-correction harmonized. Comparison only. |
| `tfce_clusters.csv` | 10/12 | raw z | Current TFCE clusters (raw). |
| `tfce_clusters_harmonized_corrected.csv` | 10/12 | harmonized z | Current TFCE clusters (harmonized) — identical to raw. |
| `univariate_v1.csv` / `_harmonized.csv` | 10/12 | raw / harmonized | Inputs to selectivity + peak stats. |
| `rsa_v1.csv` / `_harmonized.csv` | 10/12 | raw / harmonized | Inputs to distinctiveness + geometry. |
| `../sym_pt/group_results/wta_percentages.csv` | 10/12 | raw | WTA input (raw only). |
| `F_harmonization/_shelved/` | — | — | Shelved harmonized-WTA attempt (combat_07 + output). NOT USED — see WTA note. |
| `*.BAK_*` (univariate/rsa/wta) | 11/11 | — | Pre-correction input snapshots. |

## Pipeline that produces the stats (order)

1. `liu_recreation_csv_v2.py` → liu replication CSV (per-subject extraction; intact hemisphere)
2. `verified/01_univariate_analyses.py` → `univariate_v1.csv` (selectivity, peak)
3. `verified/04_multivariate_analyses.py` → `rsa_v1.csv` (distinctiveness, geometry)
4. `calc_peak_coords_mni.py` → `peak_coords_mni.csv`
5. `verified/02_tfce_analyses_not_as_verified.py` → `group_results/tfce_votc_fdr/`
6. `verified/03_wta_analyses.py` → `group_results/wta_percentages.csv`
7. `verified/05_stats.py` → `stats_results.csv` (+ `tfce_clusters.csv`)

Harmonized track: `F_harmonization/combat_01,02` (voxel matrices) → `combat_03` (harmonized TFCE)
→ `combat_05` (harmonized univariate) → `combat_06` (harmonized RSA) →
`verified/05_stats_harmony.py --univar ..._harmonized.csv --rsa ..._harmonized.csv --tag _harmonized_corrected`.
(`05_stats_harmony.py` gained `--rsa`/`--wta` flags; WTA left at raw default.)

## Key conventions

- FDR: Benjamini-Hochberg, two-tailed, within each measure's family.
- Effect sizes: Cohen's d (patient vs control); Δlog₁₀ additionally for sum-selectivity.
- Protected omnibus gating for WTA and geometry; reported-but-not-gating for peak,
  sum-selectivity, distinctiveness.
- Session rule: controls = first session; patients = last (post-surgical) session.
- `05_stats.py` uses one shared RandomState(42) threaded through all tests;
  standalone re-seeds give different borderline values — `05` output is authoritative.

## Status changes vs the pre-correction manuscript (for reviewer transparency)

- **WTA object-LH**: q .049 → .097 (no longer < .05). WTA is descriptive; object-LH
  rests inferentially on selectivity (q = .015) + TFCE, both of which hold.
- **rVWFA distinctiveness**: q .155 (raw, n.s.) → .034 (harmonized, survives).
- **TFCE extents**: word_R unchanged (663 vox); object_L 437→270; house_R 480→643
  — consequence of sub-076 moving LH→RH (alters left-object and right-house contrasts).