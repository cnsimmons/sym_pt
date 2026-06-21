# sym_pt — Symmetric Patient fMRI Pipeline

Analysis pipeline for category-selective visual cortex reorganization following
pediatric unilateral occipitotemporal cortex (OTC) resection. Extends
Liu et al. (2025, *Communications Biology*). Localizer task: faces, houses,
objects, words, scrambled.

## Two analysis tracks (read this first)

One pipeline, run on two samples. They use **different config files** and
**different extraction outputs** — keep them straight:

| Track | Config | Extraction CSV | Sample |
|---|---|---|---|
| **Manuscript (frozen)** — `D_liu/verified/` | `params.py` | `D_liu/liu_exact_replication_v2.csv` (8,358 rows) | MS sample; skips sub-017, excludes sub-108 ses-02 |
| **Extended (current)** — top-level `D_liu/` | `sym_pt_params.py` | `liu_exact_replication_v2.csv` (repo root, 9,783 rows) | current roster (22 OTC + 38 controls) |

> ⚠️ **Filename trap:** the *newer* config is named `params.py` (its header still
> reads "sym_pt_params.py", dated 06/02/26); the *older* one is `sym_pt_params.py`.
> Both must stay — each track imports its own (`from params` vs `from sym_pt_params`).

## Sample (current / extended roster)

- **Patients:** 22 OTC — 11 left-intact / right-resection, 11 right-intact / left-resection; 6 longitudinal.
- **Controls:** 38 typically developing.
- **Excluded:** sub-017 (polymicrogyria) — never analyzed.
- _TODO: confirm exact N for the frozen manuscript (verified) sample._

Roster: `sub_info.csv` (long format, one row per subject-session).
Scanner labels: `F_harmonization/sub_info_scanner.csv`.

## Scanners

Two sites: Siemens **Verio** (32-ch) and **Prisma** (64-ch). Scanner is derived
per session from BIDS JSON sidecars by `F_harmonization/add_scanner.py`.
Scanner is confounded with group, so extent/detection measures are harmonized
with ComBat (in progress — `F_harmonization/`). Relative measures (LI, WTA, RSA)
are largely scanner-robust.

## Repository structure

- `A_preprocessing/` — raw BIDS → 1st-level FEAT → registration → MNI z-stat maps (steps 00–13). FEAT templates live here: `template_1stLevel.fsf`, `template_HighLevel.fsf`.
- `B_analyses/` — searchmask/ROI creation, summary values, peak coords, distinctiveness, geometry, post-hoc stats.
- `C_results/` — result notebooks and figures.
- `D_liu/` — Liu recreation (extended track) and `verified/` (frozen manuscript track).
- `E_longitudinal/` — longitudinal trajectories.
- `F_harmonization/` — ComBat scanner harmonization (in progress).
- `z_archive/` — retired / superseded; not used for the paper.

## Configs (repo root)

- `params.py` — manuscript track (`verified/`).
- `sym_pt_params.py` — extended track (top-level `D_liu/`, `B_analyses/`, `A_preprocessing/`).
- Both read `sub_info.csv`.

## FEAT contrasts (1st level)

| COPE | Contrast |
|---|---|
| 1 | Face > Object |
| 2 | House > Object |
| 3 | Object > Scramble |
| 4 | Word > Object |

(copes 1–19 exist; 1–4 are the category-localizer contrasts.)

## Key paths

- Processed data: `/user_data/csimmon2/sym_pt` (`processed_dir`)
- Raw BIDS (read-only): `/lab_data/behrmannlab/hemi/Raw`
- Repo: `/user_data/csimmon2/git_repos/sym_pt` (`git_dir`)