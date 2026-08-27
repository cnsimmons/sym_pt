#!/usr/bin/env python3
"""
combat_07_searchlight_distinctiveness.py — whole-OTC searchlight RSA + TFCE.

Answers the limitation the manuscript concedes: the pattern analyses are confined
to 7 mm spheres at each ROI's peak. This computes distinctiveness at EVERY OTC
voxel and asks where it differs between groups.

FOUR STAGES, one script. Each stage skips work that already exists, so a failed
run resumes rather than restarting.

  1  searchlight   native betas -> 4 distinctiveness maps per subject
  2  register      -> MNI, via the subject's own anat2stand.mat
  3  harmonize     ComBat, 4 categories stacked, per hemisphere
  4  randomise     TFCE, patients vs matched-hemisphere controls

WHY DISTINCTIVENESS NEEDS FOUR MAPS
  Distinctiveness is defined relative to a preferred category, and a searchlight
  voxel has none. So one map per category:

    word map  = mean Fisher-z( word-face, word-house, word-object ) in the sphere
    face map  = mean Fisher-z( face-house, face-object, face-word )
    ... and so on.

  The alternative — one map using each voxel's argmax category — is not usable.
  The argmax comes from the same betas as the RSA, and patients' argmax differs
  sharply from controls' (left word occupies 9.7% of selective voxels in controls
  vs 37.0% in patients), so at a given voxel it would compare a patient's word
  distinctiveness against a control's object distinctiveness. Different
  quantities in the same map.

  Four maps also matches the existing TFCE family: 4 categories x 2 hemispheres
  = 8 comparisons, FWE over voxels then BH across the eight.

CONSISTENCY WITH THE EXISTING PIPELINE — no methodological divergence
  - subjects, sessions, VOTC masks, design, threshold, 10k perms, seed 42: all
    from verified/02_tfce_analyses via its own functions, as combat_01/02/03 do
  - 7 mm sphere at 2 mm resolution, ~175 voxels: matches 04_multivariate_analyses
  - raw single-condition betas (copes 15-18), as the ROI RSA uses
  - ComBat: batch=scanner, preserve group+age+sex, per hemisphere, 4 categories
    stacked as features: identical to combat_02
  - native OTC mask via the subject's own mni2anat.mat, which register_mirror.py
    already writes and already uses to warp MNI ROIs into native space

  ONE deliberate difference: the native OTC mask is warped with
  nearest-neighbour, not trilinear+binarize as warp_rois does. Trilinear
  binarizing is generous at mask edges, which would run searchlights centred on
  voxels the MNI mask does not contain. It does not affect the final result,
  since the MNI OTC mask is applied again at randomise.

WHY THE MEASURE IS COMPUTED BEFORE HARMONIZATION
  Distinctiveness is a correlation ACROSS voxels within a sphere. Harmonizing or
  interpolating betas first would alter pattern similarity and so alter the
  measure itself. Computing natively and harmonizing the finished scalar is what
  combat_06 already does for the ROI version (it harmonizes
  liu_distinctiveness and fisher_r, not the betas).

COMPUTE
  ~10k searchlights per hemisphere per subject. Submit stage 1 via SLURM.

Usage
  python combat_07_searchlight_distinctiveness.py --stage all
  python combat_07_searchlight_distinctiveness.py --stage 1        # searchlight only
  python combat_07_searchlight_distinctiveness.py --stage 1 --sub sub-004
  python combat_07_searchlight_distinctiveness.py --stage 3,4
  python combat_07_searchlight_distinctiveness.py --stage 4 --merge-thresh 0.0
  python combat_07_searchlight_distinctiveness.py --stage all --dry-run
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

GIT     = Path('/user_data/csimmon2/git_repos/sym_pt')
# The verified TFCE module has been renamed at least twice. Resolve it rather
# than hard-coding, so a future rename fails loudly here instead of silently.
_VTFCE_CANDIDATES = [
    GIT / 'D_liu' / 'verified' / '02_tfce_analyses_dontuse_useharmony.py',
    GIT / 'D_liu' / 'verified' / '02_tfce_analyses_not_as_verified.py',
    GIT / 'D_liu' / 'verified' / '02_tfce_analyses.py',
]
VTFCE = next((p for p in _VTFCE_CANDIDATES if p.exists()), None)
if VTFCE is None:
    _found = sorted((GIT / 'D_liu' / 'verified').glob('02_tfce*.py'))
    sys.exit('Cannot find the verified TFCE module.\n'
             f'  Looked for: {[p.name for p in _VTFCE_CANDIDATES]}\n'
             f'  Present:    {[p.name for p in _found] or "none"}\n'
             '  Add the correct name to _VTFCE_CANDIDATES.')
SCANNER = GIT / 'F_harmonization' / 'sub_info_scanner.csv'
MNI     = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'

sys.path.insert(0, str(GIT))
spec = importlib.util.spec_from_file_location('verified_tfce', str(VTFCE))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

from params import processed_dir

PROC     = Path(processed_dir)
OUT_DIR  = PROC / 'group_results' / 'tfce_searchlight_distinct'
NAT_DIR  = OUT_DIR / 'native_maps'
MNI_DIR  = OUT_DIR / 'mni_maps'
HARM_DIR = OUT_DIR / 'harmonized_maps'

CATEGORIES = v.CATEGORIES                 # face, house, object, word
HEMIS      = v.HEMIS                      # l, r
RSA_COPES  = {'face': 15, 'house': 16, 'object': 17, 'word': 18}

SPHERE_MM      = 7.0                      # matches 04_multivariate_analyses
FUNC_VOXEL_MM  = 2.0
ANAT_VOXEL_MM  = 1.0
DOWNSAMPLE_FAC = ANAT_VOXEL_MM / FUNC_VOXEL_MM
MIN_SPHERE_VOX = 20                       # below this the correlation is unstable
N_PERM         = 10000


# ── stage 1: searchlight ─────────────────────────────────────────────────────

def sphere_offsets(radius_mm, voxel_mm):
    r = int(np.floor(radius_mm / voxel_mm))
    rng = np.arange(-r, r + 1)
    gx, gy, gz = np.meshgrid(rng, rng, rng, indexing='ij')
    d = np.sqrt((gx * voxel_mm) ** 2 + (gy * voxel_mm) ** 2 + (gz * voxel_mm) ** 2)
    keep = d <= radius_mm
    return np.stack([gx[keep], gy[keep], gz[keep]], axis=1)


def native_otc_mask(sid, first_ses, hemi, mni_mask_path, ref_2mm, dry=False):
    """Warp the MNI VOTC mask into this subject's native space.

    Uses the subject's own mni2anat.mat, which register_mirror.py already writes
    and already uses for exactly this purpose. Nearest-neighbour so the mask
    stays a mask.
    """
    anat = PROC / sid / f'ses-{first_ses}' / 'anat'
    xfm  = anat / 'mni2anat.mat'
    out  = NAT_DIR / f'{sid}_votc_{hemi}_native.nii.gz'
    if out.exists():
        return out
    if not xfm.exists():
        print(f'    ERROR {sid}: mni2anat.mat missing (run register_mirror.py)')
        return None
    cmd = ['flirt', '-in', str(mni_mask_path), '-ref', str(ref_2mm),
           '-out', str(out), '-applyxfm', '-init', str(xfm),
           '-interp', 'nearestneighbour']
    if dry:
        print(f'    DRY {" ".join(cmd)}')
        return None
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def beta_path(sid, session, first_ses, cope):
    feat = (PROC / sid / f'ses-{session}' / 'derivatives' / 'fsl' / 'loc'
            / 'HighLevel.gfeat' / f'cope{cope}.feat' / 'stats')
    nm = 'cope1.nii.gz' if session == first_ses else f'cope1_ses{first_ses}.nii.gz'
    return feat / nm


def load_betas_2mm(sid, session, first_ses):
    """Four raw category betas, downsampled 1mm anat -> 2mm, as the ROI RSA does."""
    vols, ref = [], None
    for cat in CATEGORIES:
        p = beta_path(sid, session, first_ses, RSA_COPES[cat])
        if not p.exists():
            return None, None
        img = nib.load(str(p))
        vols.append(zoom(img.get_fdata(), DOWNSAMPLE_FAC, order=1).astype(np.float32))
        if ref is None:
            ref = img
    return np.stack(vols, axis=0), ref          # [4, X, Y, Z]


def searchlight_maps(betas, mask, offsets):
    """Four distinctiveness maps. betas [4,X,Y,Z], mask [X,Y,Z] bool.

    At each mask voxel: gather the sphere, correlate the four category patterns,
    Fisher-transform, then for each category average its three pairs.
    """
    shape = mask.shape
    out = {c: np.zeros(shape, np.float32) for c in CATEGORIES}
    centers = np.argwhere(mask)
    flat = betas.reshape(4, -1)
    strides = np.array([shape[1] * shape[2], shape[2], 1])

    for cx, cy, cz in centers:
        idx = np.array([cx, cy, cz]) + offsets
        ok = ((idx >= 0).all(1) &
              (idx[:, 0] < shape[0]) & (idx[:, 1] < shape[1]) & (idx[:, 2] < shape[2]))
        idx = idx[ok]
        lin = idx @ strides
        P = flat[:, lin]                                    # [4, n_vox]
        good = np.isfinite(P).all(0) & (P != 0).any(0)
        P = P[:, good]
        if P.shape[1] < MIN_SPHERE_VOX or (P.std(axis=1) == 0).any():
            continue
        f = np.arctanh(np.clip(np.corrcoef(P), -0.999, 0.999))
        for i, cat in enumerate(CATEGORIES):
            others = [j for j in range(4) if j != i]
            out[cat][cx, cy, cz] = np.mean(f[i, others])
    return out


def stage1(subjects, targets, dry=False):
    NAT_DIR.mkdir(parents=True, exist_ok=True)
    v.OUT_DIR = OUT_DIR
    masks = v.build_votc_masks_and_save()
    offsets = sphere_offsets(SPHERE_MM, FUNC_VOXEL_MM)
    print(f'\n=== STAGE 1 searchlight  ({len(offsets)} voxels per sphere, '
          f'{SPHERE_MM:g}mm at {FUNC_VOXEL_MM:g}mm) ===')

    t0 = time.time()
    for n, sid in enumerate(targets, 1):
        info = subjects[sid]
        hemis = HEMIS if info['hemi'] is None else [info['hemi']]
        want = [NAT_DIR / f'{sid}_{c}_{h}_distinct.nii.gz'
                for c in CATEGORIES for h in hemis]
        if all(p.exists() for p in want):
            print(f'[{n}/{len(targets)}] {sid} already done')
            continue

        betas, ref = load_betas_2mm(sid, info['session'], info['first_session'])
        if betas is None:
            print(f'[{n}/{len(targets)}] {sid} SKIP — betas missing')
            continue
        ref2 = nib.Nifti1Image(betas[0], ref.affine @ np.diag([2, 2, 2, 1]))
        ref2_path = NAT_DIR / f'{sid}_ref2mm.nii.gz'
        if not ref2_path.exists() and not dry:
            nib.save(ref2, str(ref2_path))

        for hemi in hemis:
            nm = native_otc_mask(sid, info['first_session'], hemi,
                                 masks[hemi], ref2_path, dry=dry)
            if nm is None:
                continue
            mask = nib.load(str(nm)).get_fdata() > 0.5
            if mask.shape != betas.shape[1:]:
                print(f'    ERROR {sid} {hemi}: mask {mask.shape} vs '
                      f'betas {betas.shape[1:]} — grids differ')
                continue
            print(f'[{n}/{len(targets)}] {sid} {hemi}: {int(mask.sum()):,} '
                  f'searchlights ({time.time() - t0:.0f}s)')
            if dry:
                continue
            maps = searchlight_maps(betas, mask, offsets)
            for cat, arr in maps.items():
                nib.save(nib.Nifti1Image(arr, ref2.affine),
                         str(NAT_DIR / f'{sid}_{cat}_{hemi}_distinct.nii.gz'))
    print(f'Stage 1 done in {(time.time() - t0) / 60:.1f} min')


# ── stage 2: register to MNI ──────────────────────────────────────────────────

def stage2(subjects, targets, dry=False):
    MNI_DIR.mkdir(parents=True, exist_ok=True)
    print('\n=== STAGE 2 register to MNI (anat2stand.mat, trilinear) ===')
    n_done = n_skip = 0
    for sid in targets:
        info = subjects[sid]
        xfm = PROC / sid / f'ses-{info["first_session"]}' / 'anat' / 'anat2stand.mat'
        if not xfm.exists():
            print(f'  ERROR {sid}: anat2stand.mat missing')
            continue
        hemis = HEMIS if info['hemi'] is None else [info['hemi']]
        for cat in CATEGORIES:
            for hemi in hemis:
                src = NAT_DIR / f'{sid}_{cat}_{hemi}_distinct.nii.gz'
                dst = MNI_DIR / f'{sid}_{cat}_{hemi}_distinct_mni.nii.gz'
                if not src.exists():
                    continue
                if dst.exists():
                    n_skip += 1
                    continue
                cmd = ['flirt', '-in', str(src), '-ref', MNI, '-out', str(dst),
                       '-applyxfm', '-init', str(xfm), '-interp', 'trilinear']
                if dry:
                    print(f'  DRY {dst.name}')
                    continue
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                n_done += 1
    print(f'Stage 2: {n_done} registered, {n_skip} already existed')


# ── stage 3: ComBat ──────────────────────────────────────────────────────────

def stage3(subjects, dry=False):
    """ComBat on the MNI distinctiveness maps. Structure identical to combat_02:
    4 categories stacked as features, one fit per hemisphere, batch=scanner,
    preserve group+age+sex."""
    from neuroHarmonize import harmonizationLearn
    HARM_DIR.mkdir(parents=True, exist_ok=True)
    v.OUT_DIR = OUT_DIR
    masks = v.build_votc_masks_and_save()
    scan = pd.read_csv(SCANNER)
    print('\n=== STAGE 3 ComBat (batch=scanner, preserve group+age+sex) ===')

    for hemi in HEMIS:
        mimg = nib.load(str(masks[hemi]))
        mask = mimg.get_fdata().astype(bool)
        nvox = int(mask.sum())

        subs, cov_rows = [], []
        for sid, info in subjects.items():
            if info['hemi'] is not None and info['hemi'] != hemi:
                continue
            paths = [MNI_DIR / f'{sid}_{c}_{hemi}_distinct_mni.nii.gz'
                     for c in CATEGORIES]
            if not all(p.exists() for p in paths):
                continue
            r = scan[(scan['sub'] == sid) & (scan['ses'] == f"ses-{info['session']}")]
            if not len(r) or pd.isna(r.iloc[0]['scanner']):
                print(f'  WARNING {sid}: no scanner row — dropped')
                continue
            r = r.iloc[0]
            subs.append(sid)
            cov_rows.append({'subject_id': sid, 'group': info['group'],
                             'scanner': str(r['scanner']),
                             'age': r['age'], 'sex': r['sex']})
        if len(subs) < 5:
            print(f'  [{hemi.upper()}H] too few subjects ({len(subs)}) — skipped')
            continue
        cov = pd.DataFrame(cov_rows)

        X = []
        for cat in CATEGORIES:
            X.append(np.vstack([
                nib.load(str(MNI_DIR / f'{s}_{cat}_{hemi}_distinct_mni.nii.gz'))
                   .get_fdata()[mask].astype(np.float32) for s in subs]))
        data = np.hstack(X)

        print(f'  [{hemi.upper()}H] n={len(subs)}  features={data.shape[1]:,}  '
              f'site={cov.scanner.value_counts().to_dict()}  '
              f'group={cov.group.value_counts().to_dict()}')
        if dry:
            continue

        design = pd.DataFrame({'SITE': cov['scanner'].values, 'age': cov['age'].values})
        design = pd.concat([design,
                            pd.get_dummies(cov['sex'], prefix='sex',
                                           drop_first=True).astype(int),
                            pd.get_dummies(cov['group'], prefix='group',
                                           drop_first=True).astype(int)], axis=1)
        _, adj = harmonizationLearn(data, design, smooth_terms=[])

        site = cov['scanner'].values
        ctrl = (cov['group'] == 'control').values
        def gap(M, a, b):
            return float(np.abs(np.nanmean(M[a], 0) - np.nanmean(M[b], 0)).mean())
        print(f'    site  gap: {gap(data, site == "Verio", site == "Prisma"):.4f} -> '
              f'{gap(adj, site == "Verio", site == "Prisma"):.4f}   (should shrink)')
        print(f'    group gap: {gap(data, ctrl, ~ctrl):.4f} -> '
              f'{gap(adj, ctrl, ~ctrl):.4f}   (should hold)')

        for i, cat in enumerate(CATEGORIES):
            block = adj[:, i * nvox:(i + 1) * nvox]
            for j, sid in enumerate(subs):
                vol = np.zeros(mask.shape, np.float32)
                vol[mask] = block[j]
                nib.save(nib.Nifti1Image(vol, mimg.affine),
                         str(HARM_DIR / f'{sid}_{cat}_{hemi}_harm.nii.gz'))
        np.savez_compressed(HARM_DIR / f'subs_{hemi}.npz',
                            subs=np.array(subs), group=cov['group'].values)
        print(f'    -> {len(subs) * len(CATEGORIES)} harmonized maps')


# ── stage 4: randomise ───────────────────────────────────────────────────────

def stage4(dry=False, merge_thresh=None):
    """TFCE via the verified module's own merge / design / randomise.

    THRESHOLD IS NONE HERE, AND THAT IS NOT A DIVERGENCE FROM THE PIPELINE.

    The verified TFCE merges cat-vs-all-others ZSTAT maps and thresholds them at
    0.0, which zeroes negative z. There, negative z means "not selective for this
    category", so dropping it is a defensible selectivity floor.

    The maps merged here are not zstats. They are the DISTINCTIVENESS VALUES
    THEMSELVES — a signed Fisher-z correlation in which negative means MORE
    distinct. Thresholding at 0 would delete the informative half of the outcome
    variable, not a floor on the input. Control house_PPA distinctiveness spans
    -0.42 to +0.69, so a substantial minority of genuine values are negative.

    merge_thresh is left configurable, and if a value is passed the count of
    zeroed in-mask values is printed so the cost is visible.
    """
    v.OUT_DIR = OUT_DIR
    masks = v.build_votc_masks_and_save()
    print('\n=== STAGE 4 randomise, TFCE, 10k perms, seed 42 ===')
    print(f'    merge threshold = '
          f'{"none (full signed scale)" if merge_thresh is None else merge_thresh}')

    for cat in CATEGORIES:
        for hemi in HEMIS:
            f = HARM_DIR / f'subs_{hemi}.npz'
            if not f.exists():
                print(f'  [{cat}_{hemi}] no subs_{hemi}.npz — run stage 3')
                continue
            z = np.load(f, allow_pickle=True)
            subs, groups = list(z['subs']), list(z['group'])
            ctrl = [s for s, g in zip(subs, groups) if g == 'control']
            pt   = [s for s, g in zip(subs, groups) if g == 'OTC']
            if len(ctrl) < 5 or len(pt) < 3:
                print(f'  [{cat}_{hemi}] SKIP n_ctrl={len(ctrl)} n_pt={len(pt)}')
                continue

            test_dir = OUT_DIR / f'{cat}_{hemi}_pt_vs_ctrl'
            test_dir.mkdir(parents=True, exist_ok=True)
            paths = [HARM_DIR / f'{s}_{cat}_{hemi}_harm.nii.gz' for s in ctrl + pt]
            missing = [p.name for p in paths if not p.exists()]
            if missing:
                print(f'  [{cat}_{hemi}] SKIP — {len(missing)} maps missing')
                continue

            print(f'  [{cat}_{hemi}] n_ctrl={len(ctrl)} n_pt={len(pt)}')

            if merge_thresh is not None and not dry:
                mask = nib.load(str(masks[hemi])).get_fdata().astype(bool)
                neg = tot = 0
                for p in paths:
                    d = nib.load(str(p)).get_fdata()[mask]
                    d = d[d != 0]
                    neg += int((d < merge_thresh).sum())
                    tot += int(d.size)
                if tot:
                    print(f'      threshold {merge_thresh} zeroes '
                          f'{neg:,}/{tot:,} in-mask values ({100 * neg / tot:.1f}%)')

            if dry:
                continue
            merged = test_dir / 'merged_distinct.nii.gz'
            v.merge_zstats(paths, merged, threshold=merge_thresh)
            mat, con = v.write_design_files(str(test_dir / 'design'),
                                            len(ctrl), len(pt))
            v.run_randomise(merged, test_dir / 'rand', masks[hemi], mat, con,
                            N_PERM, 'tfce', v.DEFAULT_CLUSTER_THRESH)
    print(f'\nDone -> {OUT_DIR}')
    print('Cluster tables: rand_tfce_corrp_tstat1 (ctrl>pt), _tstat2 (pt>ctrl).')
    print('Then BH-FDR across the 8 category x hemisphere comparisons, as in the MS.')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    help="'all' or comma list from 1,2,3,4")
    ap.add_argument('--sub', default=None,
                    help='comma-separated subject ids, stages 1-2 only')
    ap.add_argument('--merge-thresh', default='none',
                    help="subject-map threshold at merge. Default 'none': these "
                         "maps are signed distinctiveness values, not zstats, so "
                         "negatives are the measure. Pass a number only "
                         "deliberately.")
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    mt = None if str(args.merge_thresh).lower() == 'none' else float(args.merge_thresh)

    stages = ([1, 2, 3, 4] if args.stage == 'all'
              else [int(x) for x in args.stage.split(',')])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = v.load_subjects()
    print(f'{len(subjects)} subjects from verified load_subjects '
          f'({sum(1 for i in subjects.values() if i["group"] == "control")} control, '
          f'{sum(1 for i in subjects.values() if i["group"] == "OTC")} OTC)')

    targets = list(subjects)
    if args.sub:
        want = {s.strip() if s.startswith('sub-') else f'sub-{s.strip()}'
                for s in args.sub.split(',')}
        targets = [s for s in targets if s in want]
        print(f'  restricted to {len(targets)}: {targets}')

    if 1 in stages:
        stage1(subjects, targets, dry=args.dry_run)
    if 2 in stages:
        stage2(subjects, targets, dry=args.dry_run)
    if 3 in stages:
        stage3(subjects, dry=args.dry_run)
    if 4 in stages:
        stage4(dry=args.dry_run, merge_thresh=mt)


if __name__ == '__main__':
    main()