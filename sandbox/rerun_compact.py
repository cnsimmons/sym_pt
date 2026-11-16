#!/usr/bin/env python3
"""
rerun_compact.py - copy the rerun from lab_data to user_data in compact form.

Keeps, with the same folder layout (so the extraction scripts can be pointed at it):
  - every symlink, as a link (loc -> loc_rerun, the ROIs/anat links, the *_ses links)
  - HighLevel results: cope*.feat/stats/cope1, varcope1, zstat1 and the *_mni copies
  - first-level results: stats/cope*, varcope*, zstat* (no residuals)
  - every small file (< 1 MB): designs, contrasts, logs, reports, notes, matrices
Leaves behind the large intermediates: filtered_func_data, res4d, reg_standard,
the HighLevels' 4D inputs and 1-mm background images.

  python rerun_compact.py           plan: sizes of what is kept and left behind; writes nothing
  python rerun_compact.py copy      copies (stops if the destination exists), checks each
                                    copied file against the original, writes a manifest
  python rerun_compact.py verify    re-checks every copied file against the manifest
Nothing is ever deleted; clearing lab_data is a separate command, run by hand.
"""
import csv
import fnmatch
import hashlib
import os
import shutil
import sys
from pathlib import Path

SRC = Path('/lab_data/behrmannlab/claire/sym_pt_rerun')
DST = Path('/user_data/csimmon2/sym_pt_rerun')
SMALL = 1_000_000
KEEP_BIG = ['*/HighLevel.gfeat/cope*.feat/stats/cope1.nii.gz',
            '*/HighLevel.gfeat/cope*.feat/stats/varcope1.nii.gz',
            '*/HighLevel.gfeat/cope*.feat/stats/zstat1.nii.gz',
            '*/HighLevel.gfeat/cope*.feat/stats/*_mni.nii.gz',
            '*/1stLevel.feat/stats/cope*.nii.gz',
            '*/1stLevel.feat/stats/varcope*.nii.gz',
            '*/1stLevel.feat/stats/zstat*.nii.gz']
MANIFEST = DST / 'COMPACT_MANIFEST.csv'


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def classify():
    keep, links, drop = [], [], {}
    for root, dirs, files in os.walk(SRC):
        r = Path(root)
        for name in dirs + files:
            p = r / name
            rel = p.relative_to(SRC).as_posix()
            if p.is_symlink():
                links.append(rel)
                if name in dirs:
                    dirs.remove(name)              # do not descend into linked folders
                continue
            if name in dirs:
                continue
            size = p.stat().st_size
            if size < SMALL or any(fnmatch.fnmatch(rel, pat) for pat in KEEP_BIG):
                keep.append((rel, size))
            else:
                kind = name if not name.startswith(('cope', 'varcope', 'zstat', 'pe', 'tstat')) else 'other stats'
                drop[kind] = drop.get(kind, 0) + size
    return keep, links, drop


def plan():
    keep, links, drop = classify()
    kb, db = sum(s for _, s in keep), sum(drop.values())
    print(f'keep: {len(keep)} files, {kb / 1e9:.1f} GB, plus {len(links)} links')
    print(f'leave behind: {db / 1e9:.1f} GB, mostly:')
    for k, v in sorted(drop.items(), key=lambda kv: -kv[1])[:8]:
        print(f'  {v / 1e9:7.1f} GB  {k}')
    free = shutil.disk_usage(DST.parent).free / 1e9
    print(f'destination {DST} {"EXISTS" if DST.exists() else "is new"}; free space reported there: {free:.0f} GB')
    return keep, links


def copy():
    if DST.exists():
        sys.exit(f'STOP: {DST} already exists; nothing copied')
    keep, links = plan()
    DST.mkdir(parents=True)
    rows = []
    for i, (rel, size) in enumerate(keep, 1):
        s, d = SRC / rel, DST / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        hs, hd = md5(s), md5(d)
        if hs != hd:
            sys.exit(f'STOP: copy of {rel} does not match the original')
        rows.append({'path': rel, 'bytes': size, 'md5': hd, 'link_to': ''})
        if i % 2000 == 0:
            print(f'  {i}/{len(keep)} files copied and checked')
    for rel in links:
        s, d = SRC / rel, DST / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.readlink(s), d)
        rows.append({'path': rel, 'bytes': 0, 'md5': '', 'link_to': os.readlink(s)})
    with open(MANIFEST, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'bytes', 'md5', 'link_to'])
        w.writeheader(); w.writerows(rows)
    print(f'copied and checked {len(keep)} files and {len(links)} links into {DST}')


def verify():
    rows = list(csv.DictReader(open(MANIFEST)))
    bad = []
    for r in rows:
        d = DST / r['path']
        if r['link_to']:
            if not d.is_symlink() or os.readlink(d) != r['link_to']:
                bad.append(r['path'])
        elif not d.is_file() or md5(d) != r['md5'] or (SRC / r['path']).exists() and md5(SRC / r['path']) != r['md5']:
            bad.append(r['path'])
    keep, links, _ = classify() if SRC.exists() else ([], [], {})
    listed = {r['path'] for r in rows}
    missing = [rel for rel, _ in keep if rel not in listed] + [l for l in links if l not in listed]
    print(f'{len(rows)} entries checked; mismatched: {len(bad)}; kept files not in the copy: {len(missing)}')
    print('PASS: the compact copy is complete and identical' if not bad and not missing else
          'FAIL: ' + ', '.join((bad + missing)[:10]))


if __name__ == '__main__':
    {'plan': plan, 'copy': copy, 'verify': verify}[sys.argv[1] if len(sys.argv) > 1 else 'plan']()
