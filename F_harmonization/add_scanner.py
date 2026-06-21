#!/usr/bin/env python3
"""Add a per-session `scanner` column to sub_info.csv from BIDS JSON sidecars."""
import glob, json
import pandas as pd

raw_dir  = '/lab_data/behrmannlab/hemi/Raw'
csv_in   = '/user_data/csimmon2/git_repos/sym_pt/sub_info.csv'
csv_out  = '/user_data/csimmon2/git_repos/sym_pt/sub_info_scanner.csv'

def get_scanner(sub, ses):
    pat = f'{raw_dir}/{sub}/{ses}/func/*task-loc*_bold.json'
    hits = sorted(glob.glob(pat))
    if not hits:
        return 'MISSING'
    with open(hits[0]) as f:
        meta = json.load(f)
    name = (meta.get('ManufacturersModelName') or meta.get('Manufacturer') or '').lower()
    if 'prisma' in name: return 'Prisma'
    if 'verio'  in name: return 'Verio'
    return f'UNKNOWN:{name}'

df = pd.read_csv(csv_in)
df['scanner'] = [get_scanner(r['sub'], r['ses']) for _, r in df.iterrows()]
df.to_csv(csv_out, index=False)

print(df['scanner'].value_counts(dropna=False).to_string())
print(f'\nwrote {csv_out}')
print(df[df['scanner'].str.startswith(('MISSING', 'UNKNOWN'))][['sub', 'ses', 'scanner']].to_string(index=False))