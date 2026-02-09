import pandas as pd

raw_dir = '/lab_data/behrmannlab/hemi/Raw'
out_dir = '/user_data/csimmon2/sym_pt/sub-007/ses-03/timing'

conditions = ['Face', 'House', 'Object', 'Word', 'Scramble']

for run in ['01', '02']:
    events = pd.read_csv(f'{raw_dir}/sub-007/ses-03/func/sub-007_ses-03_task-loc_run-{run}_events.tsv', sep='\t')
    
    # Check what column and values we have
    print(f"Run {run} columns: {list(events.columns)}")
    print(f"Run {run} trial types: {events['trial_type'].unique()}")
    
    for cond in conditions:
        rows = events[events['trial_type'] == cond]
        outfile = f'{out_dir}/catloc_007_run-{run}_{cond}.txt'
        with open(outfile, 'w') as f:
            for _, row in rows.iterrows():
                f.write(f"{row['onset']:.3f} {row['duration']:.3f} 1\n")
        print(f"  {cond}: {len(rows)} blocks -> {outfile}")