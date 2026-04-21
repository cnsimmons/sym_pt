#!/usr/bin/env python3
"""
Submit registration jobs for zstats/copes to ses-01 anatomical space

to run: python 10_submit_register_zstats.py
to monitor: squeue -u $USER
"""

import subprocess
import os
import sys
import time

sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, get_sessions
from glob import glob

# Job parametersx
job_name = 'register_zstats'
mem = 16  # GB
run_time = "01:00:00"

# Script to run
SCRIPT_PATH = '/user_data/csimmon2/git_repos/sym_pt/A_preprocessing/register_zstats.py'

def setup_sbatch(job_name, script_name):
    """Create SLURM sbatch script content"""
    sbatch_setup = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem={mem}gb
#SBATCH --time {run_time}
#SBATCH --output=slurm_out/{job_name}.out

# Load modules
module load fsl/6.0.3
export FSLDIR=/opt/fsl/6.0.3
. $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH

{script_name}
"""
    return sbatch_setup

def create_job(job_name, job_cmd):
    """Create and submit a SLURM job"""
    print(f"Submitting job: {job_name}")
    
    script_file = f"{job_name}.sh"
    with open(script_file, "w") as f:
        f.write(setup_sbatch(job_name, job_cmd))
    
    try:
        result = subprocess.run(['sbatch', script_file], check=True, capture_output=True, text=True)
        print(f"  ✓ Job submitted: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error submitting job: {e}")
    
    if os.path.exists(script_file):
        os.remove(script_file)

def get_running_jobs():
    """Count currently running/pending jobs"""
    result = subprocess.run(['squeue', '-u', os.environ['USER'], '-h'], 
                          capture_output=True, text=True)
    return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

# Create output directory
os.makedirs('slurm_out', exist_ok=True)

# Get all subjects
subject_dirs = glob(f'{processed_dir}/sub-*')
subjects = sorted([os.path.basename(d).replace('sub-', '') for d in subject_dirs if os.path.isdir(d)])
#subjects = ['108']

print(f"Found {len(subjects)} subjects")
print("")

# Job submission loop with queue checking
for sub_num in subjects:
    sub = f'sub-{sub_num}'
    sessions = get_sessions(sub_num)
    
    for ses_num in sessions:
        ses = f'{ses_num:02d}'
        
        # Check that at least one FEAT run completed for this session
        feat_dir = f'{processed_dir}/{sub}/ses-{ses}/derivatives/fsl/loc'
        feat_runs = glob(f'{feat_dir}/run-*/1stLevel.feat/stats/cope1.nii.gz')
        if not feat_runs:
            print(f'  SKIP {sub} ses-{ses}: no completed FEAT runs found')
            continue

        # Wait for queue to have space (max 12 jobs)
        while get_running_jobs() >= 12:
            print(f"🛑 Waiting... ({get_running_jobs()}/12 jobs running)")
            time.sleep(30)
        
        job_cmd = f'python3 {SCRIPT_PATH} {sub} {ses}'
        job_name_full = f'{sub}_ses{ses}_regzstats'
        
        create_job(job_name_full, job_cmd)

print(f"\n✅ Finished submitting all jobs!")
print("\nTo check job status: squeue -u $USER")