#!/usr/bin/env python3
"""
Submit registration jobs for 1st level FEAT outputs
Registers filtered_func_data to ses-01 anatomical space

to run: python submit_register_1stlevel.py
to monitor: squeue -u $USER
"""

import subprocess
from glob import glob
import os
import sys
import time

# Add path for imports
sys.path.insert(0, '/user_data/csimmon2/git_repos/sym_pt')
from sym_pt_params import processed_dir, get_sessions

# Job parameters
job_name = 'register_1stlevel'
mem = 4  # GB - FLIRT with applyxfm is lightweight
run_time = "01:00:00"
pause_crit = 12  # Number of jobs before pausing
pause_time = 1   # Minutes to pause

# Script to run
SCRIPT_PATH = '/user_data/csimmon2/git_repos/sym_pt/A_preprocessing/08_register_1stLevel.py'

def setup_sbatch(job_name, script_name):
    """Create SLURM sbatch script content"""
    sbatch_setup = f"""#!/bin/bash -l
# Job name
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=csimmon2@andrew.cmu.edu

# Submit job to cpu queue                
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Job memory request
#SBATCH --mem={mem}gb

# Time limit days-hrs:min:sec
#SBATCH --time {run_time}

# Standard output and error log
#SBATCH --output=slurm_out/{job_name}.out

# Load modules
module load fsl/6.0.4
export FSLDIR=/opt/fsl/6.0.4
. $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH

{script_name}
"""
    return sbatch_setup

def create_job(job_name, job_cmd):
    """Create and submit a SLURM job"""
    print(f"Submitting job: {job_name}")
    print(f"Command: {job_cmd}")
    
    # Create temporary script file
    script_file = f"{job_name}.sh"
    with open(script_file, "w") as f:
        f.write(setup_sbatch(job_name, job_cmd))
    
    # Submit job
    try:
        result = subprocess.run(['sbatch', script_file], check=True, capture_output=True, text=True)
        print(f"  ✓ Job submitted: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error submitting job: {e}")
        print(f"  ✗ sbatch stderr: {e.stderr}")
        print(f"  ✗ sbatch stdout: {e.stdout}")
    
    # Clean up script file
    if os.path.exists(script_file):
        os.remove(script_file)

# Create output directory for slurm logs
os.makedirs('slurm_out', exist_ok=True)

# Get all subjects
subject_dirs = glob(f'{processed_dir}/sub-*')
subjects = sorted([os.path.basename(d).replace('sub-', '') for d in subject_dirs if os.path.isdir(d)])

print(f"Found {len(subjects)} subjects")
print("")

# Job submission loop
n_jobs = 0

for sub_num in subjects:
    sub = f'sub-{sub_num}'
    
    # Get sessions for this subject
    sessions = get_sessions(sub_num)
    
    for ses_num in sessions:
        ses = f'{ses_num:02d}'
        
        # Create job command
        job_cmd = f'python3 {SCRIPT_PATH} {sub} {ses}'
        job_name_full = f'{sub}_ses{ses}_register'
        
        create_job(job_name_full, job_cmd)
        n_jobs += 1
        
        # Pause if we've submitted too many jobs
        if n_jobs >= pause_crit:
            print(f"\n🛑 Pausing for {pause_time} minutes after submitting {n_jobs} jobs...")
            time.sleep(pause_time * 60)
            n_jobs = 0

print(f"\n✅ Finished submitting all jobs!")
print(f"Total jobs submitted: {n_jobs}")
print("\nTo check job status: squeue -u $USER")
print("To check job details: scontrol show job <job_id>")
print("To cancel jobs: scancel <job_id> or scancel -u $USER")