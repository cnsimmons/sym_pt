#!/usr/bin/env python3
"""
generate_fsf.py - Create FSF files for long_pt first-level analysis
Uses long_pt_params for configuration
"""
import os
import sys
sys.path.insert(0, '/user_data/csimmon2/git_repos/long_pt')

from long_pt_params import (
    raw_dir, processed_dir, task, tr, mni_brain,
    contrast_weights, contrast_names, skip_subs, session_start,
    get_sessions, get_runs
)

FSF_TEMPLATE = '''
# FEAT version number
set fmri(version) 6.00

# Are we in MELODIC?
set fmri(inmelodic) 0

# Analysis level (1=first-level)
set fmri(level) 1

# Which stages to run
set fmri(filtering_yn) 1
set fmri(stats_yn) 1
set fmri(poststats_yn) 1

# Output directory
set fmri(outputdir) "{output_dir}"

# TR
set fmri(tr) {tr}

# Total volumes
set fmri(npts) {n_vols}

# Delete volumes
set fmri(ndelete) 0

# Perfusion tag/control order
set fmri(tagfirst) 1

# Number of first-level analyses
set fmri(multiple) 1

# Higher-level input type
set fmri(inputtype) 2

# Carry out pre-stats processing?
set fmri(filtering_yn) 1

# Brain/background threshold, %
set fmri(brain_thresh) 10

# Critical z for design efficiency calculation
set fmri(critical_z) 5.3

# Noise level
set fmri(noise) 0.66

# Noise AR(1)
set fmri(noisear) 0.34

# Motion correction
set fmri(mc) 1

# Spin-history correction
set fmri(sh_yn) 0

# B0 unwarping
set fmri(regunwarp_yn) 0

# Slice timing correction
set fmri(st) 0

# BET brain extraction
set fmri(bet_yn) 1

# Spatial smoothing FWHM (mm)
set fmri(smooth) 5

# Intensity normalization
set fmri(norm_yn) 0

# Perfusion subtraction
set fmri(perfsub_yn) 0

# Highpass temporal filtering
set fmri(temphp_yn) 1

# Highpass cutoff (s)
set fmri(paradigm_hp) 100

# Lowpass temporal filtering
set fmri(templp_yn) 0

# MELODIC ICA data exploration
set fmri(melodic_yn) 0

# Carry out main stats?
set fmri(stats_yn) 1

# Carry out prewhitening?
set fmri(prewhiten_yn) 1

# Add motion parameters to model
set fmri(motionevs) 0
set fmri(motionevsbeta) ""
set fmri(scriptevsbeta) ""

# Robust outlier detection
set fmri(robust_yn) 0

# Model setup
set fmri(mixed_yn) 2

# Number of EVs
set fmri(evs_orig) 5
set fmri(evs_real) 10
set fmri(evs_vox) 0

# Number of contrasts
set fmri(ncon_orig) 19
set fmri(ncon_real) 19

# Number of F-tests
set fmri(nftests_orig) 0
set fmri(nftests_real) 0

# Add constant column to model
set fmri(constcol) 0

# Carry out post-stats?
set fmri(poststats_yn) 1

# Pre-threshold masking
set fmri(threshmask) ""

# Thresholding
set fmri(thresh) 3

# P threshold
set fmri(prob_thresh) 0.05

# Z threshold
set fmri(z_thresh) 3.1

# Z min/max for colour rendering
set fmri(zdisplay) 0
set fmri(zmin) 2
set fmri(zmax) 8

# Colour rendering type
set fmri(rendertype) 1

# Background image type
set fmri(bgimage) 1

# Create time series plots
set fmri(tsplot_yn) 1

# Registration
set fmri(reginitial_highres_yn) 0
set fmri(reghighres_yn) 1
set fmri(regstandard_yn) 1
set fmri(alternateReference_yn) 0
set fmri(reghighres_dof) BBR
set fmri(regstandard_dof) 12
set fmri(regstandard_nonlinear_yn) 0
set fmri(regstandard_nonlinear_warpres) 10

# Standard space image
set fmri(regstandard) "{mni_brain}"

# High-res structural
set highres_files(1) "{anat_brain}"

# Functional data
set feat_files(1) "{func_file}"

# Confound EVs
set fmri(confoundevs) {has_confounds}
{confound_file_line}

# ============================================================================
# EVs
# ============================================================================
# EV 1: Face
set fmri(evtitle1) "Face"
set fmri(shape1) 3
set fmri(convolve1) 2
set fmri(convolve_phase1) 0
set fmri(tempfilt_yn1) 1
set fmri(deriv_yn1) 1
set fmri(custom1) "{timing_dir}/catloc_{sub}_run-{run}_Face.txt"

# EV 2: House
set fmri(evtitle2) "House"
set fmri(shape2) 3
set fmri(convolve2) 2
set fmri(convolve_phase2) 0
set fmri(tempfilt_yn2) 1
set fmri(deriv_yn2) 1
set fmri(custom2) "{timing_dir}/catloc_{sub}_run-{run}_House.txt"

# EV 3: Object
set fmri(evtitle3) "Object"
set fmri(shape3) 3
set fmri(convolve3) 2
set fmri(convolve_phase3) 0
set fmri(tempfilt_yn3) 1
set fmri(deriv_yn3) 1
set fmri(custom3) "{timing_dir}/catloc_{sub}_run-{run}_Object.txt"

# EV 4: Word
set fmri(evtitle4) "Word"
set fmri(shape4) 3
set fmri(convolve4) 2
set fmri(convolve_phase4) 0
set fmri(tempfilt_yn4) 1
set fmri(deriv_yn4) 1
set fmri(custom4) "{timing_dir}/catloc_{sub}_run-{run}_Word.txt"

# EV 5: Scramble
set fmri(evtitle5) "Scramble"
set fmri(shape5) 3
set fmri(convolve5) 2
set fmri(convolve_phase5) 0
set fmri(tempfilt_yn5) 1
set fmri(deriv_yn5) 1
set fmri(custom5) "{timing_dir}/catloc_{sub}_run-{run}_Scramble.txt"

# EV orthogonalization (none)
set fmri(ortho1.0) 0
set fmri(ortho1.1) 0
set fmri(ortho1.2) 0
set fmri(ortho1.3) 0
set fmri(ortho1.4) 0
set fmri(ortho1.5) 0
set fmri(ortho2.0) 0
set fmri(ortho2.1) 0
set fmri(ortho2.2) 0
set fmri(ortho2.3) 0
set fmri(ortho2.4) 0
set fmri(ortho2.5) 0
set fmri(ortho3.0) 0
set fmri(ortho3.1) 0
set fmri(ortho3.2) 0
set fmri(ortho3.3) 0
set fmri(ortho3.4) 0
set fmri(ortho3.5) 0
set fmri(ortho4.0) 0
set fmri(ortho4.1) 0
set fmri(ortho4.2) 0
set fmri(ortho4.3) 0
set fmri(ortho4.4) 0
set fmri(ortho4.5) 0
set fmri(ortho5.0) 0
set fmri(ortho5.1) 0
set fmri(ortho5.2) 0
set fmri(ortho5.3) 0
set fmri(ortho5.4) 0
set fmri(ortho5.5) 0

# ============================================================================
# CONTRASTS
# ============================================================================
{contrast_section}

# ============================================================================
# F-TESTS (none)
# ============================================================================

# Contrast masking
{contrast_mask_section}
'''

def get_n_vols(nifti_path):
    """Get number of volumes in 4D nifti"""
    import nibabel as nib
    img = nib.load(nifti_path)
    return img.shape[3] if len(img.shape) > 3 else 1


def generate_contrast_section():
    """Generate FSF contrast definitions"""
    lines = []
    for cope_num, weights in contrast_weights.items():
        name = contrast_names[cope_num]
        lines.append(f'# Contrast {cope_num}: {name}')
        lines.append(f'set fmri(conpic_orig.{cope_num}) 1')
        lines.append(f'set fmri(conname_orig.{cope_num}) "{name}"')
        
        # Real contrasts (with derivatives = 2x EVs)
        lines.append(f'set fmri(conpic_real.{cope_num}) 1')
        lines.append(f'set fmri(conname_real.{cope_num}) "{name}"')
        
        for ev_num, weight in enumerate(weights, 1):
            lines.append(f'set fmri(con_orig{cope_num}.{ev_num}) {weight}')
            # Real EV (main effect)
            real_ev = (ev_num - 1) * 2 + 1
            lines.append(f'set fmri(con_real{cope_num}.{real_ev}) {weight}')
            # Derivative EV (set to 0)
            lines.append(f'set fmri(con_real{cope_num}.{real_ev + 1}) 0')
        
        lines.append('')
    return '\n'.join(lines)


def generate_contrast_mask_section():
    """Generate contrast masking section (all zeros)"""
    lines = []
    for i in range(1, 20):
        for j in range(1, 20):
            lines.append(f'set fmri(conmask{i}_{j}) 0')
    return '\n'.join(lines)


def create_fsf(sub, ses, run):
    """Create FSF file for one run"""
    sub_clean = sub.replace('sub-', '')
    ses_str = f'{ses:02d}'
    run_str = f'{run:02d}'
    
    # Paths
    func_file = f'{raw_dir}/sub-{sub_clean}/ses-{ses_str}/func/sub-{sub_clean}_ses-{ses_str}_task-{task}_run-{run_str}_bold.nii.gz'
    anat_brain = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/anat/sub-{sub_clean}_ses-{ses_str}_T1w_brain.nii.gz'
    timing_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/timing'
    output_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/derivatives/fsl/{task}/run-{run_str}/1stLevel'
    
    # Confounds
    spike_file = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/derivatives/fsl/{task}/run-{run_str}/sub-{sub_clean}_ses-{ses_str}_task-{task}_run-{run_str}_bold_spikes.txt'
    has_confounds = 1 if os.path.exists(spike_file) and os.path.getsize(spike_file) > 0 else 0
    confound_line = f'set confoundev_files(1) "{spike_file}"' if has_confounds else ''
    
    # Get volumes
    if not os.path.exists(func_file):
        print(f'  SKIP: {func_file} not found')
        return None
    n_vols = get_n_vols(func_file)
    
    # Generate FSF content
    fsf_content = FSF_TEMPLATE.format(
        output_dir=output_dir,
        tr=tr,
        n_vols=n_vols,
        mni_brain=mni_brain,
        anat_brain=anat_brain,
        func_file=func_file,
        has_confounds=has_confounds,
        confound_file_line=confound_line,
        timing_dir=timing_dir,
        sub=sub_clean,
        run=run_str,
        contrast_section=generate_contrast_section(),
        contrast_mask_section=generate_contrast_mask_section()
    )
    
    # Write FSF
    fsf_dir = f'{processed_dir}/sub-{sub_clean}/ses-{ses_str}/derivatives/fsl/{task}/run-{run_str}'
    os.makedirs(fsf_dir, exist_ok=True)
    fsf_path = f'{fsf_dir}/1stLevel.fsf'
    
    with open(fsf_path, 'w') as f:
        f.write(fsf_content)
    
    print(f'  Created: {fsf_path}')
    return fsf_path


def main():
    import pandas as pd
    df = pd.read_csv(f'{processed_dir}/../git_repos/long_pt/long_pt_sub_info.csv')
    
    print('Generating FSF files...')
    for _, row in df.iterrows():
        sub = row['sub'].replace('sub-', '')
        
        if sub in skip_subs:
            print(f'SKIP: {sub}')
            continue
        
        sessions = get_sessions(sub, df)
        print(f'\n=== sub-{sub} ({len(sessions)} sessions) ===')
        
        for ses in sessions:
            runs = get_runs(sub, ses)
            print(f'  Session {ses}: {len(runs)} runs')
            
            for run in runs:
                create_fsf(sub, ses, run)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
