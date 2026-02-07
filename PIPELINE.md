# long_pt Pipeline - Clean Start

## Quick Summary

**Problem**: Current GLM has differential contrasts only. RSA requires identity contrasts (raw per-condition estimates).

**Solution**: Add cope15-19 as identity contrasts, re-run FEAT.

---

## Contrast Specification (Liu 2025)

| Cope | Name | Weights | Purpose |
|------|------|---------|---------|
| 6 | Face-all | [4,-1,-1,-1,-1] | ROI definition |
| 7 | House-all | [-1,4,-1,-1,-1] | ROI definition |
| 8 | Object-all | [-1,-1,4,-1,-1] | ROI definition |
| 9 | Word-all | [-1,-1,-1,4,-1] | ROI definition |
| 15 | Face_raw | [1,0,0,0,0] | **RSA input** |
| 16 | House_raw | [0,1,0,0,0] | **RSA input** |
| 17 | Object_raw | [0,0,1,0,0] | **RSA input** |
| 18 | Word_raw | [0,0,0,1,0] | **RSA input** |
| 19 | Scramble_raw | [0,0,0,0,1] | **RSA input** |

---

## Pipeline Files

```
git_repos/long_pt/
├── long_pt_params.py      # Config (Ayzenberg style)
├── 01_organize.py         # Setup directories
├── 02_timing.sh           # Convert timing files
├── 03_confounds.sh        # Extract motion outliers
├── 04_register.py         # Skull strip, mirror, MNI reg
├── 05_generate_fsf.py     # Create FSF with 19 contrasts
├── 06_run_feat.sh         # Execute FEAT
└── 07_rsa_analysis.py     # RSA with identity contrasts
```

---

## Steps

1. **Update params**: Copy `long_pt_params.py` to git repo
2. **Generate new FSFs**: Run `generate_fsf.py`
3. **Re-run FEAT**: ~30min per run
4. **Verify**: Check cope15-19 exist in `1stLevel.feat/stats/`
5. **RSA**: Use cope15-18 for RDM construction

---

## RSA Analysis (Liu 2025 Method)

```python
from long_pt_params import cope_identity, cope_selective, categories

# 1. Define ROI using cope_selective (category > all others), top 10%
# 2. Extract patterns from cope_identity within ROI
# 3. Compute RDM: dissimilarity = 1 - pearson(pattern_i, pattern_j)
# 4. Compare T1 vs T2 RDMs via correlation
```

---

## Key Insight from Liu 2025

They computed **representational similarity** (Fisher-transformed correlation) between:
- Preferred category pattern
- Non-preferred category patterns

Within each ROI. Lower correlation = more selective/differentiable.

This requires raw per-condition activation patterns (cope_identity), not differential contrasts.
