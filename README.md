# Cystoscopy evaluation workflow

Implementation, exact public-dataset splits, recorded predictions, configurations,
and statistical analyses for an image-level cystoscopy evaluation study.

## Reproduce the reported analyses

Use Python 3.12 and install the analysis dependencies:

```sh
python -m pip install -r analysis_tools/requirements_analysis_locked.txt
python analysis_tools/revision_reanalysis_qa.py
python analysis_tools/revision_reanalysis.py --reps 20 --out smoke_results
python analysis_tools/revision_reanalysis.py --reps 2000 --out reproduced_results
```

These commands use saved predictions and train no model. The last command reproduces
the common-cluster sampling analysis for three fixed trained models. Run
`python analysis_tools/verify_reproduction.py reproduced_results` to compare the
11 reported analysis tables against the committed reference values.

The calibration-only solver sensitivity can be reproduced with
`python analysis_tools/revision_calibration_sensitivity.py`; it retains the original
primary temperatures and reports the converged transformations separately.

## Contents

- `implementation/src`, `implementation/scripts`, `implementation/configs`: model and evaluation source.
- `implementation/outputs/v2`: exact split manifests, saved predictions, search records and settings.
- `analysis_tools`: numerical reanalysis and independent checks.
- `revision_analysis`: reference tables and statistical definitions.

Three primary Swin-Tiny checkpoints are distributed as seven bounded-size parts
in the accompanying release. Restore and verify the full checkpoint archive with:

```sh
python analysis_tools/assemble_checkpoints.py --download
```

The command checks every part and the assembled ZIP against `CHECKPOINT_PARTS.json`.
Extract `checkpoint_release_20261030.zip` into the repository root to restore the
recorded checkpoint locations. The complete archive is also supplied directly as
Supplementary Code 2 with the manuscript revision.
The release provides no raw cystoscopy images; obtain CystoDS and EBTC from their
original repositories (https://doi.org/10.17605/osf.io/xvdhy and
https://doi.org/10.5281/zenodo.7741476).

## Interpretation and provenance

The primary estimand is the mean of image-level metrics across three fixed models.
Seed variability is reported separately from conditional sampling intervals.
EBTC resampling groups use normalized case identifiers in the released filenames;
they are analysis groups, with patient/procedure linkage unavailable in these files.

Recovered local execution records establish Python 3.10.20, PyTorch 2.11.0+cu128,
CUDA 12.8 and timm 1.0.27 for formal training. The retained environment, run
manifests and checkpoints provide corroborating evidence. Initialization identifiers
were reconstructed from this timm version, factory calls and download records.
`revision_analysis/recovered_training_provenance.json` distinguishes historical
outputs, recovered-environment versions and retained-cache hashes. Original manifests
did not embed initialization-file hashes or a Git commit. Current numerical-analysis
dependencies remain pinned separately. This release does not claim an end-to-end
retraining verification.
Existing third-party components remain subject to their original terms.

`ARCHIVED_SOURCE_SHA256.csv` records the earlier review archive source hashes.
`MANIFEST_SHA256.csv` records the current published file bytes. The independent QA
script uses repository-relative paths; numerical procedures are unchanged. The
`date` field in reference analysis manifests is the original protocol date,
2026-10-18; reproduction runs do not represent new model training.

## Matched component test comparisons

Table R11 uses fixed archived checkpoints, raw cancer probabilities and threshold
0.5 on internal and external test cohorts. Augmentation compares three matched
seeds; sampling compares seed 42. No checkpoint, calibration or threshold was
selected using these test comparisons. All listed metrics and contrasts are retained.
Reproduce the 2,000 paired cluster draws from the supplied probabilities with:

```sh
python analysis_tools/reproduce_heldout_components.py --out paired_heldout_components_reproduced.csv
```

Reference values and the comparison protocol are in
`revision_analysis/heldout_components`. The 70 independent scikit-learn metric
checks are documented there. Intervals condition on the fixed models and grouping;
they do not estimate future retraining variation.
