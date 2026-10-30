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

The recorded training dependencies and pretrained initialization identifiers do
not provide a complete historical environment lock. Current analysis versions
are pinned separately and do not identify original training versions. The present
release verifies fixed-prediction analyses; it does not claim an identical retraining
environment. Exact configurations and provenance fields are retained for inspection.
Existing third-party components remain subject to their original terms.

`ARCHIVED_SOURCE_SHA256.csv` records the earlier review archive source hashes.
`MANIFEST_SHA256.csv` records the current published file bytes. The independent QA
script uses repository-relative paths; numerical procedures are unchanged. The
`date` field in reference analysis manifests is the original protocol date,
2026-10-18; reproduction runs do not represent new model training.
