# Training and analysis provenance


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
