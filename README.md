# Cystoscopy evaluation workflow

Code and reproducibility materials for AI-based bladder cancer detection in cystoscopy.

## Contents

- `implementation/src`, `implementation/scripts`, `implementation/configs`: training and evaluation code.
- `implementation/outputs/v2`: data splits, configurations and model predictions.
- `analysis_tools`: calibration, statistical analysis and verification scripts.
- `revision_analysis`: reference results.
- [Model checkpoints](https://github.com/LLUCCIFFER/cystoscopy-evaluation/releases): trained models and download files.

## Reproduce the analyses

Use Python 3.12:

```sh
python -m pip install -r analysis_tools/requirements_analysis_locked.txt
python analysis_tools/revision_reanalysis_qa.py
python analysis_tools/revision_reanalysis.py --reps 20 --out smoke_results
python analysis_tools/revision_reanalysis.py --reps 2000 --out reproduced_results
python analysis_tools/verify_reproduction.py reproduced_results
```

These commands reproduce the reported statistical analyses from saved predictions.
For the matched component comparisons (Table R11), run:

```sh
python analysis_tools/reproduce_heldout_components.py --out paired_heldout_components_reproduced.csv
```

Calibration solver sensitivity is available through
`python analysis_tools/revision_calibration_sensitivity.py`.

## Data and checkpoints

Obtain the images from the original sources:
[CystoDS](https://doi.org/10.17605/osf.io/xvdhy) and
[EBTC](https://doi.org/10.5281/zenodo.7741476).

Download, assemble and verify the checkpoints with:

```sh
python analysis_tools/assemble_checkpoints.py --download
```

Extract the verified ZIP into the repository root to restore the checkpoint paths.
Download details and checksums are in `CHECKPOINT_PARTS.json`.

Training environment and analysis provenance are documented in
[PROVENANCE.md](PROVENANCE.md). Third-party components retain their original terms.
