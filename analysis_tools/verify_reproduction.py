from pathlib import Path
import sys
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
candidate=Path(sys.argv[1])
names=['seedwise_and_mean_cluster_intervals.csv','training_seed_variability.csv','group_composition.csv','leave_one_case_out.csv','calibration_intercept_slope.csv','calibration_paired_changes.csv','cancer_reliability_bins.csv','modality_intervals_counts.csv','component_validation_points.csv','paired_component_validation.csv','paired_comparator_differences.csv']
for name in names:
 a=pd.read_csv(root/'revision_analysis'/name);b=pd.read_csv(candidate/name)
 assert list(a.columns)==list(b.columns) and a.shape==b.shape,name
 for col in a:
  if pd.api.types.is_numeric_dtype(a[col]):assert np.allclose(a[col],b[col],atol=1e-12,rtol=0,equal_nan=True),(name,col)
  else:assert a[col].fillna('').equals(b[col].fillna('')),(name,col)
print('PASS: 11 analysis tables reproduce within absolute tolerance 1e-12')
