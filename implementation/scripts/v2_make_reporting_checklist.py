"""Create a CLAIM/STARD-AI style reporting checklist for supplementary material."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("outputs/v2/reports")


ROWS = [
    {
        "domain": "Data source",
        "item": "Describe source datasets, acquisition context, patient/sample counts, modalities, and labels.",
        "status": "Complete",
        "evidence": "outputs/v2/tables/table1_dataset_split_statistics.csv; outputs/v2/reports/experiment_report_v2_final.md",
        "notes": "CystoDS is the main training dataset; EBTC is external validation.",
    },
    {
        "domain": "Inclusion/exclusion",
        "item": "State image validity checks, excluded invalid samples, and reasons where available.",
        "status": "Partial",
        "evidence": "outputs/v2/splits/*.csv; data/processed/master_metadata.csv",
        "notes": "Metadata includes is_valid and error_reason, but manuscript should explicitly summarize exclusion counts.",
    },
    {
        "domain": "Splits",
        "item": "Report train/validation/calibration/internal-test/external-test splits and patient-level leakage controls.",
        "status": "Complete",
        "evidence": "outputs/v2/splits/split_manifest.json; tests/test_split_no_leakage.py",
        "notes": "EBTC is held out from model selection, calibration, and threshold selection.",
    },
    {
        "domain": "Index test",
        "item": "Define the AI index test, inputs, preprocessing, backbone, loss, sampler, augmentation, and checkpoint selection.",
        "status": "Complete",
        "evidence": "outputs/v2/reports/experiment_report_v2_final.md; outputs/v2/final_validation/frozen_baselines/selected_frozen_runs.csv",
        "notes": "Primary v2 model is frozen as Swin-Tiny + CE class-balanced sampler + modality-aware augmentation.",
    },
    {
        "domain": "Reference labels",
        "item": "Describe target labels and mappings for CystoDS cancer binary and EBTC HGC/LGC/NTL/NST labels.",
        "status": "Complete",
        "evidence": "src/cdg_cysto/data/label_mapping.py; outputs/v2/splits/ebtc_external_all.csv",
        "notes": "Cancer-positive mapping: HGC/LGC positive; NTL/NST negative.",
    },
    {
        "domain": "Threshold selection",
        "item": "Specify that thresholds are selected only on CystoDS calibration data and then frozen.",
        "status": "Complete",
        "evidence": "outputs/v2/final_validation/frozen_baselines/threshold_metrics_by_seed.csv; outputs/v2/operating_points/named_operating_points_summary.csv",
        "notes": "Report default_0.5, Youden, and sensitivity_90 operating points.",
    },
    {
        "domain": "External validation",
        "item": "Report external validation dataset, no-tuning policy, and final EBTC metrics.",
        "status": "Complete",
        "evidence": "outputs/v2/tables/table5_frozen_baseline_ebtc_external.csv; outputs/v2/final_validation/frozen_baselines/frozen_baseline_comparison.csv",
        "notes": "B1 and B5 now include three-seed frozen validation.",
    },
    {
        "domain": "Calibration",
        "item": "Report temperature scaling fit set and before/after ECE, Brier score, and NLL.",
        "status": "Complete",
        "evidence": "outputs/v2/tables/table9_calibration_before_after.csv; outputs/v2/final_validation/frozen_baselines/calibration_before_after.csv",
        "notes": "Do not claim universal calibration improvement when metrics worsen.",
    },
    {
        "domain": "Operating points",
        "item": "Show sensitivity-specificity and PPV-NPV trade-offs for clinically relevant thresholds.",
        "status": "Complete",
        "evidence": "outputs/v2/operating_points/operating_point_analysis.md; outputs/v2/operating_points/threshold_tradeoff_ebtc_external.png",
        "notes": "sensitivity_90 increases cancer sensitivity but increases false positives.",
    },
    {
        "domain": "Subgroup analysis",
        "item": "Report modality-stratified performance and failure modes for BLC/WLC and WLI/NBI.",
        "status": "Complete",
        "evidence": "outputs/v2/tables/table8_modality_stratified_analysis.csv; outputs/v2/modality_analysis/modality_analysis.md",
        "notes": "BLC specificity remains a limitation.",
    },
    {
        "domain": "Failure mode",
        "item": "Report false negatives, false positives, high-confidence errors, and class-stratified EBTC errors.",
        "status": "Complete",
        "evidence": "outputs/v2/error_analysis/final_ebtc/high_confidence_errors.csv; outputs/v2/tables/table7_ebtc_class_error_analysis.csv",
        "notes": "Emphasize NTL false positives and BLC low specificity.",
    },
    {
        "domain": "Uncertainty",
        "item": "Report multi-seed variability and bootstrap confidence intervals for key metrics.",
        "status": "Complete",
        "evidence": "outputs/v2/bootstrap/bootstrap_ci.csv; outputs/v2/leaderboards/stage05_multiseed_summary.csv",
        "notes": "B1/B5 baseline fairness improved by three-seed frozen evaluation.",
    },
    {
        "domain": "Explainability",
        "item": "Report CAM/attention methods and quantitative mask-based localization metrics where available.",
        "status": "Partial",
        "evidence": "outputs/v3/reports/v3_multiseed_analysis.md; outputs/v3/diagnostics/ntl_confusion/ntl_confusion_analysis.md",
        "notes": "v3 attention improves localization but is not a robust classifier improvement.",
    },
    {
        "domain": "Limitations",
        "item": "Explicitly disclose external-domain gaps, BLC specificity, NTL false positives, and dataset-size constraints.",
        "status": "Complete",
        "evidence": "outputs/v2/reports/experiment_report_v2_final.md; outputs/v3/diagnostics/blc_specialist/blc_specialist_analysis.md",
        "notes": "BLC-specific threshold/calibration did not solve low specificity.",
    },
    {
        "domain": "Reproducibility",
        "item": "Provide code structure, configs, seeds, dependencies, and generated artifacts.",
        "status": "Complete",
        "evidence": "README.md; requirements.txt; pyproject.toml; configs/; scripts/",
        "notes": "All formal runs write resolved configs, metrics, logits, predictions, and reports.",
    },
    {
        "domain": "Ethics and availability",
        "item": "Report dataset licenses, access routes, privacy considerations, and intended-use limitations.",
        "status": "Partial",
        "evidence": "scripts/download_ebtc.sh; scripts/download_cystods.sh; README.md",
        "notes": "Manuscript should include dataset licensing/access wording and clinical non-deployment caveat.",
    },
]


def _to_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# CLAIM / STARD-AI Style Reporting Checklist",
        "",
        "This is a supplementary-material self-audit, not a verbatim official checklist. It is aligned with the CLAIM 2024 Update and STARD-AI diagnostic-accuracy reporting themes.",
        "",
        "References:",
        "- CLAIM 2024 Update: https://pubs.rsna.org/doi/abs/10.1148/ryai.240300",
        "- EQUATOR CLAIM record: https://www.equator-network.org/reporting-guidelines/checklist-for-artificial-intelligence-in-medical-imaging-claim-a-guide-for-authors-and-reviewers/",
        "- STARD-AI 2025: https://www.nature.com/articles/s41591-025-03953-8",
        "- EQUATOR STARD-AI record: https://www.equator-network.org/reporting-guidelines/the-stard-ai-reporting-guideline-for-diagnostic-accuracy-studies-using-artificial-intelligence/",
        "",
        "| Domain | Item | Status | Evidence | Notes |",
        "|---|---|---|---|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['domain']} | {row['item']} | {row['status']} | `{row['evidence']}` | {row['notes']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(ROWS)
    frame.to_csv(OUTPUT_DIR / "claim_stard_ai_style_checklist.csv", index=False)
    (OUTPUT_DIR / "claim_stard_ai_style_checklist.md").write_text(
        _to_markdown(frame),
        encoding="utf-8",
    )
    print(OUTPUT_DIR / "claim_stard_ai_style_checklist.md")


if __name__ == "__main__":
    main()
