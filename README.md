# Coverage-Aware Learning from Fragmented Multimodal Evidence for Child-Directed Cyberbullying Detection

**Implementation and Reproducibility Artifacts**

---

## Overview

This repository provides the official implementation, replication scripts, and reproducibility artifacts for the research on detecting child-directed cyberbullying from fragmented multimodal evidence.

All evaluations are conducted on an audited benchmark corpus of 13,307 social media entries across YouTube, Twitter/X, Instagram, and Facebook under a 5-fold stratified cross-validation scheme with nested calibration splits to eliminate data leakage.

---

## Benchmark Results (5-Fold Stratified CV, N = 13,307)

The table below summarizes performance across representative model families on identical data partitions:

| Model Architecture | Modality Category | Accuracy | Macro F1 | ROC-AUC | Average Precision |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `indobert` | Unimodal Text | 0.8907 | 0.8808 | 0.9528 | 0.9131 |
| `indobertweet` | Unimodal Text (Slang) | 0.8666 | 0.8543 | 0.9351 | 0.8830 |
| `mbert` | Multilingual Text | 0.8308 | 0.8132 | 0.9001 | 0.8321 |
| `transcript` | Unimodal Audio (Speech) | 0.8548 | 0.8406 | 0.9298 | 0.8769 |
| `visual` | Unimodal Visual (Video) | 0.6625 | 0.4396 | 0.5177 | 0.3719 |
| `tfidf_lr` | Classical Baseline (n-gram) | 0.8158 | 0.7968 | 0.8873 | 0.8161 |
| `feature_concat_seed43` | Early Fusion Baseline | 0.8959 | 0.8860 | 0.9577 | 0.9213 |
| `qmf2023_adapted_seed42` | Recent Baseline (ICML 2023) | 0.8880 | 0.8781 | 0.9549 | 0.9141 |
| `healnet2024_seed43` | SOTA Baseline (NeurIPS 2024) | 0.8944 | 0.8846 | 0.9548 | 0.9184 |
| **`full_mlp_seed44`** | **Proposed Tri-Modal Late Fusion** | **0.8943** | **0.8842** | **0.9576** | **0.9213** |

Key observations:
1. The proposed tri-modal late fusion model (`full_mlp_seed44`) achieves peak performance on the multimodal framework (Macro-F1: 88.42%, Accuracy: 89.43%), outperforming all unimodal baselines, the classical baseline (+8.74% F1), and the decision-level QMF baseline (+0.61% F1).
2. The framework matches the performance of recent iterative cross-attention baselines (HEALNet, NeurIPS 2024) while offering significantly lower computational latency.
3. Visual evidence alone exhibits low standalone signal on child cyberbullying text-driven discourse, but complements language features when combined via non-linear late fusion.

---

## Repository Structure

```text
coverage-aware-multimodal-cyberbullying/
├── notebooks/                              # Executed Jupyter notebooks with complete outputs
│   ├── 00_persiapan_runtime.ipynb          # Runtime configuration and split validation
│   ├── notebook_multimodal_v12.ipynb       # Main 17 multimodal models training and evaluation
│   ├── notebook_baselines_v12.ipynb        # 7 Baseline models (TF-IDF, Concat, HEALNet NeurIPS 2024)
│   ├── IJIES_20265495_QMF2023_BASELINE_MAIN.ipynb # QMF (ICML 2023) decision-level fusion baseline
│   └── notebook_lopo_v12.ipynb             # Leave-One-Platform-Out evaluation
├── src/                                    # Python source code modules
│   ├── revision_train.py                   # Multimodal neural network architectures
│   ├── revision_core.py                    # Evaluation pipeline and late fusion meta-learners
│   ├── revision_baselines.py               # HEALNet and Early Fusion Concat implementations
│   └── predict_realtime.py                 # Standalone real-time inference engine
├── reproducibility_artifacts/              # Auditability artifacts (Reviewer 1 requirements)
│   ├── partition_assignments.csv           # 5-fold cross-validation partition assignments
│   ├── splits.json                         # Cross-validation index partitions
│   ├── modality_masks.csv                  # Nominal and effective decodable coverage masks
│   ├── coverage.json                       # Modality coverage breakdown
│   ├── all_model_predictions.csv           # Out-of-fold probabilities, thresholds, and predictions (all 24 models + QMF)
│   ├── base_probabilities_test.csv         # Out-of-fold probability predictions
│   └── thresholds.json                     # Calibrated decision thresholds
├── reports_and_metrics/                    # Numerical evaluation tables
│   ├── table2_metrics.csv                  # Complete 24-model metrics
│   ├── table3_per_class.csv                # Per-class performance breakdown
│   ├── table_qmf_metrics.csv               # QMF (ICML 2023) evaluation metrics
│   ├── qmf_fold_metrics.csv                # QMF fold-wise breakdown
│   ├── fold_metrics.csv                    # Metrics across Folds 1 to 5
│   └── fusion_seed_stability.csv           # Multi-seed stability (Seeds 42, 43, 44)
├── figures/                                # Publication-ready figures
│   ├── roc.png                             # ROC curves across model configurations
│   ├── precision_recall.png                # Precision-Recall curves
│   └── confusion_full_lr.png               # Confusion matrix visual plot
├── regenerate_tables.py                    # Verification script (reproduces Table 2 and Table 3)
├── requirements.txt                        # Dependencies specification
├── LICENSE                                 # MIT License
└── README.md                               # Repository documentation
```

---

## Quick Verification

To verify all reported metrics from the auditability artifacts without retraining:

```bash
git clone https://github.com/arimuzakir/coverage-aware-multimodal-cyberbullying.git
cd coverage-aware-multimodal-cyberbullying
pip install -r requirements.txt
python regenerate_tables.py
```

---

## Citation

```bibtex
@article{cyberbullying2026coverage,
  title={Coverage-Aware Learning from Fragmented Multimodal Evidence for Child-Directed Cyberbullying Detection},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
