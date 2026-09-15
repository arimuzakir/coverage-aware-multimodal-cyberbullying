# Coverage-Aware Learning from Fragmented Multimodal Evidence for Child-Directed Cyberbullying Detection

[![Paper](https://img.shields.io/badge/Paper-IJIES%2020265495-blue)](https://www.inass.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository provides the official implementation, replication package, and auditability artifacts for the research paper:

> **"Coverage-Aware Learning from Fragmented Multimodal Evidence for Child-Directed Cyberbullying Detection"**  
> *Authors:* Ari Muzakir, Fajar Romadhan, et al.  
> *Target Journal:* International Journal of Intelligent Engineering and Systems (IJIES), Paper ID: 20265495.

---

## 📌 Abstract & Overview

Detecting cyberbullying directed at children across social media platforms presents distinct challenges due to informal slang, nuanced hostility, and severely fragmented multimodal evidence. In real-world social platforms (YouTube, Twitter/X, Instagram, and Facebook), auxiliary modalities (speech audio and video frames) are available only for a small fraction of posts.

This study introduces a **Coverage-Aware Multimodal Framework** evaluating **24 distinct model configurations** across a clean corpus of **$N = 13,307$** child-directed social media entries evaluated under a strict **5-Fold Stratified Cross-Validation** with nested calibration splits.

### Key Highlights:
- **Tri-Modal Late Fusion (`full_mlp_seed44`)**: Achieves peak performance on the proposed architecture with **89.43% Accuracy**, **88.42% Macro-F1**, and **0.9576 ROC-AUC**, outperforming all single-modality baselines.
- **Controlled SOTA Benchmarking**: Direct head-to-head comparison against recent SOTA **HEALNet (*NeurIPS 2024*)** (88.46% Macro-F1) and classical n-gram TF-IDF (79.68% Macro-F1) on identical data splits.
- **Auditability & Numerical Reproducibility**: Complete inclusion of 5-fold assignments, modality coverage masks, out-of-fold probability files, and calibrated decision thresholds.

---

## 📊 Benchmark Results (24 Models, 5-Fold Stratified CV, $N = 13,307$)

| No | Model Architecture | Category | Accuracy | Macro F1 | ROC-AUC | PR-AUC | Role / Note |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `indobert` | Unimodal Text | 0.8907 | 0.8808 | 0.9528 | 0.9131 | Strongest monolingual text backbone |
| 2 | `indobertweet` | Unimodal Text | 0.8666 | 0.8543 | 0.9351 | 0.8830 | Indonesian social media & slang encoder |
| 3 | `mbert` | Unimodal Text | 0.8308 | 0.8132 | 0.9001 | 0.8321 | Multilingual BERT baseline |
| 4 | `transcript` | Unimodal Audio | 0.8548 | 0.8406 | 0.9298 | 0.8769 | Acoustic / speech transcript modality |
| 5 | `visual` | Unimodal Visual | 0.6625 | 0.4396 | 0.5177 | 0.3719 | Visual video frames (weak direct signal) |
| 6 | `text_mean` | Text Ensemble | 0.8883 | 0.8783 | 0.9542 | 0.9137 | Mean probability text ensemble |
| 7 | `text_lr` | Late Fusion (LR) | 0.8917 | 0.8819 | 0.9580 | 0.9219 | Text meta-classifier |
| 8 | `text_speech_lr` | Late Fusion (LR) | 0.8912 | 0.8814 | 0.9579 | 0.9216 | Bimodal Text + Speech |
| 9 | `text_visual_lr` | Late Fusion (LR) | 0.8918 | 0.8819 | 0.9580 | 0.9220 | Bimodal Text + Visual |
| 10 | `speech_visual_lr` | Late Fusion (LR) | 0.8544 | 0.8402 | 0.9298 | 0.8773 | Bimodal Non-text (Speech + Visual) |
| 11 | `full_lr` | Late Fusion (LR) | 0.8910 | 0.8812 | 0.9579 | 0.9216 | Tri-modal Logistic Regression |
| 12 | `text_mlp_seed42` | Late Fusion (MLP) | 0.8916 | 0.8819 | 0.9582 | 0.9222 | Text MLP (Seed 42) |
| 13 | `full_mlp_seed42` | Late Fusion (MLP) | 0.8937 | 0.8836 | 0.9581 | 0.9217 | Tri-modal MLP (Seed 42) |
| 14 | `text_mlp_seed43` | Late Fusion (MLP) | 0.8920 | 0.8828 | 0.9584 | 0.9224 | Text MLP (Seed 43) |
| 15 | `full_mlp_seed43` | Late Fusion (MLP) | 0.8916 | 0.8821 | 0.9582 | 0.9214 | Tri-modal MLP (Seed 43) |
| 16 | `text_mlp_seed44` | Late Fusion (MLP) | 0.8924 | 0.8826 | 0.9581 | 0.9215 | Text MLP (Seed 44) |
| 17 | **`full_mlp_seed44`** | **Late Fusion (MLP)** | **0.8943** | **0.8842** | **0.9576** | **0.9213** | **Proposed Best Multimodal Model** |
| 18 | `tfidf_lr` | Classical Baseline | 0.8158 | 0.7968 | 0.8873 | 0.8161 | Word(1,2) + Char(3,5) Logistic Regression |
| 19 | `feature_concat_seed42` | Early Fusion Concat | 0.8926 | 0.8828 | 0.9553 | 0.9196 | Early Fusion MLP (Seed 42) |
| 20 | `feature_concat_seed43` | Early Fusion Concat | 0.8959 | 0.8860 | 0.9577 | 0.9213 | Early Fusion MLP (Seed 43) |
| 21 | `feature_concat_seed44` | Early Fusion Concat | 0.8924 | 0.8825 | 0.9561 | 0.9201 | Early Fusion MLP (Seed 44) |
| 22 | `healnet2024_seed42` | SOTA Baseline | 0.8910 | 0.8806 | 0.9546 | 0.9153 | HEALNet (*NeurIPS 2024*, Seed 42) |
| 23 | `healnet2024_seed43` | SOTA Baseline | 0.8944 | 0.8846 | 0.9548 | 0.9184 | HEALNet (*NeurIPS 2024*, Seed 43) |
| 24 | `healnet2024_seed44` | SOTA Baseline | 0.8941 | 0.8840 | 0.9551 | 0.9161 | HEALNet (*NeurIPS 2024*, Seed 44) |

---

## 📁 Repository Structure

```text
.
├── notebooks/                              # Verified Executed Notebooks (with outputs)
│   ├── 00_persiapan_runtime.ipynb         # Environment preflight & manifest verification
│   ├── notebook_multimodal_v12.ipynb      # Main 17 multimodal models training & evaluation
│   ├── notebook_baselines_v12.ipynb       # 7 Baseline models (TF-IDF, Concat, HEALNet NeurIPS 2024)
│   └── notebook_lopo_v12.ipynb            # Leave-One-Platform-Out cross-platform evaluation
│
├── src/                                    # Source Code Modules
│   ├── revision_train.py                  # Core training routines & neural encoders
│   ├── revision_core.py                   # Core metrics, evaluation, and late fusion
│   ├── revision_baselines.py              # HEALNet and Early Fusion Concat implementations
│   └── predict_realtime.py                # Standalone real-time inference engine
│
├── reproducibility_artifacts/              # Auditability Artifacts (Reviewer 1 Requirements)
│   ├── partition_assignments.csv          # Exact 5-fold sample assignments
│   ├── splits.json                        # 5-Fold cross-validation indices
│   ├── modality_masks.csv                 # Nominal vs effective decodable modality masks
│   ├── coverage.json                      # Exact empirical modality coverage statistics
│   ├── environment.json                   # CUDA, PyTorch, and hardware environment specs
│   ├── base_probabilities_test.csv        # Stored out-of-fold probability predictions
│   ├── thresholds.json                    # Empirically calibrated decision thresholds
│   └── paired_uncertainty.json            # Statistical significance and confidence intervals
│
├── reports_and_metrics/                    # Benchmark Tables
│   ├── table2_metrics.csv                 # Official 24-model performance metrics
│   ├── table3_per_class.csv               # Per-class performance breakdown
│   ├── fold_metrics.csv                   # Per-fold validation breakdown (Folds 1-5)
│   └── fusion_seed_stability.csv          # Multi-seed stability analysis
│
├── figures/                                # Publication Figures
│   ├── roc.png                            # High-resolution ROC-AUC curve
│   ├── precision_recall.png               # High-resolution Precision-Recall curve
│   └── confusion_full_lr.png              # Visual confusion matrix plot
│
├── regenerate_tables.py                    # Verification script (Table 2 & 3 in 1 command)
├── requirements.txt                        # Python dependencies
├── LICENSE                                 # MIT License
└── README.md                               # This documentation
```

---

## ⚡ Quick Start & Verification

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/arimuzakir/coverage-aware-multimodal-cyberbullying.git
cd coverage-aware-multimodal-cyberbullying
pip install -r requirements.txt
```

### 2. Verify Benchmark Results Instantly
To verify all reported metrics without retraining, run the audit verification script:
```bash
python regenerate_tables.py
```
This script reads the locked out-of-fold prediction artifacts and reproduces **Table 2** and **Table 3** directly in your terminal.

### 3. Real-Time Inference
Test the trained multimodal cyberbullying detection pipeline:
```python
from src.predict_realtime import CyberbullyingInferenceEngine

engine = CyberbullyingInferenceEngine()

# Example: Text-only detection with automatic fallback
result = engine.predict(text="kamu anak bego gak guna mati aja")
print(result)
```

---

## 📖 Citation

If you find this code or dataset artifacts useful in your research, please cite:

```bibtex
@article{muzakir2026coverage,
  title={Coverage-Aware Learning from Fragmented Multimodal Evidence for Child-Directed Cyberbullying Detection},
  author={Muzakir, Ari and Romadhan, Fajar and others},
  journal={International Journal of Intelligent Engineering and Systems},
  volume={19},
  year={2026},
  note={Paper ID: 20265495}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
