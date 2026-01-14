# 📊 Benchmarking TabPFN Against Traditional and Relational Models for Provider Fraud Detection

This repository contains the **fully reproducible Python pipeline** used in the paper:

> **Benchmarking TabPFN Against Traditional and Relational Models for Provider Fraud Detection**

The project benchmarks **traditional tabular models**, a **tabular foundation model (TabPFN)**, and a **graph neural network (GraphSAGE)** on a real-world healthcare provider fraud detection dataset, with a focus on **class imbalance, calibration, and cost-sensitive evaluation**.

---

## 🔍 Overview

Healthcare fraud detection is challenging due to:
- Extreme class imbalance  
- Heterogeneous tabular features  
- Relational dependencies between providers and beneficiaries  

This repository evaluates the following approaches at the **provider level**:

- Logistic Regression  
- LightGBM  
- CatBoost  
- **TabPFN (Tabular Foundation Model)**  
- **GraphSAGE (Graph Neural Network)**  

**Key findings:**
- **TabPFN achieves the strongest overall performance without task-specific tuning**
- GraphSAGE remains competitive by leveraging provider–beneficiary relationships
- Cost-sensitive evaluation reveals practical trade-offs beyond ROC-AUC

---

## 📂 Repository Structure

```
.
├── benchmark_provider_fraud.py   # Main experiment script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── figures/
│   ├── PR_curve_selected.png
│   ├── calibration_curves_selected.png
│   └── AP_bar_chart.png
└── data/                         # (Not included – see Dataset section)
```

---

## 📊 Dataset

This project uses the **Healthcare Provider Fraud Detection Analysis** dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis

### Files used:
- `Train-1542865627584.csv`
- `Train_Inpatientdata-1542865627584.csv`
- `Train_Outpatientdata-1542865627584.csv`
- `Train_Beneficiarydata-1542865627584.csv`

⚠️ **Note:**  
The dataset is **not included** in this repository due to licensing restrictions.  
Download it manually from Kaggle and update the path in the script:

```python
base = r"C:\path\to\healthcare_dataset\"
```

---

## 🧠 Feature Construction

All modeling is performed at the **provider level** to avoid label leakage.

Features include:
- Inpatient and outpatient claim statistics (mean, sum, count)
- Number of distinct beneficiaries per provider
- Aggregated beneficiary demographics (gender mode, race mean)

Missing values are imputed with zero.

---

## 🧩 Models Implemented

### Tabular Models
- Logistic Regression (class-balanced)
- LightGBM (class-weighted)
- CatBoost (class-weighted)
- **TabPFN** (no hyperparameter tuning)

### Graph Model
- **GraphSAGE** with:
  - Providers as nodes
  - Edges between providers sharing beneficiaries
  - Two aggregation layers (64 → 16)

---

## ⚙️ Installation

### Requirements
- Python **3.8 – 3.11**
- CPU sufficient (GPU optional)

### Install dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas scikit-learn matplotlib
pip install lightgbm catboost
pip install torch torchvision torchaudio
pip install torch-geometric
pip install tabpfn
```

📌 PyTorch Geometric docs: https://pytorch-geometric.readthedocs.io/

---

## ▶️ Running the Benchmark

```bash
python benchmark_provider_fraud.py
```

The script will:
1. Load and aggregate the dataset
2. Train tabular and graph models
3. Compute discrimination, calibration, and cost-sensitive metrics
4. Export publication-ready figures

---

## 📈 Outputs

Generated figures:
- Precision–Recall curves
- Calibration curves
- Average Precision (AP) bar chart

Saved as:
```
PR_curve_selected.png
calibration_curves_selected.png
AP_bar_chart.png
```

---

## 📐 Evaluation Metrics

- ROC-AUC
- PR-AUC
- F1 score
- Balanced Accuracy
- Brier score
- **Maximum Expected Net Gain**

---

## 🔁 Reproducibility

- Fixed random seed
- Unified train–test split
- Identical features across models
- CPU-compatible execution

---

## 📄 Citation

```bibtex
@article{tabpfn_provider_fraud,
  title={Benchmarking TabPFN Against Traditional and Relational Models for Provider Fraud Detection},
  author={Your Name},
  year={2026}
}
```

---

## 🤝 License

MIT License

---

## 📬 Contact

For questions or collaboration:
- GitHub: https://github.com/yourusername
