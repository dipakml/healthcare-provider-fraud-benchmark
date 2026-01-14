"""
Benchmarking TabPFN Against Traditional and Relational Models
for Healthcare Provider Fraud Detection.

Run:
    python benchmark_provider_fraud.py

Requirements:
    Python >= 3.8
"""

# =========================================================
# Imports & Environment Setup
# =========================================================

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, balanced_accuracy_score,
    brier_score_loss, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier


# =========================================================
# GraphSAGE Model
# =========================================================

class GraphSAGE(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c1 = SAGEConv(d, 64)
        self.c2 = SAGEConv(64, 16)
        self.fc = torch.nn.Linear(16, 2)

    def forward(self, x, e):
        x = self.c1(x, e).relu()
        x = self.c2(x, e)
        return self.fc(x)


def train_gnn(data, epochs=200, lr=0.01):
    model = GraphSAGE(data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model(data.x, data.edge_index).softmax(1)[:, 1].detach().numpy()


# =========================================================
# Utility Functions
# =========================================================

def pr_auc(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    return auc(rc, pr)


def max_gain(y, p, R=10, C=1):
    best = -1e9
    for t in np.linspace(0, 1, 101):
        d = p > t
        best = max(
            best,
            (d & y).sum() * R - (d & ~y.astype(bool)).sum() * C
        )
    return best


# =========================================================
# Main Pipeline
# =========================================================

def main():

    # ---------------------------
    # 1. LOAD DATA
    # ---------------------------

    base = r"C:\Users\dipak\healthcare_dataset\\"

    labels = pd.read_csv(base + "Train-1542865627584.csv")
    inp = pd.read_csv(base + "Train_Inpatientdata-1542865627584.csv")
    outp = pd.read_csv(base + "Train_Outpatientdata-1542865627584.csv")
    bene = pd.read_csv(base + "Train_Beneficiarydata-1542865627584.csv")

    # ---------------------------
    # 2. PROVIDER FEATURES
    # ---------------------------

    agg_inp = inp.groupby("Provider")["InscClaimAmtReimbursed"] \
        .agg(["mean", "sum", "count"]).add_prefix("IP_")
    agg_out = outp.groupby("Provider")["InscClaimAmtReimbursed"] \
        .agg(["mean", "sum", "count"]).add_prefix("OP_")

    prov = agg_inp.join(agg_out, how="outer").fillna(0)

    prov["DistinctBene"] = pd.concat([inp, outp]) \
        .groupby("Provider")["BeneID"].nunique() \
        .reindex(prov.index).fillna(0)

    all_claims = pd.concat([inp, outp])
    merged = all_claims.merge(bene, on="BeneID", how="left")

    demo = merged.groupby("Provider").agg({
        "Gender": lambda x: x.mode().iloc[0] if len(x) else 0,
        "Race": "mean"
    }).fillna(0)

    prov = prov.join(demo, how="left").fillna(0)

    prov["y"] = labels.set_index("Provider")["PotentialFraud"] \
        .map({"Yes": 1, "No": 0}).reindex(prov.index).fillna(0).astype(int)

    X = prov.drop("y", axis=1).values
    y = prov["y"].values

    # ---------------------------
    # 3. TRAIN–TEST SPLIT
    # ---------------------------

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, stratify=y, test_size=0.2, random_state=42
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------
    # 4. GRAPH CONSTRUCTION
    # ---------------------------

    prov_list = prov.index.tolist()
    p2i = {p: i for i, p in enumerate(prov_list)}

    edges = set()
    for _, g in all_claims.groupby("BeneID"):
        ps = g["Provider"].unique()
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = p2i[ps[i]], p2i[ps[j]]
                edges.add((a, b))
                edges.add((b, a))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t()

    x = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_t,
        train_mask=train_mask,
        test_mask=test_mask
    )

    gnn_preds = train_gnn(data)[test_idx]

    # ---------------------------
    # 5. TABULAR MODELS
    # ---------------------------

    models = {
        "Logistic": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "LightGBM": LGBMClassifier(class_weight="balanced", verbosity=-1),
        "CatBoost": CatBoostClassifier(class_weights=[10, 1], verbose=False),
        "TabPFN": TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    }

    preds = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds[name] = model.predict_proba(X_test_scaled)[:, 1]

    preds["GraphSAGE"] = gnn_preds

    # ---------------------------
    # 6. METRICS
    # ---------------------------

    rows = []
    for k, p in preds.items():
        rows.append([
            k,
            roc_auc_score(y_test, p),
            pr_auc(y_test, p),
            f1_score(y_test, p > 0.5),
            balanced_accuracy_score(y_test, p > 0.5),
            brier_score_loss(y_test, p),
            max_gain(y_test, p)
        ])

    df = pd.DataFrame(rows, columns=[
        "Model", "ROC-AUC", "PR-AUC", "F1", "BalAcc", "Brier", "MaxNetGain"
    ])

    print("\nModel Performance Summary\n")
    print(df.to_string(index=False))

    # ---------------------------
    # 7. PLOTS
    # ---------------------------

    # Precision–Recall
    plt.figure(figsize=(10, 8))
    no_skill = y_test.mean()
    plt.plot([0, 1], [no_skill, no_skill], "--", color="gray",
             label=f"No Skill (Precision={no_skill:.3f})")

    for m in ["Logistic", "LightGBM", "TabPFN", "GraphSAGE"]:
        x_, y_, _ = precision_recall_curve(y_test, preds[m])
        plt.plot(x_, y_, linewidth=2.5, label=m)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curves - Provider Fraud Detection", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("PR_curve_selected.png", dpi=300)
    plt.close()

    # Calibration
    plt.figure(figsize=(10, 8))
    for m in ["Logistic", "TabPFN", "GraphSAGE"]:
        f, m_ = calibration_curve(y_test, preds[m], n_bins=10)
        b = brier_score_loss(y_test, preds[m])
        plt.plot(m_, f, marker="o", linewidth=3, label=f"{m} (Brier={b:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves - Provider Fraud Detection", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("calibration_curves_selected.png", dpi=300)
    plt.close()

    # Average Precision Bar Chart
    ap = {k: average_precision_score(y_test, p) for k, p in preds.items()}
    plt.figure(figsize=(8, 6))
    plt.bar(ap.keys(), ap.values())
    plt.ylabel("Average Precision")
    plt.title("Average Precision by Model", fontweight="bold")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("AP_bar_chart.png", dpi=300)
    plt.close()

    print("\n✅ Figures saved:")
    print("  - PR_curve_selected.png")
    print("  - calibration_curves_selected.png")
    print("  - AP_bar_chart.png")


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()
