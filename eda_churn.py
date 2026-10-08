"""
Week 2 — Telco Customer Churn
Statistical Profiling, EDA & Naive Majority-Class Baseline

Expected input:
    WA_Fn-UseC_-Telco-Customer-Churn.csv
or:
    Telco-Customer-Churn.csv

Run:
    python eda_churn.py

The script saves plots to ./figures and prints the numerical results needed
for EDA_FINDINGS.md and BASELINE.md.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_CANDIDATES = [
    Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    Path("Telco-Customer-Churn.csv"),
    Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    Path("data/Telco-Customer-Churn.csv"),
]

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data():
    for path in DATA_CANDIDATES:
        if path.exists():
            print(f"Loading: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        "Dataset not found. Download the IBM/Kaggle Telco Customer Churn CSV "
        "and place it beside this script or in ./data/."
    )


def clean_total_charges(df):
    df = df.copy()
    # Preserve the raw string behavior long enough to document the known quirk.
    raw_blank = df["TotalCharges"].astype(str).str.strip().eq("").sum()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    zero_tenure_blank = df["TotalCharges"].isna() & df["tenure"].eq(0)
    other_missing = df["TotalCharges"].isna() & ~df["tenure"].eq(0)

    print("\nTotalCharges quality check")
    print(f"  Blank/non-numeric TotalCharges values: {raw_blank}")
    print(f"  Converted NaN values: {df['TotalCharges'].isna().sum()}")
    print(f"  NaN with tenure == 0: {zero_tenure_blank.sum()}")
    print(f"  NaN with tenure > 0: {other_missing.sum()}")

    # For EDA, the blank values represent customers with zero tenure / no billed
    # total yet. Set them to 0 rather than treating them as random missingness.
    df.loc[zero_tenure_blank, "TotalCharges"] = 0.0
    return df


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    df = load_data()
    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    df = clean_total_charges(df)

    # -----------------------------
    # 1. Target distribution
    # -----------------------------
    target_counts = df["Churn"].value_counts().reindex(["No", "Yes"])
    target_pct = target_counts / len(df) * 100

    print("\n=== TARGET DISTRIBUTION ===")
    print(pd.DataFrame({"count": target_counts, "percent": target_pct.round(2)}))

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="Churn", order=["No", "Yes"])
    for container in ax.containers:
        ax.bar_label(container)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of customers")
    savefig("01_churn_distribution.png")

    # -----------------------------
    # 2. Numerical summary statistics
    # -----------------------------
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    stats = df[numeric_cols].describe().T
    stats["variance"] = df[numeric_cols].var()
    stats["skewness"] = df[numeric_cols].skew()
    print("\n=== NUMERICAL SUMMARY ===")
    print(stats.round(3).to_string())

    # -----------------------------
    # 3. Univariate distributions
    # -----------------------------
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        savefig(f"02_{col.lower()}_distribution.png")

    # -----------------------------
    # 4. Bivariate analysis #1:
    # Contract vs churn rate
    # -----------------------------
    contract_rate = (
        df.groupby("Contract", observed=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )
    print("\n=== CONTRACT VS CHURN RATE (%) ===")
    print(contract_rate.round(2).to_string())

    plt.figure(figsize=(8, 5))
    order = contract_rate.index.tolist()
    ax = sns.barplot(
        data=df, x="Contract", y=(df["Churn"] == "Yes").astype(int),
        order=order, estimator=np.mean, errorbar=None
    )
    ax.set_ylabel("Churn rate")
    ax.set_xlabel("Contract")
    ax.set_title("Churn Rate by Contract Type")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    savefig("03_contract_vs_churn.png")

    # -----------------------------
    # 5. Bivariate analysis #2:
    # Tenure vs churn
    # -----------------------------
    tenure_by_churn = df.groupby("Churn")["tenure"].agg(["mean", "median", "std"])
    print("\n=== TENURE BY CHURN ===")
    print(tenure_by_churn.round(2).to_string())

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="tenure", order=["No", "Yes"])
    plt.title("Tenure by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (months)")
    savefig("04_tenure_vs_churn.png")

    # Also provide binned churn rates, which are easier to interpret.
    bins = [-1, 6, 12, 24, 48, 72]
    labels = ["0–6", "7–12", "13–24", "25–48", "49–72"]
    df["tenure_band"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    tenure_rate = (
        df.groupby("tenure_band", observed=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
    )
    print("\nTenure-band churn rates (%)")
    print(tenure_rate.round(2).to_string())

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=tenure_rate.index, y=tenure_rate.values, errorbar=None
    )
    ax.set_title("Churn Rate by Tenure Band")
    ax.set_xlabel("Tenure band (months)")
    ax.set_ylabel("Churn rate (%)")
    savefig("05_tenure_band_churn_rate.png")

    # -----------------------------
    # 6. Bivariate analysis #3:
    # Monthly charges vs churn
    # -----------------------------
    monthly_by_churn = df.groupby("Churn")["MonthlyCharges"].agg(["mean", "median", "std"])
    print("\n=== MONTHLY CHARGES BY CHURN ===")
    print(monthly_by_churn.round(2).to_string())

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", order=["No", "Yes"])
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    savefig("06_monthly_charges_vs_churn.png")

    # -----------------------------
    # 7. Numerical correlation heatmap
    # -----------------------------
    corr = df[numeric_cols].corr(method="pearson")
    print("\n=== NUMERICAL CORRELATION MATRIX ===")
    print(corr.round(3).to_string())

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap — Numerical Features")
    savefig("07_correlation_heatmap.png")

    # -----------------------------
    # 8. Majority-class baseline
    # -----------------------------
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X, y)
    y_pred = dummy.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision_churn": precision_score(y, y_pred, pos_label="Yes", zero_division=0),
        "recall_churn": recall_score(y, y_pred, pos_label="Yes", zero_division=0),
        "f1_churn": f1_score(y, y_pred, pos_label="Yes", zero_division=0),
    }

    print("\n=== MAJORITY-CLASS BASELINE ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y, y_pred, labels=["No", "Yes"]))

    print("\nDone. Figures saved in ./figures")
    print("Use the printed tables/metrics to update the two markdown files if your CSV differs.")
    return df


if __name__ == "__main__":
    main()
