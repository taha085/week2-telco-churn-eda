# Week 2 — Telco Customer Churn EDA & Naive Baseline

## Files

- `eda_churn.py` — complete reproducible EDA + majority-class baseline script
- `EDA_FINDINGS.md` — written interpretation of the three required bivariate analyses, skewness, class balance, and correlations
- `BASELINE.md` — baseline metrics and explanation of why accuracy is misleading
- `figures/` — generated automatically by `eda_churn.py`

## Dataset

Download the **Telco Customer Churn** CSV from the Kaggle/IBM sample dataset and place it in the project folder as either:

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`, or
- `Telco-Customer-Churn.csv`

The script also checks the `data/` directory.

## Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python eda_churn.py
```

The script generates:

1. Churn distribution
2. Tenure distribution
3. Monthly charges distribution
4. Total charges distribution
5. Contract vs churn rate
6. Tenure vs churn
7. Tenure-band churn rate
8. Monthly charges vs churn
9. Numerical correlation heatmap

It also prints the numerical summary statistics, skewness, bivariate group statistics, correlations, and majority-class baseline metrics.

## Important data-quality note

`TotalCharges` is stored as text in the original CSV. The 11 blank entries belong to customers with `tenure = 0`, so they are treated as zero accumulated charges for this EDA rather than as random missingness.

## Expected baseline

For the standard 7,043-row dataset:

- Accuracy: 73.46%
- Churn precision: 0.00%
- Churn recall: 0.00%
- Churn F1: 0.00%

## Submission

Push this folder to a public GitHub repository and submit the public repository URL, or publish the notebook/script and markdown files through the platform accepted by your internship.
