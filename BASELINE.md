# Majority-Class Baseline — Telco Customer Churn

## Baseline definition

The naive baseline uses scikit-learn's `DummyClassifier(strategy="most_frequent")`.

It always predicts the majority class:

> **No Churn**

This is intentionally simple. The purpose is not to build a useful production model; it is to establish the minimum performance that any later classifier should beat.

## Results on the standard 7,043-row dataset

The standard dataset contains:

- Actual **No Churn:** 5,174 (73.46%)
- Actual **Churn:** 1,869 (26.54%)

Therefore, always predicting `No` gives:

| Metric | Baseline |
|---|---:|
| Accuracy | **73.46%** |
| Precision — Churn | **0.00%** |
| Recall — Churn | **0.00%** |
| F1 — Churn | **0.00%** |

The confusion matrix is:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 5,174 | 0 |
| Actual Yes | 1,869 | 0 |

## Interpretation

The baseline achieves **73.46% accuracy**, which initially looks reasonable. However, this is exactly why accuracy alone is misleading for an imbalanced classification problem.

The classifier identifies **zero churners**. Consequently, its churn precision, recall, and F1-score are all zero. In a churn-management setting, this baseline would be operationally poor because every real churner would be missed.

The accuracy is high only because the model receives credit for correctly predicting the much larger `No Churn` class. A later model should therefore be compared against this baseline using multiple metrics, especially **churn recall and F1**, rather than accuracy alone.

## Why keep this baseline?

The majority-class baseline provides a concrete reference point. If a sophisticated model cannot beat this simple rule on appropriate validation/test metrics, the additional complexity is not justified. The baseline also prevents us from interpreting a 70–75% accuracy score as impressive without considering the class distribution.

## Reproducible code

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X, y)

pred = dummy.predict(X)

print("Accuracy:", accuracy_score(y, pred))
print("Precision:", precision_score(y, pred, pos_label="Yes", zero_division=0))
print("Recall:", recall_score(y, pred, pos_label="Yes", zero_division=0))
print("F1:", f1_score(y, pred, pos_label="Yes", zero_division=0))
```

**Note:** This is an EDA/baseline exercise, so no train/test split is used here, as required by the task. Later modeling should use a proper train/test split and evaluate the final classifier on held-out data.
