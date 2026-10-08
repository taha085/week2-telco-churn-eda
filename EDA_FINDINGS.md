# EDA Findings — Telco Customer Churn

## Dataset and preparation

This analysis uses the IBM Telco Customer Churn sample dataset distributed through Kaggle. The standard file contains **7,043 customers** and a binary `Churn` target.

The known `TotalCharges` issue was handled explicitly: the column is stored as text because **11 customers have blank values**, and those customers have `tenure = 0`. These blanks represent customers who have not accumulated billed total charges yet, rather than random missing observations. For numerical EDA, the blanks are converted to numeric and set to `0.0`.

No train/test split is used because this task is full-dataset EDA only. No predictive model is trained.

## Target distribution and class balance

The standard dataset contains:

- **No Churn:** 5,174 customers (**73.46%**)
- **Churn:** 1,869 customers (**26.54%**)

This is a meaningful class imbalance: the non-churn class is about 2.77 times as large as the churn class.

The imbalance matters because a model can obtain apparently strong accuracy by mostly predicting the majority class while doing a poor job identifying actual churners. For the later modeling stages, accuracy should therefore be reported alongside **precision, recall, F1-score**, and ideally a confusion matrix. Recall is especially important if the business objective is to identify as many potential churners as possible.

## Bivariate insight 1 — Contract type vs churn

Contract type shows one of the clearest relationships with churn. In the standard dataset, the approximate churn rates are:

| Contract | Approx. churn rate |
|---|---:|
| Month-to-month | **42.7%** |
| One year | **11.3%** |
| Two year | **2.8%** |

Customers on month-to-month contracts churn at a dramatically higher rate than customers on one- or two-year contracts. This is consistent with the idea that customers with less contractual commitment have fewer barriers to leaving. The analysis is descriptive, not causal: the chart does not prove that the contract itself causes churn. Contract type may also be associated with tenure, pricing, services, or customer segment. Nevertheless, it is a high-value feature to investigate in later modeling.

## Bivariate insight 2 — Tenure vs churn

Churned customers tend to have substantially shorter tenure than retained customers. A commonly reproduced result on the standard dataset is approximately:

- **Churn = Yes:** mean tenure ≈ **18 months**
- **Churn = No:** mean tenure ≈ **38 months**

The boxplot and tenure-band analysis should show that churn is concentrated more heavily among newer customers, while long-tenure customers are much more likely to remain. This suggests an important customer-lifecycle pattern: the early part of the customer relationship is a higher-risk period. Again, this is an association rather than proof that tenure itself causes retention.

## Bivariate insight 3 — Monthly charges vs churn

Churned customers generally have higher monthly charges than retained customers. On the standard dataset, the approximate group means are:

- **Churn = Yes:** mean monthly charges ≈ **$74.4**
- **Churn = No:** mean monthly charges ≈ **$61.3**

The boxplot should show a higher central tendency for churners, although there is substantial overlap between the two groups. This means monthly charges alone cannot cleanly separate churners from non-churners. The relationship may also reflect differences in contract type, internet service, add-on services, and customer tenure. Therefore, the result should be interpreted as an association rather than a causal effect.

## Numerical distributions and skewness

The main continuous variables are `tenure`, `MonthlyCharges`, and `TotalCharges`; `SeniorCitizen` is technically numeric in the CSV but is actually a binary indicator.

Two distributions worth flagging as skewed are:

1. **TotalCharges:** strongly right-skewed because customers accumulate charges over time. A relatively small number of long-tenure/high-spending customers produce a long upper tail. The mean is therefore pulled upward and should be considered alongside the median.
2. **Tenure:** moderately right-skewed because the distribution contains many newer customers and a smaller concentration of customers at high tenure values. Median and quantiles provide useful context in addition to the mean.

`MonthlyCharges` is comparatively less skewed, but still has a non-uniform distribution caused by different service combinations.

### Why mean, median, variance, and standard deviation differ in usefulness

- **Mean:** useful as an overall average, but sensitive to extreme values and skew.
- **Median:** more robust for skewed variables because it is less affected by unusually large observations.
- **Variance:** measures squared dispersion and is useful for comparing variability, but its units are squared and it is sensitive to outliers.
- **Standard deviation:** is the square root of variance and is easier to interpret because it is in the original units.

## Correlation heatmap

The Pearson correlation heatmap for numerical features should highlight:

- **Tenure vs TotalCharges:** a strong positive relationship (roughly **0.83**). This is expected because total charges accumulate as tenure increases.
- **MonthlyCharges vs TotalCharges:** a moderately strong positive relationship (roughly **0.65**).
- **Tenure vs MonthlyCharges:** a weaker positive relationship (roughly **0.25**).
- `SeniorCitizen` has relatively weak linear relationships with the other numerical variables.

The strong `tenure`–`TotalCharges` relationship is important for later feature engineering. `TotalCharges` is close to a derived cumulative measure involving tenure and monthly charges, so these variables may carry overlapping information. This is not leakage in the current EDA because no model is being trained, but it is worth remembering when building a future model.

Correlation also has clear limits: it measures statistical association, not causation, and Pearson correlation only captures linear association between the numeric representations. A weak correlation does not necessarily mean that two variables are unrelated in a nonlinear or categorical sense.

## Overall conclusion

The EDA establishes three practical patterns before any modeling begins: **month-to-month customers churn much more often, churn is concentrated among shorter-tenure customers, and churners tend to have higher monthly charges**. At the same time, the target is imbalanced at roughly 26.5% positive class, so a majority-class baseline is essential. The baseline demonstrates why accuracy by itself is not sufficient for churn prediction.
