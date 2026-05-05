# Session 5: Practical Assessment – Advanced Anomaly Detection

**Course:** Machine Learning III (Unsupervised Learning) @Albert School  
**Format:** Groups of 1 to 3 students.  
**Duration:** 3 hours (Due at the end of the session).  
**Grading:** Graded (Low-impact, incentive-based).

---

## The Business Scenario

You are the Lead Data Science team for a major manufacturing firm. The company operates expensive, heavy machinery that occasionally suffers from catastrophic failures, halting production and costing €100,000 per hour of downtime.

Your operations team has provided telemetry data collected from these machines (temperatures, torque, tool wear, etc.). Standard rules-based monitoring is no longer sufficient. Your objective is to design and implement an unsupervised anomaly detection pipeline capable of flagging potential machine failures *before* they occur, while minimizing alert fatigue (False Positives) for the maintenance crew.

---

## The Dataset

**AI4I 2020 Predictive Maintenance Dataset** (Available on the UCI Machine Learning Repository or Kaggle).

This dataset contains 10,000 observations with multiple sensor readings describing machine behavior.

> **Constraint:** The dataset contains a column named *Machine failure*. You must drop this column during the training phase. You are only allowed to use it at the very end to evaluate whether your unsupervised models successfully identified real failures.

---

## Expected Deliverables

You must submit a comprehensive Jupyter Notebook (or equivalent report) structured around two complementary perspectives: the **Technical Data Scientist** and the **Business Manager**.

---

## Part 1: Exploratory Data Analysis (EDA) & Cleaning

1. Investigate the features: identify missing values, skewed distributions, and correlations.
2. **Preprocessing:** Standardize numerical features. Explain why scaling is necessary for distance-based methods such as LOF and Elliptic Envelope, but not strictly required for Isolation Forest.
3. **Categorical handling:** Explain how you treat the machine *Type* feature in distance-based algorithms.

---

## Part 2: Modeling & Hyperparameter Tuning

Train and tune the four core anomaly detection models covered in Lecture 4:

1. Isolation Forest
2. One-Class SVM
3. Local Outlier Factor (LOF)
4. Robust Covariance (Elliptic Envelope)

> **Requirement:** Do not keep default hyperparameters. Tune parameters such as *contamination*, *nu*, and *n_neighbors*. Justify all choices using insights derived from your EDA.

---

## Part 3: Technical Comparison & Visualizations

1. **Dimensionality Reduction for Visualization:** use PCA or t-SNE to project the data into 2D or 3D.
2. Visualize model boundaries by overlaying predictions (`-1` for anomaly, `1` for normal) for all four models.
3. **Deep Dive:** identify at least one observation flagged as anomalous by one model but not another. Inspect raw sensor values and explain mathematically or geometrically why this discrepancy occurs based on each model's assumptions.

---

## Part 4: Managerial Conclusion & Actionable Strategy

1. **Cost Analysis:** Assume a False Positive costs €500 and a False Negative costs €15,000.
2. **The Reveal:** Reintroduce the *Machine failure* column and compare predictions with ground truth.
3. **Final Decision:** determine which model minimizes financial loss and recommend the optimal business threshold (`contamination` or `nu`).

---

## Grading Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Code Quality & ML Pipeline** | 30% | Clean, well-commented code and correct use of Scikit-Learn APIs |
| **Analytical Rigor** | 30% | Depth of EDA and justified hyperparameter choices |
| **Critical Insight** | 40% | Interpretation of results, explanation of anomalies, and accurate financial reasoning |
