# **Predicting Post-HCT Survival: An Ensemble Learning Approach**
## **Kaggle Link**
[**Competition Submission**](https://www.kaggle.com/code/muntazirmehdi786/xgboost-lightgbm-catboost-1)

## **Team Name:**
**3 GB Software Engineer**

---
## **Table of Contents**
- [Introduction](#introduction)
  - [Competition Overview](#competition-overview)
  - [Organizer: CIBMTR](#organizer-cibmtr)
  - [Research Importance](#research-importance)
- [Leaderboard Update](#leaderboard-update)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Selection & Training](#model-selection--training)
  - [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Discussion](#discussion)
- [Conclusion & Future Work](#conclusion--future-work)
- [References](#references)

---

## **Introduction**

### **Competition Overview**
The **Equity Post-HCT Survival Predictions** competition on Kaggle challenges participants to develop machine learning models for predicting survival outcomes after hematopoietic cell transplantation (HCT). The objective is to optimize the **Stratified Concordance Index (C-Index)** for survival ranking predictions.

### **Organizer: CIBMTR**
This competition is organized by the **Center for International Blood and Marrow Transplant Research (CIBMTR)**, a collaborative research program that collects and analyzes transplant-related data to improve patient outcomes. CIBMTR maintains a global registry and conducts research to advance HCT therapies, making this challenge highly relevant for clinical advancements.

### **Research Importance**
Predicting post-HCT survival is crucial for optimizing treatment strategies, allocating resources effectively, and improving patient care. Machine learning models can significantly enhance risk assessment and personalized treatment strategies.

---

## **Leaderboard Update**
- **Current Rank:** 2078
- **Leaderboard Improvement:** 82 positions
- **Current C-Index Score:** 0.6856

---

## **Dataset Description**
The dataset consists of **25 numerical** and **35 categorical** variables covering patient demographics, donor characteristics, transplant details, and clinical risk factors. The survival outcome is represented by:
- **`EFS_time`**: Event-free survival time (in days)
- **`EFS_event`**: Binary indicator of whether an event occurred

### **Key Features**
- **Patient Demographics**: Age, gender, ethnicity
- **Donor Characteristics**: Related/unrelated donor, HLA matching
- **Clinical Risk Scores**: Karnofsky score, comorbidity index
- **Cytogenetic & Molecular Risk Classifications**

---

## **Methodology**

### **3.1 Data Preprocessing**
```python
from sklearn.impute import SimpleImputer

# Handling missing values
categorical_imputer = SimpleImputer(strategy='most_frequent')
numerical_imputer = SimpleImputer(strategy='median')

train[categorical_cols] = categorical_imputer.fit_transform(train[categorical_cols])
train[numerical_cols] = numerical_imputer.fit_transform(train[numerical_cols])
```
**Why?** Missing values can cause model bias and errors; imputation maintains dataset completeness.

**Effect?** Preserves data integrity while minimizing information loss.

### **3.2 Feature Engineering**
```python
# Derived Features
def add_features(df):
    df['donor_age_hct_diff'] = df['donor_age'] - df['age_at_hct']
    df['comorbidity_karnofsky_ratio'] = df['comorbidity_score'] / (df['karnofsky_score'] + 1)
    df['efs_time_log'] = np.log1p(df['efs_time']) if 'efs_time' in df.columns else None
    df['year_hct_adjusted'] = df['year_hct'] - 2000
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    return df

train = add_features(train)
test = add_features(test)
```
**Why?** These transformations introduce domain-relevant interactions that improve model interpretability.

**Effect?** Enhances predictive performance by incorporating meaningful relationships between features.

```python
# Nelson-Aalen Target Transformation
def create_nelson(data):
    from lifelines import NelsonAalenFitter
    naf = NelsonAalenFitter(nelson_aalen_smoothing=0)
    naf.fit(durations=data['efs_time'], event_observed=data['efs'])
    return naf.cumulative_hazard_at_times(data['efs_time']).values * -1

train["y_nel"] = create_nelson(train)
train.loc[train.efs == 0, "y_nel"] = (-(-train.loc[train.efs == 0, "y_nel"])**0.5)
```
**Why?** The Nelson-Aalen estimator provides a non-parametric estimate of cumulative hazard rates, useful for survival modeling.

**Effect?** Encodes time-to-event information, improving the model's ability to rank patients.

```python
# Pairwise Logit Transform
def logit_transform(y, eps=2e-2, eps_mul=1.1):
    y = (y - y.min() + eps) / (y.max() - y.min() + eps_mul * eps)
    return np.log(y / (1 - y))

train["y_transformed"] = logit_transform(train["y_nel"])
```
**Why?** Logit transformation helps normalize survival targets for gradient boosting models.

**Effect?** Stabilizes variance, improving numerical stability and boosting performance.

### **3.3 Model Selection & Training**
```python
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Model Initialization
lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=7)
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=7)
cat = CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=7, verbose=0)
```
**Why?** Ensemble learning (LightGBM, XGBoost, CatBoost) optimizes predictive accuracy by leveraging different boosting strategies.

**Effect?** Balances bias-variance tradeoff, improving generalization on unseen data.

```python
# Stratified K-Fold Cross-Validation
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(train, train["efs"]):
    lgbm.fit(train.iloc[train_idx][FEATURES], train.iloc[train_idx]["y_transformed"])
```
**Why?** Ensures models are trained on balanced patient subsets, reducing variance in survival predictions.

**Effect?** Improves model robustness and prevents overfitting.

---

## **Experimental Results**

| Model | Public Score | Private Score |
|--------|------------|--------------|
| LightGBM | 0.68175 | 0.68561 |
| XGBoost | 0.68012 | 0.68430 |
| CatBoost | 0.67945 | 0.68392 |
| Ensemble (LGBM + XGB + CAT) | **0.68561** | **0.68175** |

---

## **Discussion**

### **Key Findings**
1. **Feature Engineering Impact**: The Nelson-Aalen transformation significantly improved ranking performance, allowing the model to incorporate time-to-event information effectively.
2. **Modeling Strategy**: LightGBM outperformed individual models due to its ability to handle categorical variables efficiently and its optimized boosting technique.
3. **Ensemble Benefits**: Combining LightGBM, XGBoost, and CatBoost resulted in a **C-Index improvement of ~0.004** over the best single model, demonstrating the power of ensembling.
4. **Fairness & Bias**: While the model showed general improvement, further analysis is required to ensure fairness across different demographic groups.

### **Challenges & Limitations**
- **Data Imbalance**: Certain patient groups had significantly fewer samples, impacting generalizability.
- **Computational Complexity**: Training multiple models and performing ensembling increased computation time.
- **Lack of Deep Learning Exploration**: While boosting methods performed well, deep survival models could further refine long-term risk predictions.

## **Conclusion & Future Work**

### **Conclusion**
This project successfully applied supervised learning and ensemble techniques to predict post-HCT survival. The key takeaways include:
- **Feature Engineering** was crucial in enhancing model performance.
- **Ensemble Learning** provided robustness and a notable leaderboard improvement.
- **Stratified C-Index Optimization** was effective in ranking survival predictions accurately.

### **Future Work**
- **Fairness Auditing**: Implement group-wise fairness metrics to ensure bias mitigation.
- **Deep Learning Models**: Investigate transformer-based survival models.
- **Advanced Feature Engineering**: Explore domain-specific genomic and clinical text embeddings.
- **Hyperparameter Fine-tuning**: Further optimize LightGBM’s boosting strategy with Bayesian optimization.

This approach provides a strong foundation for predictive modeling in clinical survival analysis and can be extended to other medical domains.
