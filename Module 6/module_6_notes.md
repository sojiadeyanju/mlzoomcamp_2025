# 🌳 Mastering Decision Trees and Ensemble Learning: The Heart of Machine Learning Models

##### From simple trees to powerful ensembles — learn how Decision Trees, Random Forests, and XGBoost transform predictive modeling.
---
## 🧠 Introduction

##### In this module, we explore one of the most powerful families of algorithms in machine learning — Decision Trees and their ensemble counterparts (Random Forests and XGBoost).

##### We’ll use a Credit Risk Scoring Project as our case study: predicting whether a bank should lend money to a client based on their financial history.

* A “0” means client will likely repay → loan approved.
* A “1” means potential defaulter → loan denied.

## 🧹 Data Cleaning & Preparation

* Download and reformat categorical columns (status, home, marital, records, job).
* Replace extreme values in income, assets, and debt with NaN.
* Fill missing values with 0.
* Keep only clients with status = ok or default.
* Split into:
  * 60% training
  * 20% validation
  * 20% test (using random seed = 11).
* Convert target (status) to binary:
  * 0 → ok
  * 1 → default.
---
## 🌳 Decision Trees

##### Decision Trees make predictions using a series of if/else rules based on feature thresholds.

* Pros: Simple to interpret, flexible for both classification and regression.
* Cons: Prone to overfitting, especially with deep trees.

> 💡 A Decision Stump is a shallow tree with only one split — useful for understanding base logic.

### Key Parameters:

* max_depth: limits how deep the tree can grow.
* min_samples_leaf: ensures each leaf has enough data to generalize.
* export_text: shows readable decision rules.

##### To prevent overfitting: reduce max_depth and adjust min_samples_leaf.

## ⚙️ Decision Tree Learning Algorithm

##### At each node:

1. Evaluate all features and possible thresholds.
2. Choose the split that minimizes impurity (or misclassification rate).
3. Repeat recursively until stopping criteria are met.

### Impurity Criteria:

* Gini Impurity
* Entropy
* (For regression: MSE)

### Stopping Conditions:

* Node is pure (0% impurity).
* max_depth reached.
* Group too small or max number of leaves reached.

---

## 🧩 Parameter Tuning

##### Tuning max_depth and min_samples_leaf is crucial for balancing bias vs variance:

* Increase max_depth → more complex, higher variance.
* Decrease max_depth → simpler, higher bias.

##### Use heatmaps to visualize combinations of these hyperparameters and find the sweet spot for model AUC.

`from sklearn.tree import DecisionTreeClassifier`

`from sklearn.metrics import roc_auc_score`

`import seaborn as sns`

`import pandas as pd`

---
## 🌲 Ensemble Learning & Random Forests

##### Ensemble learning combines multiple “weak” models (e.g., decision trees) to create a stronger predictor.

### 🪵 Random Forests

* Each tree is trained on a bootstrapped sample (random rows + random features).
* Predictions from all trees are aggregated (majority vote).
* Reduces overfitting and improves generalization.

### Key Parameters:

* max_depth: controls complexity.
* n_estimators: number of trees.
* random_state: ensures reproducibility.
---
> 🌿 Randomness in data sampling and feature selection keeps trees decorrelated and robust.
---
## ⚡ Gradient Boosting & XGBoost

##### Unlike Random Forests (parallel), Gradient Boosting builds models sequentially — each model corrects the errors of the previous one.

### 🔥 XGBoost (Extreme Gradient Boosting)

* Highly optimized implementation of gradient boosting.
* Handles both classification and regression.
* Requires data wrapped in a special structure: DMatrix.

### Key Hyperparameters:
| Parameter | Purpose |
| ------------- |:-------------:|
| eta      | Learning rate; controls step size     |
| max_depth      | Tree depth (complexity)    |
| min_child_weight      | Minimum samples per leaf     |
| subsample       |  Fraction of data used per iteration  |
| colsample_bytree  |  Fraction of features per tree  |
| lambda / alpha  |  L2/L1 regularization  |
---
> 🚧 On Mac, XGBoost may require installing a compatible version of libomp.
Use conda install -c conda-forge xgboost for smooth setup.
---
## 🧠 Model Selection

##### Compare AUC scores across:

* **Decision Tree**
* **Random Forest**
* **XGBoost**

##### Choose the model that best generalizes across train, validation, and test sets.
##### Usually, XGBoost performs best on tabular data but needs careful tuning to prevent overfitting.

## 🏁 Summary
| Algorithm     | Description              | Strengths        | Weaknesses      |
| ------------- | ------------------------ | ---------------- | --------------- |
| Decision Tree | Simple if/else rules     | Interpretability | Overfitting     |
| Random Forest | Ensemble of trees        | Robust, stable   | Slower training |
| XGBoost       | Sequential boosted trees | Best performance | Complex tuning  |
---

## 💬 Explore More

* Try **Extra Trees Classifier** for faster training.
* Use **feature importance** from trees for interpretability.
* Apply to regression tasks with DecisionTreeRegressor or RandomForestRegressor.

