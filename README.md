# Unsupervised Discovery and Predictive Modeling of Therapeutic Failure Phenotypes in the FAERS Database

This project performs end-to-end analysis of therapeutic failures reported in the FDA Adverse Event Reporting System (FAERS).
It combines NLP-based clustering and machine learning classification to discover hidden phenotypes, label failure cases, and build a predictive model for real-world deployment.

**<h3>Project Overview</h3>**

The end-to-end pipeline consists of:

- Data Integration & Preprocessing
- Unsupervised Failure Phenotype Discovery (Clustering)
- Feature Engineering for Classification
- Imbalanced Learning (SMOTE + Undersampling)
- Supervised Model Training
- Grid Search Optimization
- Final Real-World Evaluation
- Exported Artifacts for Deployment

**<h4>1. Data Preprocessing</h4>**

FAERS quarterly datasets (Drug, Reaction, Outcome tables) were merged and cleaned.
Key preprocessing steps:
- Merge all FAERS tables by primaryid
- Collapse multiple reactions per case
- Handle missing values
- Extract severity and hospitalization signals
- Create a binary is_failure target (based on FDA seriousness indicators)
- Extract clean text from all_reaction_pts


**<h4>2. Unsupervised Discovery of Failure Phenotypes (Clustering)</h4>**

We cluster failure cases using NLP-based representation:
- Text Embedding Pipeline
- TF-IDF Vectorizer
- max_features = 2000
- ngram_range = (1, 2)
- custom FAERS stopwords
- Dimensionality Reduction
- Truncated SVD → 50 components
- KMeans Clustering
- Searched K = 2 to 6 using silhouette score
- Optimal or forced K = 3

<h5>Discovered Phenotypes (Clusters)</h5>

| Cluster   | Meaning                                                                     |
| --------- | --------------------------------------------------------------------------- |
| Cluster 0 | **Critical Failure** — life-threatening terms (death, shock, organ failure) |
| Cluster 1 | **Hospitalization Failure** — emergency, pneumonia, infection               |
| Cluster 2 | **Side-Effect Failure** — headaches, nausea, mild symptoms                  |



**<h4>3. Classification Feature Engineering</h4>**
Dropped non-predictive identifiers:
primaryid, caseid, caseversion, fda_dt_parsed, all_reaction_pts, severity fields

Features were scaled using StandardScaler.
- The phenotype label was encoded using LabelEncoder.

<h5> Artifacts saved:</h5>

- scaler.joblib
- label_encoder.joblib
- training_feature_names.joblib

<h4>4. Imbalanced Learning Pipeline</h4>

To avoid data leakage, only the training split was balanced.

We used:
- SMOTE (oversampling minority classes) — 60%
- RandomUnderSampler — to keep 0/1/2 ratios reasonable

Saved artifacts:

- X_train_bal.joblib
- y_train_bal.joblib
- X_test.joblib
- y_test.joblib

**<h4> 5. Baseline Model Evaluation (Untuned)</h4>**

Trained on balanced data, tested on untouched test split.
Models evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

Scores recorded:
untuned_model_scores.csv

The top 5 models were automatically selected for tuning.

**<h4> 6. Grid Search Optimization</h4>**

A optimized grid search was run on:
-Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

Evaluation metrics:
- F1-Macro
- Accuracy
- PR-AUC

Each model produced a confusion matrix and classification report.

Results saved:
- tuned_model_scores.csv
- best_tuned_<model>.joblib

The best model (by F1-Macro) was selected as:

LightGBM (Tuned)
Saved as:
- best_classifier_final.joblib

**<h4>7. Final Real-World Evaluation</h4>**
Evaluated tuned model on unseen, real-world FAERS data:
Final Score:
- Accuracy: 0.9601
- F1-Macro: 0.8277
Performance Highlights
- Critical_Failure: F1 ≈ 0.98
- Hospitalization_Failure: F1 ≈ 0.95
- SideEffect_Failure: F1 ≈ 0.56

This confirms:
- Strong generalization
- No data leakage
- High real-world reliability

The full predictions were exported:
final_evaluation_predictions.csv
