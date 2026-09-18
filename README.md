# Mental Health Treatment Prediction

Predicting whether a survey respondent is likely to seek mental health treatment, using EDA, feature engineering, model comparison, hyperparameter tuning, and SHAP-based explainability.

## 📌 Overview

This project analyzes a public mental health survey dataset (~94K cleaned responses) to answer a practical question: **who is likely to seek mental health treatment, and what actually predicts it?** Beyond building an accurate classifier, the goal is to surface explainable, business-relevant drivers — the kind of insight an HR or wellness team could act on.

## 🗂️ Dataset

- **Source:** [Mental Health Dataset — Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset)
- **Size:** ~94,000 rows after cleaning and de-duplication, 15 features
- **Note:** the raw CSV isn't included in this repo (it exceeds GitHub's file size limits without Git LFS) — download it from the Kaggle link above and place it in the project root before running the notebook.
- **Target:** `treatment` — will this person seek mental health treatment? (binary, ~50/50 class balance)
- **Features:** demographics (`Gender`, `Country`, `self_employed`), workplace context (`care_options`, `mental_health_interview`), and self-reported wellbeing (`Growing_Stress`, `Mood_Swings`, `Coping_Struggles`, `Days_Indoors`, `Work_Interest`, `Social_Weakness`, etc.)

## 🔍 Workflow

1. **EDA & Preprocessing** — inspected structure and missingness, dropped the non-predictive `Timestamp` column, imputed missing values, removed duplicates, and label-encoded all categorical fields.
2. **Feature Engineering & Selection** — compared class balance across four candidate targets before committing to `treatment`, and used a correlation heatmap to check for redundant features.
3. **Train/Test Split & Scaling** — stratified 80/20 split to preserve class ratio, `RobustScaler` for outlier-resistant feature scaling.
4. **Class Imbalance Handling** — measured the imbalance ratio programmatically; applied `class_weight='balanced'` (SMOTE reserved for genuinely imbalanced cases).
5. **Model Comparison** — trained and evaluated Logistic Regression, Random Forest, and XGBoost on Accuracy, Precision, Recall, F1, and ROC-AUC.
6. **Hyperparameter Tuning** — `RandomizedSearchCV` with 5-fold stratified cross-validation on the top-performing model.
7. **Evaluation** — confusion matrix, ROC curve, and full classification report on a held-out test set.
8. **Explainability** — feature importances and SHAP values to explain *why* the model predicts what it predicts, not just what it predicts.

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost (tuned) | 0.80 | 0.79 | 0.80 | 0.79 | **0.92** |
| XGBoost (baseline) | 0.72 | 0.70 | 0.71 | 0.71 | 0.88 |
| Random Forest | 0.63 | 0.61 | 0.61 | 0.61 | 0.71 |
| Logistic Regression | 0.64 | 0.63 | 0.59 | 0.61 | 0.70 |

**Tuned XGBoost** was selected as the final model.

## 💡 Key Insights

- **Family history is the strongest predictor** of treatment-seeking behavior — consistent with greater awareness and lower stigma in affected families.
- **Workplace factors matter**: awareness of employer `care_options` and comfort with a `mental_health_interview` both carry real predictive signal — a lever organizations can actually influence.
- **Self-reported daily symptoms are weak predictors** on their own — `Growing_Stress`, `Mood_Swings`, and `Days_Indoors` rank far below access- and environment-related features.
- **Practical takeaway:** improving benefits communication and normalizing mental-health conversations at work is likely to move the needle more than trying to detect "who seems stressed."

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `SHAP` · `imbalanced-learn` · `Matplotlib` · `Seaborn` · `Jupyter`

## 🚀 Getting Started

```bash
# clone the repo
git clone <your-repo-url>
cd <repo-folder>

# install dependencies
pip install -r requirements.txt

# download the dataset from Kaggle and place the CSV in the project root
# https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset

# launch the notebook
jupyter notebook mental-health-dataset.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
imbalanced-learn
jupyter
```

## 📁 Project Structure

```
.
├── mental-health-dataset.ipynb   # full analysis: EDA → modeling → explainability
├── requirements.txt
└── README.md

# Mental_Health_Dataset.csv goes here too, once downloaded — not committed to the repo (see Dataset section)
```

## ⚠️ Limitations

- Self-reported survey data — subject to reporting bias.
- `family_history`, `care_options`, and `mental_health_interview` dominating feature importance is also consistent with survey-structure artifacts in this public dataset; validate against an external sample before using this for real decisions.

## Author

**Faaiz**
