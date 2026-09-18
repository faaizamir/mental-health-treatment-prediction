"""
Trains the final mental-health-treatment model and saves everything the
Streamlit app needs to serve predictions: the fitted model, the scaler,
the per-column label encoders, and the feature order.

Run this once (or whenever the data/pipeline changes):
    python train_and_save_model.py
"""

import json
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "Mental Health Dataset.csv"
TARGET_COL = "treatment"
ARTIFACT_DIR = "artifacts"


def main():
    import os
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ---- Load & clean (mirrors the notebook) -----------------------------
    data = pd.read_csv(DATA_PATH)
    data.drop("Timestamp", axis=1, inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)
    data.drop_duplicates(inplace=True)

    # ---- Label-encode every categorical column, keep the encoders --------
    encoders = {}
    for col in data.columns:
        if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_string_dtype(data[col]):
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le

    feature_columns = [c for c in data.columns if c != TARGET_COL]

    X = data[feature_columns]
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    # ---- Class imbalance check (same logic as the notebook) --------------
    imbalance_ratio = y_train.value_counts(normalize=True).max()
    if imbalance_ratio > 0.60:
        from imblearn.over_sampling import SMOTE
        X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    # ---- Compare baseline models, pick the best by ROC-AUC ----------------
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, n_jobs=-1),
    }

    results = []
    for name, model in candidate_models.items():
        model.fit(X_train_res, y_train_res)
        pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        results.append({
            "model_name": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1_score": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        })
    best_model_name = max(results, key=lambda r: r["roc_auc"])["model_name"]
    print("Baseline comparison:", results)
    print("Best baseline model:", best_model_name)

    # ---- Hyperparameter tuning on the winner ------------------------------
    xgb_param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    rf_param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    tuning_space = {"XGBoost": xgb_param_dist, "Random Forest": rf_param_dist}

    if best_model_name in tuning_space:
        search = RandomizedSearchCV(
            estimator=candidate_models[best_model_name],
            param_distributions=tuning_space[best_model_name],
            n_iter=15,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=-1,
        )
    else:
        from sklearn.model_selection import GridSearchCV
        search = GridSearchCV(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            param_grid={"C": [0.01, 0.1, 1, 10, 100]},
            scoring="roc_auc", cv=5, n_jobs=-1,
        )

    search.fit(X_train_res, y_train_res)
    best_model = search.best_estimator_
    print("Best hyperparameters:", search.best_params_)

    # ---- Final evaluation on the held-out test set -------------------------
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics = {
        "model_name": best_model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print("Final tuned metrics:", metrics)

    # ---- Save every artifact the app needs ---------------------------------
    joblib.dump(best_model, f"{ARTIFACT_DIR}/model.joblib")
    joblib.dump(scaler, f"{ARTIFACT_DIR}/scaler.joblib")
    joblib.dump(encoders, f"{ARTIFACT_DIR}/encoders.joblib")
    with open(f"{ARTIFACT_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{ARTIFACT_DIR}/baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model, scaler, encoders, feature list, and metrics to ./{ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
