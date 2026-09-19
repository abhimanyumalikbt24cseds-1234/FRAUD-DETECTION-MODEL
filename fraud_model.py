"""
Financial Transaction Fraud Detection System — model training
----------------------------------------------------------------
Dataset: the real Kaggle "Credit Card Fraud Detection" dataset — 284,807
European credit card transactions from September 2013, 492 of them (0.17%)
confirmed fraud. Features V1-V28 are already anonymized via PCA (this is
real customer data, so the bank/Kaggle scrubbed the original fields before
release) — Time and Amount are the only two features in their original form.

This script:
  1. Loads the data and engineers a few extra features from what's available.
  2. Trains a supervised CLASSIFIER (Random Forest) — needs labeled fraud
     examples, learns exactly what past fraud looked like.
  3. Trains an unsupervised ANOMALY DETECTOR (Isolation Forest) — needs no
     labels at all, just flags transactions that look statistically unusual.
     This matters in the real world because new fraud patterns won't match
     anything the classifier was trained on; an anomaly detector can catch
     those "never seen before" cases the classifier would miss.
  4. Evaluates both properly for an imbalanced problem (precision, recall,
     F1, ROC-AUC, and PR-AUC — plain accuracy would be meaningless here,
     since predicting "not fraud" for everything already scores 99.8%).
  5. Saves the trained model + scaler, and writes the held-out test set
     (with risk scores attached) into a SQLite database for the dashboard.
"""

import sqlite3
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix,
)

CSV_PATH = "creditcard.csv"
DB_PATH = "fraud_transactions.db"
MODEL_PATH = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"
RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few features on top of the raw columns.

    Note: this public dataset has no customer/account ID, so true
    per-customer behavioral features (this resume bullet's "customer
    behavior features") aren't possible here — a real bank system would
    join in account history. We approximate what signal we can from what's
    actually available: time-of-day and how busy the overall transaction
    stream was around each transaction.
    """
    df = df.copy()
    df["hour_of_day"] = (df["Time"] % 86400) // 3600
    df["amount_log"] = np.log1p(df["Amount"])

    # Rolling count of transactions in the same hour-bucket, as a coarse
    # proxy for "unusually busy period" (a real system would do this per
    # account; here it's dataset-wide, which is an honest limitation).
    hour_bucket = (df["Time"] // 3600).astype(int)
    txns_per_hour = hour_bucket.value_counts()
    df["txns_in_hour_bucket"] = hour_bucket.map(txns_per_hour)

    return df


def load_and_prepare():
    df = pd.read_csv(CSV_PATH)
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c not in ("Class",)]
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    # Only scale Amount/Time-derived columns — the V1-V28 columns are
    # already PCA-transformed (mean ~0, unit-ish variance) by Kaggle.
    scale_cols = ["Time", "Amount", "hour_of_day", "amount_log", "txns_in_hour_bucket"]
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train, X_test, y_train, y_test, feature_cols, scaler


def train_classifier(X_train, y_train):
    # class_weight="balanced" tells the model to pay much more attention to
    # the rare fraud class instead of just learning to always predict "not
    # fraud" and being right 99.8% of the time by default.
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_baseline(X_train, y_train):
    """A simple, interpretable baseline to compare the Random Forest against."""
    base = LogisticRegression(class_weight="balanced", max_iter=2000)
    base.fit(X_train, y_train)
    return base


def train_anomaly_detector(X_train, y_train):
    # Isolation Forest never sees the labels — it only learns what "normal"
    # looks like from the (mostly non-fraud) data, then flags whatever
    # doesn't fit that pattern. contamination is set to the real fraud rate
    # so it flags a realistic proportion of transactions.
    contamination = y_train.mean()
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=RANDOM_STATE
    )
    iso.fit(X_train)
    return iso


def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["legit", "fraud"], digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_true, y_proba), 4))
    print("PR-AUC (average precision):", round(average_precision_score(y_true, y_proba), 4))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, y_pred))


def main():
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_cols, scaler = load_and_prepare()
    print(f"Train: {len(X_train):,} rows ({y_train.sum()} fraud) | "
          f"Test: {len(X_test):,} rows ({y_test.sum()} fraud)")

    print("\nTraining baseline (Logistic Regression)...")
    baseline = train_baseline(X_train, y_train)
    base_proba = baseline.predict_proba(X_test)[:, 1]
    base_pred = (base_proba >= 0.5).astype(int)
    evaluate("Logistic Regression (baseline)", y_test, base_pred, base_proba)

    print("\nTraining classifier (Random Forest)...")
    clf = train_classifier(X_train, y_train)
    clf_proba = clf.predict_proba(X_test)[:, 1]
    clf_pred = (clf_proba >= 0.5).astype(int)
    evaluate("Random Forest (main classifier)", y_test, clf_pred, clf_proba)

    print("\nTraining anomaly detector (Isolation Forest, unsupervised)...")
    iso = train_anomaly_detector(X_train, y_train)
    # IsolationForest scores: lower (more negative) = more anomalous.
    # decision_function gives a continuous score; predict gives -1/1.
    iso_raw_scores = -iso.decision_function(X_test)  # flip sign: higher = more suspicious
    iso_pred = (iso.predict(X_test) == -1).astype(int)
    # Normalize the raw scores to a 0-1 range so they're comparable to a probability.
    iso_proba = (iso_raw_scores - iso_raw_scores.min()) / (iso_raw_scores.max() - iso_raw_scores.min())
    evaluate("Isolation Forest (anomaly detector, no labels used)", y_test, iso_pred, iso_proba)

    print("\nSaving model artifacts...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, "feature_cols.pkl")
    joblib.dump(["Time", "Amount", "hour_of_day", "amount_log", "txns_in_hour_bucket"], "scale_cols.pkl")

    print("Writing held-out test set with risk scores to SQLite...")
    results = X_test.copy()
    # Unscale Time/Amount back to human-readable values for the dashboard
    results[["Time", "Amount", "hour_of_day", "amount_log", "txns_in_hour_bucket"]] = \
        scaler.inverse_transform(results[["Time", "Amount", "hour_of_day", "amount_log", "txns_in_hour_bucket"]])
    results["actual_class"] = y_test.values
    results["risk_score"] = (clf_proba * 100).round(2)
    results["anomaly_flag"] = iso_pred
    results["transaction_id"] = [f"TXN{100000+i}" for i in range(len(results))]
    results = results.reset_index(drop=True)

    conn = sqlite3.connect(DB_PATH)
    results.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(results):,} scored transactions -> {DB_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
