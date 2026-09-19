"""
Streamlit Dashboard — Financial Transaction Fraud Detection System
---------------------------------------------------------------------
Run locally:   streamlit run app.py
From Colab:    see the notebook's dashboard section (uses pyngrok)

Reads the already-trained model + the held-out, risk-scored test set that
fraud_model.py produces. Nothing here re-trains anything — this is a fast,
read-only dashboard on top of work already done, exactly like a real fraud
analyst's tool would be.
"""

import sqlite3
import joblib
import numpy as np
import pandas as pd
import streamlit as st

DB_PATH = "fraud_transactions.db"
MODEL_PATH = "fraud_model.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Financial Transaction Fraud Detection System")
st.caption("Random Forest classifier + Isolation Forest anomaly detector · scikit-learn · SQLite · Streamlit")


@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    return df


@st.cache_resource
def load_model():
    clf = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return clf, feature_cols


try:
    df = load_data()
    clf, feature_cols = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python fraud_model.py` first (or the "
        "matching notebook cells) to train the model and create the database."
    )
    st.stop()

tab_overview, tab_investigate = st.tabs(["📊 Overview", "🔍 Investigate Transactions"])

# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions (held-out test set)", f"{len(df):,}")
    col2.metric("Actual frauds", int(df["actual_class"].sum()))
    col3.metric("Fraud rate", f"{100 * df['actual_class'].mean():.3f}%")
    col4.metric("Avg. transaction amount", f"${df['Amount'].mean():.2f}")

    st.subheader("Risk score distribution: legit vs. actual fraud")
    st.caption(
        "A good model should push fraud (orange) toward high risk scores and "
        "legit transactions (blue) toward low ones. Overlap is where mistakes happen."
    )
    hist_data = pd.DataFrame({
        "risk_score": df["risk_score"],
        "label": df["actual_class"].map({0: "Legit", 1: "Fraud"}),
    })
    st.bar_chart(
        hist_data.assign(bucket=(hist_data["risk_score"] // 10 * 10))
        .groupby(["bucket", "label"]).size().unstack(fill_value=0)
    )

    st.subheader("Pick a risk-score threshold and see the trade-off live")
    threshold = st.slider(
        "Flag any transaction with risk score ≥ this as fraud",
        min_value=0.0, max_value=100.0, value=50.0, step=1.0,
    )
    flagged = df["risk_score"] >= threshold
    actual = df["actual_class"] == 1

    tp = int((flagged & actual).sum())
    fp = int((flagged & ~actual).sum())
    fn = int((~flagged & actual).sum())
    tn = int((~flagged & ~actual).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flagged as fraud", int(flagged.sum()))
    m2.metric("Precision", f"{precision:.1%}", help="Of what we flagged, how much was really fraud")
    m3.metric("Recall", f"{recall:.1%}", help="Of all real fraud, how much did we catch")
    m4.metric("Missed frauds", fn, help="Real fraud we did NOT flag at this threshold")

    st.caption(
        "Lower the threshold to catch more fraud (higher recall) at the cost of more "
        "false alarms (lower precision) — this trade-off is the central decision a "
        "bank has to make, and it's a business choice, not a purely technical one."
    )

    st.subheader("What the model actually looks at (feature importance)")
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False).head(12)
    st.bar_chart(top_features)
    st.caption(
        "V1-V28 are anonymized (PCA-transformed) features from the original bank "
        "data — we can see *how much* each one matters, but not what it originally "
        "represented, since the bank scrubbed that for privacy before releasing this data."
    )

# ---------------------------------------------------------------------------
# Investigate tab
# ---------------------------------------------------------------------------
with tab_investigate:
    st.subheader("Most suspicious transactions")
    n_show = st.slider("How many to show", 10, 200, 30)
    suspicious = df.sort_values("risk_score", ascending=False).head(n_show)
    display_cols = ["transaction_id", "Amount", "hour_of_day", "risk_score",
                     "anomaly_flag", "actual_class"]
    st.dataframe(
        suspicious[display_cols].rename(columns={
            "actual_class": "was_actually_fraud",
            "anomaly_flag": "flagged_as_anomaly",
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Where the classifier and the anomaly detector disagree")
    st.caption(
        "The Random Forest learned from past labeled fraud; the Isolation Forest "
        "never saw any labels and just flags statistically unusual transactions. "
        "When they disagree, it's often worth a second look — the anomaly detector "
        "might be catching a new pattern the classifier has never seen before."
    )
    disagree = df[(df["risk_score"] >= 50) != (df["anomaly_flag"] == 1)]
    st.dataframe(
        disagree[display_cols].rename(columns={
            "actual_class": "was_actually_fraud",
            "anomaly_flag": "flagged_as_anomaly",
        }).head(50),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Look up a specific transaction")
    txn_id = st.text_input("Transaction ID (e.g. TXN100005)")
    if txn_id:
        row = df[df["transaction_id"] == txn_id]
        if row.empty:
            st.warning("No transaction with that ID in this dataset.")
        else:
            st.write(row[display_cols].rename(columns={
                "actual_class": "was_actually_fraud",
                "anomaly_flag": "flagged_as_anomaly",
            }))
