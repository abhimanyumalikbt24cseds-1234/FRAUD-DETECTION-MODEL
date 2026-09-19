# Financial Transaction Fraud Detection System

A machine learning system that identifies fraudulent credit card transactions using a
supervised classifier (Random Forest) and an unsupervised anomaly detector (Isolation
Forest), with an interactive Streamlit dashboard showing risk scores, suspicious
transactions, and key fraud indicators.

Built to match this resume line:
> *Developed a machine learning system to identify potentially fraudulent financial
> transactions using transaction and customer behavior features. Implemented feature
> engineering and classification/anomaly-detection techniques to identify unusual
> transaction patterns. Built an interactive dashboard displaying transaction risk scores,
> suspicious transactions, and key fraud indicators.*

## Files

| File | Purpose |
|---|---|
| `Fraud_Detection_System.ipynb` | **Start here.** Full walkthrough — run top to bottom in Google Colab. |
| `fraud_model.py` | Feature engineering, training (classifier + anomaly detector), evaluation, saves model + database. |
| `app.py` | Streamlit dashboard: risk scores, threshold explorer, suspicious transactions, feature importance. |
| `requirements.txt` | Pinned dependency versions. |
| `creditcard.csv` | **Not included here** — you provide this (see below). |

## The dataset

This project uses the real Kaggle "Credit Card Fraud Detection" dataset: 284,807 European
credit card transactions from September 2013, with only 492 (0.17%) confirmed fraud. The
28 main features (`V1`-`V28`) are already anonymized via PCA by the original data
publishers — `Time` and `Amount` are the only two features in their original form.

You already have this file (`creditcard.csv`) uploaded. If you need it again: it's
publicly available on Kaggle, search "Credit Card Fraud Detection Dataset."

## Quick start (Google Colab)

1. Upload `Fraud_Detection_System.ipynb` to Colab.
2. Upload `creditcard.csv` into the Colab session's file browser (folder icon, left
   sidebar) — it's too large to bundle into the notebook itself.
3. Run the cells top to bottom.
4. For the dashboard section, you'll need a free ngrok authtoken (same as the other
   project, if you already have one) — https://dashboard.ngrok.com/get-started/your-authtoken.

## Quick start (local machine)

```bash
pip install -r requirements.txt
python fraud_model.py          # trains models, builds fraud_transactions.db
streamlit run app.py
```

## Why these specific techniques

**Random Forest (the main classifier).** Handles the non-linear patterns in the PCA
features well, gives usable feature importances, and — with `class_weight="balanced"` —
handles the severe class imbalance without needing more complex resampling techniques like
SMOTE.

**Logistic Regression (the baseline).** Included specifically to show the imbalanced-data
trade-off: it catches more raw fraud (higher recall) but at a much lower precision than the
Random Forest, since a simple linear boundary can't separate the classes as cleanly.

**Isolation Forest (the anomaly detector).** Trained with zero access to fraud labels —
it only learns what "normal" transactions look like and flags whatever doesn't fit. This
matters because a purely supervised classifier can only ever catch fraud patterns similar
to what it was trained on; an anomaly detector is a genuine safety net for the unknown.

**Why not just report accuracy?** With only 0.17% fraud, a model that predicts "not fraud"
for every single transaction already scores 99.8% accuracy while catching zero fraud.
Precision, recall, and PR-AUC (precision-recall area under the curve) are the metrics that
actually reveal whether the model works — this is called out explicitly in the code
comments and notebook.

## Known limitations (be upfront about these in the interview)

- This public dataset has no customer/account ID, so genuine per-customer behavioral
  features aren't possible — a real bank system would join in account history to compare a
  transaction against *that specific customer's* normal pattern. We approximated with
  time-of-day and dataset-wide transaction-stream busyness instead.
- The `V1`-`V28` features are anonymized (PCA-transformed), so we can measure *how much*
  each one matters to the model, but not what it originally represented.
- This is a single, static dataset from a two-day window in September 2013 — a production
  system would need continuous retraining as fraud patterns evolve over time.
- The dashboard evaluates against a held-out test set (data the model never trained on) —
  this is the right practice for honest evaluation, but it means the numbers shown are
  from ~57,000 transactions, not the full 284,807.

## Extending it further

- Add SMOTE or other resampling techniques and compare against the current
  `class_weight="balanced"` approach.
- Try gradient boosting (XGBoost/LightGBM) and compare against the Random Forest.
- Add a proper time-based train/test split (train on earlier transactions, test on later
  ones) instead of a random split, to better simulate real deployment where you're always
  predicting on the future.
- If you get access to a dataset with account IDs, add real per-customer behavioral
  features: rolling average spend, typical transaction times, typical merchant categories.
