"""
Train phishing detection models.

Features are extracted from raw URLs using the same extract_features() function
that app.py uses at inference time — ensuring training and prediction are consistent.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Allow importing from src/ regardless of working directory
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import URL_FEATURES, extract_features  # noqa: E402

DATA_PATH  = os.path.join(BASE_DIR, 'data', 'dataset_phishing.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply extract_features() to every URL row and return a DataFrame."""
    print("Extracting features from URLs — this may take a minute...")
    rows = []
    for i, url in enumerate(df['url'], 1):
        try:
            rows.append(extract_features(str(url)))
        except Exception:
            # Skip malformed URLs; their index matches df, so align later
            rows.append({k: 0 for k in URL_FEATURES})
        if i % 5000 == 0:
            print(f"  {i}/{len(df)} URLs processed")
    return pd.DataFrame(rows, columns=URL_FEATURES)


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    X = build_feature_matrix(df)
    y = df['status'].map({'legitimate': 0, 'phishing': 1})

    # Drop rows where the URL was malformed and status couldn't be mapped
    valid = y.notna()
    X, y = X[valid], y[valid].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples")
    print(f"Using {len(URL_FEATURES)} URL-computable features\n")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=['legitimate', 'phishing']))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(models['XGBoost'],       os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    joblib.dump(models['Random Forest'], os.path.join(MODELS_DIR, 'rf_model.pkl'))
    joblib.dump(URL_FEATURES,            os.path.join(MODELS_DIR, 'url_features.pkl'))
    print("Models and feature list saved.")


if __name__ == '__main__':
    main()
