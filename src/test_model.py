"""
Diagnostic script — run this to verify the model is loading correctly
and check predictions for a set of known URLs.

Usage (from project root, venv activated):
    python src/test_model.py
"""

import os
import sys
import joblib
import pandas as pd
from datetime import datetime

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from features import URL_FEATURES, extract_features

# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
model_path = os.path.join(BASE_DIR, 'models', 'xgb_model.pkl')
print("=" * 60)
print("MODEL DIAGNOSTICS")
print("=" * 60)
print(f"Model path   : {model_path}")
print(f"File exists  : {os.path.exists(model_path)}")
if os.path.exists(model_path):
    mtime = os.path.getmtime(model_path)
    print(f"Last saved   : {datetime.fromtimestamp(mtime)}")

model = joblib.load(model_path)
print(f"n_features_in: {model.n_features_in_}")
print(f"URL_FEATURES : {len(URL_FEATURES)}")
print(f"Match        : {model.n_features_in_ == len(URL_FEATURES)}")
print()

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
THRESHOLD = 0.75

test_cases = [
    # Legitimate — well-known sites
    ("https://facebook.com/",                               "legitimate"),
    ("https://github.com/",                                 "legitimate"),
    ("https://amazon.com/",                                 "legitimate"),
    ("https://www.bbc.co.uk/",                              "legitimate"),
    ("https://stackoverflow.com/",                          "legitimate"),
    ("https://instagram.com/",                              "legitimate"),
    ("https://microsoft.com/",                              "legitimate"),
    ("https://youtube.com/",                                "legitimate"),
    # Phishing — suspicious signals
    ("http://secure-paypal-login.xyz/verify",               "phishing"),
    ("http://192.168.1.1/admin/login",                      "phishing"),
    ("http://support-appleld.secureupdate.example.com/ap/", "phishing"),
]

print(f"{'URL':<52} {'PROBA':>6}  {'PRED':<12} {'EXPECTED':<12} {'OK?'}")
print("-" * 90)
for url, expected in test_cases:
    feats = extract_features(url)
    X     = pd.DataFrame([feats])
    proba = model.predict_proba(X)[0][1]
    pred  = 'phishing' if proba > THRESHOLD else 'legitimate'
    ok    = "✓" if pred == expected else "✗  <-- mismatch"
    print(f"{url[:51]:<52} {proba:>6.1%}  {pred:<12} {expected:<12} {ok}")

print()
print("Done.")
