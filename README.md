# Phishing URL Detector

A machine learning web application that classifies URLs as **phishing** or **legitimate** using features extracted purely from the URL string — no external lookups required.

---

## Overview

This project trains three classifiers (Logistic Regression, Random Forest, and XGBoost) on a labelled dataset of URLs and serves the best-performing model (XGBoost) via a lightweight Flask web interface. Users paste a URL into the browser, and the app returns a verdict, a confidence score, and the top three reasons behind the prediction.

---

## Features

- URL-only feature extraction (50 features derived from the URL string alone)
- Three trained models: Logistic Regression, Random Forest, XGBoost
- Flask REST API with a `/predict` endpoint returning JSON
- Human-readable explanations for each prediction
- Prediction logging to `logs/predictions.log`
- Simple, dependency-free front-end (plain HTML + JS)

---

## Project Structure

```
phishing-detection/
├── app/
│   ├── app.py                  # Flask application and feature extraction
│   └── templates/
│       └── index.html          # Front-end UI
├── data/
│   └── dataset_phishing.csv    # Labelled URL dataset (87 features + status)
├── models/
│   ├── rf_model.pkl            # Trained Random Forest model
│   ├── xgb_model.pkl           # Trained XGBoost model (used in production)
│   └── url_features.pkl        # Ordered feature list saved at train time
├── notebooks/                  # Jupyter notebooks (exploratory analysis)
├── src/
│   └── train.py                # Model training and evaluation script
├── venv/                       # Python virtual environment (not committed)
├── README.md
└── SUGGESTED_CHANGES.md
```

---

## Requirements

- Python 3.9+
- See [requirements.txt](#setup) — key dependencies:
  - `flask`
  - `scikit-learn`
  - `xgboost`
  - `pandas`
  - `joblib`

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd phishing-detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install flask scikit-learn xgboost pandas joblib
```

> **Tip:** Once a `requirements.txt` is added (see [Suggested Changes](SUGGESTED_CHANGES.md)), use `pip install -r requirements.txt` instead.

---

## Training the Models

Run the training script from the **project root**:

```bash
python src/train.py
```

This will:
1. Load `data/dataset_phishing.csv`
2. Train Logistic Regression, Random Forest, and XGBoost on an 80/20 split
3. Print classification reports for each model
4. Save `models/xgb_model.pkl`, `models/rf_model.pkl`, and `models/url_features.pkl`

Pre-trained models are already included in `models/`, so this step is optional unless you want to retrain.

---

## Running the Web App

Run the Flask app from the **project root**:

```bash
python app/app.py
```

Then open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

Enter a URL (including `http://` or `https://`) and click **Check URL**.

### Example URLs to try

| URL | Expected result |
|-----|----------------|
| `https://www.bbc.co.uk` | Legitimate |
| `http://secure-paypal-login.xyz/verify` | Phishing |
| `http://192.168.1.1/admin` | Phishing |
| `https://support-appleld.com.secureupdate.example.com/ap/login` | Phishing |

---

## API Reference

### `POST /predict`

**Request body (JSON):**
```json
{ "url": "http://example.com/path" }
```

**Response (JSON):**
```json
{
  "label": "phishing",
  "confidence": 87.4,
  "reasons": [
    "Phishing keywords found (e.g. login, verify, secure)",
    "Suspicious top-level domain (e.g. .xyz, .top)",
    "Abnormal subdomain structure"
  ]
}
```

**Error response:**
```json
{ "error": "Invalid URL. Make sure it starts with http:// or https:// and has a valid domain." }
```

---

## How It Works

Feature extraction is performed entirely from the URL string using `urllib.parse` and regular expressions. The 50 features used include:

- **Length-based:** `length_url`, `length_hostname`
- **Character counts:** `nb_dots`, `nb_hyphens`, `nb_at`, `nb_slash`, etc.
- **Structural signals:** `nb_subdomains`, `abnormal_subdomain`, `prefix_suffix`, `port`
- **Content signals:** `phish_hints`, `suspecious_tld`, `login_form`, `shortening_service`
- **Encoding signals:** `punycode`, `http_in_path`, `https_token`, `path_extension`

The XGBoost model uses a classification threshold of **0.4** (rather than the default 0.5), biasing slightly towards catching more phishing URLs at the cost of a small increase in false positives.

---

## Prediction Logging

All predictions are appended to `logs/predictions.log` in the format:

```
2026-05-05T10:23:41.123456 | phishing | 87.4% | http://secure-paypal-login.xyz/verify
```

---

## Dataset

The dataset (`dataset_phishing.csv`) contains 87 features per URL and a `status` column (`legitimate` / `phishing`). Only the 50 URL-computable features are used for training; the remaining 37 features require external lookups (WHOIS, page rank, DNS records, etc.) and are excluded to keep inference fast and self-contained.

---

## Licence

MIT
