import logging
import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the app works from any directory
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, 'src')

# Import shared feature extraction (same code used at train time)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from features import URL_FEATURES, extract_features  # noqa: E402

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
try:
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgb_model.pkl'))
except FileNotFoundError:
    raise RuntimeError(
        "Model files not found. Run 'python src/train.py' from the project root first."
    )

# Verify the saved feature list matches what the loaded model expects
assert len(URL_FEATURES) == model.n_features_in_, (
    f"Feature mismatch: model expects {model.n_features_in_} features, "
    f"but features.py defines {len(URL_FEATURES)}."
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Lower threshold biases towards catching phishing at the cost of more false
# positives. Adjust between 0.3–0.5 based on your acceptable false-positive rate.
PHISHING_THRESHOLD = 0.4

MAX_URL_LENGTH = 2048

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'predictions.log'),
    level=logging.INFO,
    format='%(message)s'
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)


def is_valid_url(url: str) -> bool:
    """Return True if the URL has a valid scheme and a non-empty hostname."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'xgb_model'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'Please enter a URL.'}), 400
    if len(url) > MAX_URL_LENGTH:
        return jsonify({'error': f'URL too long (max {MAX_URL_LENGTH} characters).'}), 400
    if not is_valid_url(url):
        return jsonify({'error': 'Invalid URL. Make sure it starts with http:// or https:// and has a valid domain.'}), 400

    features = extract_features(url)
    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0][1]
    label = 'phishing' if proba > PHISHING_THRESHOLD else 'legitimate'
    confidence = proba if label == 'phishing' else 1 - proba

    # Reasons are only meaningful for phishing verdicts.
    # For legitimate URLs we return an empty list — the front-end shows a
    # generic "no suspicious signals" message instead.
    reasons = []

    if label == 'phishing':
        # Top 3 features by importance × value, but only where the feature
        # value is actually non-zero (i.e. something suspicious was detected).
        importances = model.feature_importances_
        feature_values = X.iloc[0].to_dict()
        scores = {
            f: importances[i] * abs(feature_values[f])
            for i, f in enumerate(URL_FEATURES)
            if feature_values[f] != 0
        }
        top_features = sorted(scores, key=scores.get, reverse=True)[:3]

        # Human-readable labels for features
        FEATURE_LABELS = {
        'length_url':         'URL is unusually long',
        'length_hostname':    'Hostname is unusually long',
        'ip':                 'IP address used instead of domain name',
        'nb_dots':            'High number of dots in URL',
        'nb_hyphens':         'High number of hyphens in URL',
        'nb_at':              'Contains @ symbol',
        'nb_qm':              'Contains query parameters',
        'nb_percent':         'Contains encoded characters (%)',
        'nb_subdomains':      'Excessive number of subdomains',
        'abnormal_subdomain': 'Abnormal subdomain structure',
        'prefix_suffix':      'Hyphen in domain name',
        'shortening_service': 'URL shortening service detected',
        'phish_hints':        'Phishing keywords found (e.g. login, verify, secure)',
        'suspecious_tld':     'Suspicious top-level domain (e.g. .xyz, .top)',
        'login_form':         'Login-related keyword in URL',
        'http_in_path':       'HTTP found inside URL path',
        'https_token':        'Uses HTTPS',
        'punycode':           'Punycode/internationalised domain detected',
        'port':               'Non-standard port in URL',
        'tld_in_path':        'TLD found inside the URL path',
        'tld_in_subdomain':   'TLD found inside subdomain',
        'path_extension':     'Suspicious file extension in path (.php, .exe)',
        'nb_redirection':     'Multiple redirections in URL',
        'ratio_digits_url':   'High ratio of digits in URL',
        'ratio_digits_host':  'High ratio of digits in hostname',
        'char_repeat':        'Repeated characters detected',
        'longest_words_raw':  'Unusually long words in URL',
        'nb_slash':           'High number of slashes',
        'nb_eq':              'High number of equals signs',
        'nb_and':             'High number of ampersands',
        }
        reasons = [FEATURE_LABELS.get(f, f.replace('_', ' ').capitalize()) for f in top_features]

    logging.info(f"{datetime.now(timezone.utc).isoformat()} | {label} | {round(float(confidence) * 100, 1)}% | {url}")

    return jsonify({
        'label': label,
        'confidence': round(float(confidence) * 100, 1),
        'reasons': reasons
    })


if __name__ == '__main__':
    app.run(debug=False)
