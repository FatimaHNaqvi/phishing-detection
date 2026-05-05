"""
Shared feature extraction module.

Used by both train.py (at train time) and app.py (at inference time) so that
the model always sees features computed by identical code.
"""

import ipaddress
import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Feature list — must stay in sync with what the model was trained on
# ---------------------------------------------------------------------------
URL_FEATURES = [
    'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens',
    'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore',
    'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon',
    'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www',
    'nb_com', 'nb_dslash', 'http_in_path', 'https_token',
    'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port',
    'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain',
    'nb_subdomains', 'prefix_suffix', 'shortening_service',
    'path_extension', 'nb_redirection', 'length_words_raw',
    'char_repeat', 'shortest_words_raw', 'shortest_word_host',
    'shortest_word_path', 'longest_words_raw', 'longest_word_host',
    'longest_word_path', 'avg_words_raw', 'avg_word_host',
    'avg_word_path', 'phish_hints', 'suspecious_tld', 'login_form',
]

PHISHING_KEYWORDS = [
    'login', 'secure', 'account', 'verify', 'paypal',
    'update', 'password', 'confirm', 'banking',
]

SHORTENING_SERVICES = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly']
SUSPICIOUS_TLDS     = ['.xyz', '.top', '.click', '.loan']
SUSPICIOUS_EXTS     = ['.php', '.html', '.exe']
COMMON_TLDS         = ['.com', '.org', '.net']


def _is_ip_address(host: str) -> int:
    """Return 1 if host is a bare IP address (v4 or v6), else 0."""
    host = host.split(':')[0].strip('[]')
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def _words(text: str) -> list:
    """Split text on non-word characters, discarding empty tokens."""
    return [w for w in re.split(r'\W+', text) if w]


def extract_features(url: str) -> dict:
    """
    Compute all URL_FEATURES from a raw URL string.
    Returns a dict keyed by feature name, ordered to match URL_FEATURES.
    """
    parsed  = urlparse(url)
    domain  = parsed.netloc.split(':')[0]   # hostname without port
    path    = parsed.path
    url_words    = _words(url)
    domain_parts = [p for p in domain.split('.') if p]
    path_words   = _words(path)

    features = {
        'length_url':         len(url),
        'length_hostname':    len(domain),
        'ip':                 _is_ip_address(domain),
        'nb_dots':            url.count('.'),
        'nb_hyphens':         url.count('-'),
        'nb_at':              url.count('@'),
        'nb_qm':              url.count('?'),
        'nb_and':             url.count('&'),
        'nb_or':              url.count('|'),
        'nb_eq':              url.count('='),
        'nb_underscore':      url.count('_'),
        'nb_tilde':           url.count('~'),
        'nb_percent':         url.count('%'),
        'nb_slash':           url.count('/'),
        'nb_star':            url.count('*'),
        'nb_colon':           url.count(':'),
        'nb_comma':           url.count(','),
        'nb_semicolumn':      url.count(';'),
        'nb_dollar':          url.count('$'),
        'nb_space':           url.count(' '),
        'nb_www':             url.lower().count('www'),
        'nb_com':             url.lower().count('.com'),
        'nb_dslash':          url.count('//'),
        'http_in_path':       1 if 'http' in path.lower() else 0,
        'https_token':        1 if parsed.scheme == 'https' else 0,
        'ratio_digits_url':   sum(c.isdigit() for c in url) / len(url) if url else 0,
        'ratio_digits_host':  sum(c.isdigit() for c in domain) / len(domain) if domain else 0,
        'punycode':           1 if 'xn--' in url else 0,
        'port':               1 if parsed.port else 0,
        'tld_in_path':        1 if any(t in path for t in COMMON_TLDS) else 0,
        'tld_in_subdomain':   1 if len(domain_parts) > 2 and any(t.lstrip('.') in domain_parts[0] for t in COMMON_TLDS) else 0,
        'abnormal_subdomain': 1 if len(domain_parts) > 3 else 0,
        'nb_subdomains':      max(len(domain_parts) - 2, 0),
        'prefix_suffix':      1 if '-' in domain else 0,
        'shortening_service': 1 if any(s in url for s in SHORTENING_SERVICES) else 0,
        'path_extension':     1 if any(path.endswith(e) for e in SUSPICIOUS_EXTS) else 0,
        'nb_redirection':     url.count('//') - 1 if url.count('//') > 1 else 0,
        'length_words_raw':   len(url_words),
        'char_repeat':        max((url.count(c) for c in set(url)), default=0),
        'shortest_words_raw': min((len(w) for w in url_words), default=0),
        'shortest_word_host': min((len(p) for p in domain_parts), default=0),
        'shortest_word_path': min((len(w) for w in path_words), default=0),
        'longest_words_raw':  max((len(w) for w in url_words), default=0),
        'longest_word_host':  max((len(p) for p in domain_parts), default=0),
        'longest_word_path':  max((len(w) for w in path_words), default=0),
        'avg_words_raw':      sum(len(w) for w in url_words) / len(url_words) if url_words else 0,
        'avg_word_host':      sum(len(p) for p in domain_parts) / len(domain_parts) if domain_parts else 0,
        'avg_word_path':      sum(len(w) for w in path_words) / len(path_words) if path_words else 0,
        'phish_hints':        sum(1 for kw in PHISHING_KEYWORDS if kw in url.lower()),
        'suspecious_tld':     1 if any(url.lower().endswith(t) for t in SUSPICIOUS_TLDS) else 0,
        'login_form':         1 if 'login' in url.lower() else 0,
    }

    return {k: features[k] for k in URL_FEATURES}
