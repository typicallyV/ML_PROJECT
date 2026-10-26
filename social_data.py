"""Shared data layer for the Social Media Engagement Prediction project.
Used by the Colab notebook AND the Streamlit app, so both always see the same data/features.

Dataset : Facebook Metrics (UCI ML Repository, ID 368) - 500 posts of a cosmetics brand page.
Fallback: if the UCI download fails (no internet), a synthetic dataset with the SAME columns is generated.
"""
import io, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

URLS = [
    "https://archive.ics.uci.edu/static/public/368/facebook+metrics.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip",
]
BASE_DIR = Path(__file__).resolve().parent
LOCAL_FILE = BASE_DIR / "dataset_Facebook.csv"  # optional local UCI CSV for offline use
TYPES = ["Link", "Photo", "Status", "Video"]
NUM_FEATURES = ["Page total likes", "Category", "Post Month", "Post Weekday", "Post Hour", "Paid"]
KEEP = NUM_FEATURES + ["Type", "comment", "like", "share", "Total Interactions"]
WEEKDAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}


def _synthetic(n=500, seed=42):
    rng = np.random.default_rng(seed)
    likes = rng.integers(80000, 140000, n)
    ptype = rng.choice(TYPES, n, p=[0.06, 0.85, 0.06, 0.03])
    cat = rng.integers(1, 4, n)
    month = rng.integers(1, 13, n)
    wd = rng.integers(1, 8, n)
    hr = rng.integers(1, 24, n)
    paid = rng.binomial(1, 0.28, n).astype(float)
    mu = (4.4 + 0.5 * (ptype == "Status") + 0.35 * (ptype == "Video") - 0.25 * (ptype == "Link")
          + 0.3 * paid + 0.15 * (cat == 2) + 0.2 * np.isin(hr, [3, 10, 11, 12, 13])
          + 0.000004 * (likes - 100000) + 0.1 * np.isin(wd, [6, 7]))
    total = np.exp(rng.normal(mu, 0.85))
    like = np.rint(total * 0.82)
    comment = np.rint(total * 0.04)
    share = np.maximum(np.rint(total * 0.14), 0)
    df = pd.DataFrame({"Page total likes": likes, "Type": ptype, "Category": cat, "Post Month": month,
                       "Post Weekday": wd, "Post Hour": hr, "Paid": paid, "comment": comment,
                       "like": like, "share": share})
    df["Total Interactions"] = df["like"] + df["comment"] + df["share"]
    df.loc[rng.choice(n, 1), "Paid"] = np.nan          # mimic the real data's few missing values
    df.loc[rng.choice(n, 1), "like"] = np.nan
    df.loc[rng.choice(n, 4, replace=False), "share"] = np.nan
    return df


def load_raw(verbose=True):
    """Return (dataframe, source_description)."""
    def ok(d):
        return all(c in d.columns for c in KEEP)
    if LOCAL_FILE.exists():
        d = pd.read_csv(LOCAL_FILE, sep=";")
        if ok(d):
            return d[KEEP].copy(), "UCI Facebook Metrics (local file)"
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=25).read()))
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            d = pd.read_csv(z.open(name), sep=";")
            if ok(d):
                return d[KEEP].copy(), "UCI Facebook Metrics (downloaded)"
        except Exception as e:                         # try next URL
            if verbose:
                print("Download failed:", url, "->", type(e).__name__)
    return _synthetic(), "SYNTHETIC fallback (UCI download unavailable)"


def build_xy(df):
    """Clean + encode. Returns X (features), y_reg (Total Interactions), y_clf (1 = top-25% engagement), threshold."""
    d = df.copy()
    for c in ["Paid", "like", "share", "comment"]:
        d[c] = d[c].fillna(d[c].median())
    d = d.dropna(subset=["Total Interactions"]).reset_index(drop=True)
    X = d[NUM_FEATURES].copy()
    for c in NUM_FEATURES:
        X[c] = X[c].fillna(X[c].median())
    T = pd.get_dummies(d["Type"]).reindex(columns=TYPES, fill_value=0).astype(int)
    T.columns = ["Type_" + c for c in TYPES]
    X = pd.concat([X, T], axis=1)
    y_reg = d["Total Interactions"].astype(float)
    thr = float(y_reg.quantile(0.75))
    y_clf = (y_reg >= thr).astype(int)
    return X, y_reg, y_clf, thr


def make_row(columns, page_likes, post_type, category, month, weekday, hour, paid):
    """Build a one-row DataFrame for a NEW post, aligned to the training columns."""
    row = {"Page total likes": page_likes, "Category": category, "Post Month": month,
           "Post Weekday": weekday, "Post Hour": hour, "Paid": paid}
    for t in TYPES:
        row["Type_" + t] = int(post_type == t)
    return pd.DataFrame([row]).reindex(columns=columns, fill_value=0)




