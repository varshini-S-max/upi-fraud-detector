"""Core logic: synthetic UPI data, model training, scoring, risk factors, cost analysis."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

TXN_TYPES = ["P2P", "P2M", "BILL", "COLLECT_REQUEST"]
FEATURES = [
    "amount", "hour", "txn_type", "is_new_receiver", "device_changed",
    "txns_last_1h", "account_age_days", "failed_pin_attempts",
]


def generate_data(n=50000, seed=42):
    """Create SYNTHETIC UPI transactions. Fraud labels follow hand-written risk rules."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "amount": np.round(rng.lognormal(mean=6.0, sigma=1.0, size=n), 2),
        "hour": rng.integers(0, 24, n),
        "txn_type": rng.choice(TXN_TYPES, n, p=[0.40, 0.40, 0.15, 0.05]),
        "is_new_receiver": rng.choice([0, 1], n, p=[0.80, 0.20]),
        "device_changed": rng.choice([0, 1], n, p=[0.95, 0.05]),
        "txns_last_1h": rng.poisson(0.5, n),
        "account_age_days": rng.integers(1, 2000, n),
        "failed_pin_attempts": rng.choice([0, 1, 2, 3], n, p=[0.93, 0.04, 0.02, 0.01]),
    })
    risk = (-7.5
            + 2.0 * (df["amount"] > 5000)
            + 1.8 * (df["hour"] <= 4)
            + 2.2 * df["is_new_receiver"]
            + 3.0 * df["device_changed"]
            + 0.9 * df["txns_last_1h"]
            + 1.8 * (df["account_age_days"] < 30)
            + 1.2 * df["failed_pin_attempts"]
            + 2.2 * (df["txn_type"] == "COLLECT_REQUEST"))
    prob = 1 / (1 + np.exp(-risk))
    df["is_fraud"] = (rng.random(n) < prob).astype(int)
    return df


def _encode(df, columns=None):
    X = pd.get_dummies(df[FEATURES], columns=["txn_type"])
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def metrics_at(y_true, p, threshold):
    """Confusion matrix + precision / recall / F1 at a given threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def business_cost(m, miss_cost, alarm_cost):
    """Total Rs cost = missed frauds x loss + false alarms x friction cost."""
    return m["fn"] * miss_cost + m["fp"] * alarm_cost


def best_cost_threshold(bundle, miss_cost, alarm_cost):
    """Scan thresholds and return (threshold, cost) with the lowest business cost."""
    best = None
    for thr in np.linspace(0.02, 0.98, 97):
        m = metrics_at(bundle["y_test"], bundle["p_test"], thr)
        c = business_cost(m, miss_cost, alarm_cost)
        if best is None or c < best[1]:
            best = (float(thr), float(c))
    return best


def train_model(df=None, seed=42):
    """Train a Random Forest and tune the decision threshold for best F1."""
    if df is None:
        df = generate_data(seed=seed)
    X = _encode(df)
    y = df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample",
        min_samples_leaf=3, n_jobs=-1, random_state=seed)
    model.fit(X_tr, y_tr)

    p = model.predict_proba(X_te)[:, 1]
    prec, rec, thr = precision_recall_curve(y_te, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[int(f1[:-1].argmax())])

    m = metrics_at(y_te, p, threshold)
    metrics = {
        "threshold": round(threshold, 3),
        "precision": round(m["precision"], 3),
        "recall": round(m["recall"], 3),
        "f1": round(m["f1"], 3),
        "pr_auc": round(float(average_precision_score(y_te, p)), 3),
        "baseline_accuracy": round(float(1 - y_te.mean()), 4),
        "accuracy": round(float((m["tp"] + m["tn"]) / len(y_te)), 4),
        "confusion": {k: m[k] for k in ("tn", "fp", "fn", "tp")},
        "fraud_rate": round(float(y.mean()), 4),
        "n_transactions": int(len(df)),
    }
    return {
        "model": model, "columns": list(X.columns), "threshold": threshold,
        "metrics": metrics, "y_test": y_te.to_numpy(), "p_test": p,
        "importances": pd.Series(model.feature_importances_, index=X.columns),
    }


def score_transaction(bundle, txn):
    """Return fraud risk probability (0-1) for one transaction (dict)."""
    row = _encode(pd.DataFrame([txn]), bundle["columns"])
    return float(bundle["model"].predict_proba(row)[0, 1])


def score_dataframe(bundle, df):
    """Score many transactions. Adds risk_score and flag columns."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    out = df.copy()
    X = _encode(out, bundle["columns"])
    out["risk_score"] = np.round(bundle["model"].predict_proba(X)[:, 1], 3)
    out["flag"] = np.where(out["risk_score"] >= bundle["threshold"], "SUSPICIOUS", "ok")
    return out


def risk_factors(txn):
    """Human-readable red flags found in a transaction (simple rule check)."""
    flags = []
    if txn["amount"] > 5000:
        flags.append("High amount (above Rs 5,000)")
    if txn["hour"] <= 4:
        flags.append("Late-night transaction (12 AM - 4 AM)")
    if txn["is_new_receiver"]:
        flags.append("Money sent to a new receiver")
    if txn["device_changed"]:
        flags.append("Login from a new / changed device")
    if txn["txns_last_1h"] >= 3:
        flags.append("Many transactions in the last hour")
    if txn["account_age_days"] < 30:
        flags.append("Very new account (under 30 days)")
    if txn["failed_pin_attempts"] >= 1:
        flags.append("Failed PIN attempts before this payment")
    if txn["txn_type"] == "COLLECT_REQUEST":
        flags.append("Collect request (common scam method)")
    return flags
