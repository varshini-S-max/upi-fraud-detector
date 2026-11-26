import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fraud_detector import (FEATURES, best_cost_threshold, business_cost, generate_data,
                            metrics_at, risk_factors, score_dataframe, score_transaction,
                            train_model)

NORMAL = dict(amount=250, hour=14, txn_type="P2M", is_new_receiver=0, device_changed=0,
              txns_last_1h=0, account_age_days=800, failed_pin_attempts=0)
SUSPICIOUS = dict(amount=25000, hour=3, txn_type="COLLECT_REQUEST", is_new_receiver=1,
                  device_changed=1, txns_last_1h=4, account_age_days=10, failed_pin_attempts=2)

_BUNDLE = {}


def bundle():
    if not _BUNDLE:
        _BUNDLE["b"] = train_model(generate_data(n=20000, seed=3), seed=3)
    return _BUNDLE["b"]


def test_generate_data_shape_and_fraud_rate():
    df = generate_data(n=5000, seed=1)
    assert len(df) == 5000
    assert 0.003 < df["is_fraud"].mean() < 0.08


def test_model_scores_suspicious_higher_than_normal():
    b = bundle()
    s_normal, s_susp = score_transaction(b, NORMAL), score_transaction(b, SUSPICIOUS)
    assert 0 <= s_normal <= 1 and 0 <= s_susp <= 1
    assert s_susp > s_normal


def test_risk_factors():
    assert risk_factors(NORMAL) == []
    assert len(risk_factors(SUSPICIOUS)) >= 6


def test_metrics_at_simple_case():
    m = metrics_at([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_business_cost_and_best_threshold():
    m = {"fn": 2, "fp": 10}
    assert business_cost(m, 5000, 100) == 11000
    thr, cost = best_cost_threshold(bundle(), 5000, 100)
    assert 0.02 <= thr <= 0.98 and cost >= 0


def test_score_dataframe_adds_columns():
    out = score_dataframe(bundle(), pd.DataFrame([NORMAL, SUSPICIOUS]))
    assert {"risk_score", "flag"} <= set(out.columns)
    assert out.loc[1, "risk_score"] > out.loc[0, "risk_score"]


def test_score_dataframe_missing_column_raises():
    try:
        score_dataframe(bundle(), pd.DataFrame([NORMAL]).drop(columns=["hour"]))
    except ValueError as e:
        assert "hour" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_sample_csv_matches_features():
    sample = Path(__file__).resolve().parent / "sample_transactions.csv"
    df = pd.read_csv(sample)
    assert list(df.columns) == FEATURES
