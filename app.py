"""UPI Fraud Detector: Streamlit app. Run: streamlit run app.py"""
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from charts import confusion_fig, importance_fig, pr_fig
from fraud_detector import (TXN_TYPES, best_cost_threshold, business_cost, metrics_at,
                            risk_factors, score_dataframe, score_transaction, train_model)

HERE = Path(__file__).parent
BANNER = HERE / "banner.jpg"
SAMPLE = HERE / "sample_transactions.csv"

st.set_page_config(page_title="UPI Fraud Detector", page_icon="🛡️")

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(0, 240, 255, 0.16), transparent 40%),
        radial-gradient(circle at 85% 75%, rgba(168, 85, 247, 0.20), transparent 40%),
        linear-gradient(160deg, #050816, #0b1026 50%, #120a2e);
    background-attachment: fixed;
}
.stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(0, 240, 255, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 240, 255, 0.06) 1px, transparent 1px);
    background-size: 44px 44px;
    animation: gridMove 12s linear infinite;
}
@keyframes gridMove { from { background-position: 0 0; } to { background-position: 44px 44px; } }
[data-testid="stHeader"] { background: transparent; }
.banner-wrap { text-align: center; margin-bottom: 8px; }
.banner-wrap img {
    width: 100%; max-width: 420px; border-radius: 18px;
    border: 1px solid rgba(0, 240, 255, 0.6);
    animation: pulseGlow 3.5s ease-in-out infinite alternate;
}
@keyframes pulseGlow {
    from { box-shadow: 0 0 15px rgba(0, 240, 255, 0.35), 0 0 40px rgba(168, 85, 247, 0.20); }
    to   { box-shadow: 0 0 35px rgba(0, 240, 255, 0.70), 0 0 80px rgba(168, 85, 247, 0.45); }
}
h1 {
    text-align: center;
    background: linear-gradient(90deg, #00f0ff, #a855f7, #ff4ecd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.tagline { text-align: center; color: #9fb3d9; margin-top: -10px; margin-bottom: 18px; }
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #7b2ff7); color: white; border: none;
    border-radius: 12px; padding: 10px 28px; font-weight: 700; transition: all 0.25s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 0 22px rgba(0, 198, 255, 0.8); color: white; }
.gauge-box { text-align: center; margin-top: 10px; }
.gauge {
    width: 170px; height: 170px; border-radius: 50%;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255, 255, 255, 0.10) 0);
    display: grid; place-items: center; margin: 0 auto 8px auto; box-shadow: 0 0 30px var(--c);
}
.gauge-inner {
    width: 132px; height: 132px; border-radius: 50%; background: #0b1026;
    display: grid; place-items: center; font-size: 2rem; font-weight: 800; color: var(--c);
}
.verdict { text-align: center; font-size: 1.3rem; font-weight: 700; margin: 6px 0 10px 0; }
.glass {
    background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 16px; padding: 16px 20px; backdrop-filter: blur(10px); margin-top: 14px;
}
.glass h4 { margin: 0 0 10px 0; }
.chip {
    display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600;
    background: rgba(255, 82, 82, 0.15); color: #ff8a8a; border: 1px solid #ff5252;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training model on synthetic data (first load only)...")
def load_bundle():
    return train_model()


PRESETS = {
    "Custom": dict(amount=1500, hour=12, txn_type="P2P", new=False, dev=False, n1h=0, age=365, pin=0),
    "Normal payment": dict(amount=250, hour=14, txn_type="P2M", new=False, dev=False, n1h=0, age=800, pin=0),
    "Suspicious payment": dict(amount=25000, hour=3, txn_type="COLLECT_REQUEST", new=True, dev=True,
                               n1h=4, age=10, pin=2),
}


def gauge(pct, color):
    return (f'<div class="gauge-box"><div class="gauge" style="--p:{pct};--c:{color}">'
            f'<div class="gauge-inner">{pct:g}%</div></div></div>')


bundle = load_bundle()

if BANNER.exists():
    img64 = base64.b64encode(BANNER.read_bytes()).decode()
    st.markdown(f'<div class="banner-wrap"><img src="data:image/jpeg;base64,{img64}" alt="AI"></div>',
                unsafe_allow_html=True)
st.title("UPI Fraud Detector")
st.markdown('<p class="tagline">Score UPI transactions for fraud risk, one at a time or in bulk.</p>',
            unsafe_allow_html=True)

tab_check, tab_batch, tab_insights = st.tabs(
    ["🔍 Check Transaction", "📂 Batch Scan", "📊 Model Insights"])

# ------------------------------------------------------------------ Tab 1
with tab_check:
    preset_name = st.radio("Load an example", list(PRESETS), horizontal=True)
    d = PRESETS[preset_name]
    k = preset_name  # new key per example, so inputs reset when the example changes

    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount (Rs)", 1.0, 1_000_000.0, float(d["amount"]), key=f"a{k}")
    hour = c2.slider("Hour of day (0-23)", 0, 23, d["hour"], key=f"h{k}")
    txn_type = c1.selectbox("Transaction type", TXN_TYPES, TXN_TYPES.index(d["txn_type"]), key=f"t{k}")
    n1h = c2.number_input("Transactions in last 1 hour", 0, 50, d["n1h"], key=f"n{k}")
    age = c1.number_input("Account age (days)", 1, 10000, d["age"], key=f"g{k}")
    pin = c2.selectbox("Failed PIN attempts", [0, 1, 2, 3], d["pin"], key=f"p{k}")
    new = c1.checkbox("New receiver (first time paying)", d["new"], key=f"r{k}")
    dev = c2.checkbox("Device changed recently", d["dev"], key=f"d{k}")

    if st.button("🛡️ Check transaction"):
        txn = dict(amount=amount, hour=hour, txn_type=txn_type, is_new_receiver=int(new),
                   device_changed=int(dev), txns_last_1h=int(n1h), account_age_days=int(age),
                   failed_pin_attempts=int(pin))
        score = score_transaction(bundle, txn)
        pct = round(score * 100, 1)
        suspicious = score >= bundle["threshold"]
        color = "#ff4d6d" if suspicious else "#2ee59d"
        verdict = ("🚨 SUSPICIOUS: block or ask for extra verification" if suspicious
                   else "✅ Looks normal")
        st.markdown(gauge(pct, color), unsafe_allow_html=True)
        st.markdown(f'<div class="verdict" style="color:{color}">{verdict}</div>',
                    unsafe_allow_html=True)
        flags = risk_factors(txn)
        chips = "".join(f'<span class="chip">{f}</span>' for f in flags) or "<i>No red flags found</i>"
        st.markdown(f'<div class="glass"><h4>⚠️ Risk factors</h4>{chips}</div>',
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ Tab 2
with tab_batch:
    st.write("Upload a CSV of transactions and get a risk score for every row.")
    st.caption("Required columns: amount, hour, txn_type, is_new_receiver, device_changed, "
               "txns_last_1h, account_age_days, failed_pin_attempts")
    uploaded = st.file_uploader("Upload transactions CSV", type="csv")
    use_sample = st.checkbox("Or use the included sample file (sample_transactions.csv)")

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample and SAMPLE.exists():
        data = pd.read_csv(SAMPLE)

    if data is not None:
        try:
            result = score_dataframe(bundle, data).sort_values("risk_score", ascending=False)
        except ValueError as err:
            st.error(str(err))
        else:
            flagged = int((result["flag"] == "SUSPICIOUS").sum())
            a, b, c = st.columns(3)
            a.metric("Transactions scanned", len(result))
            b.metric("Flagged suspicious", flagged)
            c.metric("Flag rate", f"{100 * flagged / len(result):.1f}%")
            st.dataframe(result)
            st.download_button("⬇️ Download results (CSV)", result.to_csv(index=False).encode("utf-8"),
                               file_name="fraud_scan_results.csv", mime="text/csv")

# ------------------------------------------------------------------ Tab 3
with tab_insights:
    m = bundle["metrics"]
    st.write(f"Trained on **{m['n_transactions']:,}** synthetic transactions "
             f"(fraud rate {m['fraud_rate'] * 100:.1f}%).")
    st.caption(f"A model that always says 'not fraud' scores {m['baseline_accuracy'] * 100:.1f}% accuracy "
               "and catches nothing. So precision, recall and F1 matter, not accuracy.")

    thr = st.slider("Decision threshold (flag if risk score is above this)", 0.05, 0.95,
                    float(round(bundle["threshold"], 2)), 0.01)
    at = metrics_at(bundle["y_test"], bundle["p_test"], thr)
    a, b, c = st.columns(3)
    a.metric("Precision", f"{at['precision']:.2f}")
    b.metric("Recall", f"{at['recall']:.2f}")
    c.metric("F1", f"{at['f1']:.2f}")

    left, right = st.columns(2)
    fig1 = confusion_fig(at, thr)
    left.pyplot(fig1)
    plt.close(fig1)
    fig2 = pr_fig(bundle["y_test"], bundle["p_test"], thr)
    right.pyplot(fig2)
    plt.close(fig2)

    st.markdown("#### 💰 Business cost view")
    st.caption("A missed fraud costs far more than annoying a genuine customer with a false alarm. "
               "Set the costs and see the best threshold.")
    k1, k2 = st.columns(2)
    miss_cost = k1.number_input("Loss per missed fraud (Rs)", 100, 1_000_000, 5000, step=100)
    alarm_cost = k2.number_input("Cost per false alarm (Rs)", 1, 100_000, 100, step=10)
    here = business_cost(at, miss_cost, alarm_cost)
    best_thr, best_cost = best_cost_threshold(bundle, miss_cost, alarm_cost)
    x, y_ = st.columns(2)
    x.metric("Cost at your threshold", f"Rs {here:,.0f}")
    y_.metric(f"Lowest cost (threshold {best_thr:.2f})", f"Rs {best_cost:,.0f}")

    st.markdown("#### 🔎 What drives the predictions?")
    fig3 = importance_fig(bundle["importances"])
    st.pyplot(fig3)
    plt.close(fig3)

st.caption("⚠️ Demo built on synthetic data. Not for real financial decisions.")
