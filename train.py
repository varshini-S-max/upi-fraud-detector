"""Train the model, print metrics, save charts. Run: python train.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from charts import confusion_fig, importance_fig, pr_fig
from fraud_detector import metrics_at, train_model

HERE = Path(__file__).parent

bundle = train_model()
m = bundle["metrics"]
print(json.dumps(m, indent=2))
(HERE / "metrics.json").write_text(json.dumps(m, indent=2))

at = metrics_at(bundle["y_test"], bundle["p_test"], bundle["threshold"])
for name, fig in [
    ("feature_importance.png", importance_fig(bundle["importances"])),
    ("confusion_matrix.png", confusion_fig(at, bundle["threshold"])),
    ("pr_curve.png", pr_fig(bundle["y_test"], bundle["p_test"], bundle["threshold"])),
]:
    fig.savefig(HERE / name, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
print("Saved metrics.json, feature_importance.png, confusion_matrix.png, pr_curve.png")
