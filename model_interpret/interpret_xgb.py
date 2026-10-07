"""
Interpretable XGBoost on the TSA master features.
Outputs tree diagrams + feature importance + SHAP so you can SEE what the model does.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

OUT = os.path.dirname(os.path.abspath(__file__))
CSV = "/home/manit/Desktop/fun_projects/tsa/master_features.csv"

df = pd.read_csv(CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
y = df["Volume"]
X = df.drop(columns=["Date", "Volume"])
# drop rows with no target
mask = y.notna()
X, y, dates = X[mask], y[mask], df["Date"][mask]

# time-based split: last 90 days as test
split = len(X) - 90
Xtr, Xte = X.iloc[:split], X.iloc[split:]
ytr, yte = y.iloc[:split], y.iloc[split:]

print(f"rows={len(X)}  features={X.shape[1]}  train={len(Xtr)}  test={len(Xte)}")

# -------------------------------------------------------------------
# 1) ONE shallow, fully readable tree (depth 3) — a single diagram
# -------------------------------------------------------------------
single = xgb.XGBRegressor(
    n_estimators=1, max_depth=3, learning_rate=1.0,
    base_score=float(ytr.mean()), tree_method="exact",
)
single.fit(Xtr, ytr)

# render that one tree big and readable
import graphviz  # noqa
g = xgb.to_graphviz(single, num_trees=0,
                    condition_node_params={"shape": "box"},
                    leaf_node_params={"shape": "box"})
g.render(filename=os.path.join(OUT, "single_tree"), format="png", cleanup=True)
print("wrote single_tree.png")

# text version too
booster = single.get_booster()
dump = booster.get_dump(with_stats=True)[0]
with open(os.path.join(OUT, "single_tree.txt"), "w") as f:
    f.write(dump)
print("wrote single_tree.txt")

# -------------------------------------------------------------------
# 2) REAL boosted ensemble
# -------------------------------------------------------------------
model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_lambda=1.0, random_state=0,
)
model.fit(Xtr, ytr)

pred = model.predict(Xte)
mae = np.mean(np.abs(pred - yte))
mape = np.mean(np.abs((pred - yte) / yte)) * 100
print(f"ensemble: n_trees={model.n_estimators}  test MAE={mae:,.0f}  MAPE={mape:.2f}%")

# diagrams of the first 2 individual trees in the ensemble
for t in (0, 1):
    gt = xgb.to_graphviz(model, num_trees=t)
    gt.render(filename=os.path.join(OUT, f"ensemble_tree_{t}"), format="png", cleanup=True)
    print(f"wrote ensemble_tree_{t}.png")

# feature importance (gain = how much each feature improved splits)
imp = (pd.Series(model.get_booster().get_score(importance_type="gain"))
       .sort_values(ascending=False))
imp.to_csv(os.path.join(OUT, "feature_importance_gain.csv"))
top = imp.head(15)[::-1]
plt.figure(figsize=(9, 6))
plt.barh(top.index, top.values, color="#2b7bba")
plt.title("XGBoost feature importance (gain) — top 15")
plt.xlabel("total gain")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=110)
print("wrote feature_importance.png")
print("\nTop 10 drivers by gain:")
print(imp.head(10).to_string())

# -------------------------------------------------------------------
# 3) SHAP — per-feature contribution to a prediction (optional)
# -------------------------------------------------------------------
try:
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xte)
    shap.summary_plot(sv, Xte, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=110, bbox_inches="tight")
    print("wrote shap_summary.png")
except Exception as e:
    print(f"(shap skipped: {e})")
