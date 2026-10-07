"""Generate JSON data + embedded images for the XGBoost explainer HTML page."""
import json, re, base64, os
import numpy as np, pandas as pd, xgboost as xgb

ROOT = "/home/manit/Desktop/fun_projects/tsa"
OUT = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(f"{ROOT}/master_features.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
y = df["Volume"]; X = df.drop(columns=["Date", "Volume"]); m = y.notna()
X, y, dates = X[m], y[m], df["Date"][m]
split = len(X) - 90
Xtr, ytr = X.iloc[:split], y.iloc[:split]
Xte, yte, dte = X.iloc[split:], y.iloc[split:], dates.iloc[split:]

model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         reg_lambda=1.0, random_state=0)
model.fit(Xtr, ytr)
b = model.get_booster()
base = float(json.loads(b.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))

# pick one representative test row
i = 0
row = Xte.iloc[[i]]
actual = float(yte.iloc[i])
example_date = str(dte.iloc[i].date())

# per-tree leaf contribution for this row
leaf_idx = b.predict(xgb.DMatrix(row), pred_leaf=True)[0]
dumps = b.get_dump()
def leaf_val(dump, idx):
    for line in dump.split("\n"):
        mo = re.match(r"(\d+):leaf=([-0-9.eE+]+)", line.strip())
        if mo and int(mo.group(1)) == int(idx):
            return float(mo.group(2))
    return 0.0
contribs = [leaf_val(dumps[t], leaf_idx[t]) for t in range(len(dumps))]
running = base + np.cumsum(contribs)

# overall accuracy on test set
pred_all = model.predict(Xte)
mae = float(np.mean(np.abs(pred_all - yte)))
mape = float(np.mean(np.abs((pred_all - yte) / yte)) * 100)

# feature importance (gain)
imp = pd.Series(b.get_score(importance_type="gain")).sort_values(ascending=False)
imp_top = [{"feat": k, "gain": float(v)} for k, v in imp.head(12).items()]

# model stats
total_leaves = sum(d.count("leaf=") for d in dumps)

data = {
    "base": base,
    "actual": actual,
    "example_date": example_date,
    "final": float(running[-1]),
    "contribs": [float(c) for c in contribs],
    "running": [float(r) for r in running],
    "n_trees": len(dumps),
    "max_depth": 4,
    "learning_rate": 0.05,
    "mae": mae,
    "mape": mape,
    "total_leaves": total_leaves,
    "importance": imp_top,
    # the actual decision path of the single readable tree for this concept
}
with open(f"{OUT}/html_data.json", "w") as f:
    json.dump(data, f)

# embed images as base64
imgs = {}
for name in ["single_tree.png", "feature_importance.png", "shap_summary.png",
             "ensemble_tree_0.png", "ensemble_tree_1.png"]:
    p = f"{ROOT}/model_interpret/{name}"
    with open(p, "rb") as f:
        imgs[name] = "data:image/png;base64," + base64.b64encode(f.read()).decode()
with open(f"{OUT}/html_imgs.json", "w") as f:
    json.dump(imgs, f)

print("n_trees", data["n_trees"], "total_leaves", total_leaves)
print("base", round(base), "final", round(data["final"]), "actual", round(actual))
print("MAE", round(mae), "MAPE", round(mape, 2))
print("first contribs", [round(c, 1) for c in contribs[:6]])
print("wrote html_data.json + html_imgs.json")
