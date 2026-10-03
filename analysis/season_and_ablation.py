"""Season-by-season stability and feature ablation for premier-league-match-prediction.

Questions the main notebook's shuffled-fold evaluation does not answer:
1. Do the engineered features hold up season by season, or does the signal decay?
2. Which feature groups carry the signal? (ablation: travel fatigue vs form vs venue splits)
3. How far above naive baselines (always-home-win, class priors) does the model get?

Data: epl_features_2324.csv (1,130 matches, 2023-24 through 2025-26).
Model here: multinomial logistic regression (scaled) on the 12 engineered features.
Metric: log-loss (proper scoring rule for probabilities) + accuracy.

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

SEED = 42
df = pd.read_csv("/home/hatch/workspace/expand-work/epl.csv", parse_dates=["Date"])
df["season"] = df["Date"].apply(
    lambda d: f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8 else f"{d.year-1}-{str(d.year)[-2:]}")
le = LabelEncoder()
y = le.fit_transform(df["FTR"])  # A=0, D=1, H=2

FEATS_ALL = ["H_Roll_GF_5", "H_Roll_SOT_5", "H_Home_GF_Form", "H_Home_SOT_Form",
             "H_Roll_Pts_5", "H_Travel_Fatigue_5", "A_Roll_GF_5", "A_Roll_SOT_5",
             "A_Away_GF_Form", "A_Away_SOT_Form", "A_Roll_Pts_5", "A_Travel_Fatigue_5"]
FEATS_NO_TRAVEL = [f for f in FEATS_ALL if "Travel_Fatigue" not in f]
FEATS_FORM_ONLY = [f for f in FEATS_ALL if "Roll" in f]
FEATS_VENUE_ONLY = ["H_Home_GF_Form", "H_Home_SOT_Form", "A_Away_GF_Form", "A_Away_SOT_Form"]

GROUPS = {"all features": FEATS_ALL, "no travel fatigue": FEATS_NO_TRAVEL,
          "rolling form only": FEATS_FORM_ONLY, "venue splits only": FEATS_VENUE_ONLY}

def make_clf():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=5000, random_state=SEED))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
out = {"seed": SEED, "n": len(df),
       "class_balance": {k: float(v) for k, v in
                         pd.Series(df["FTR"]).value_counts(normalize=True).items()},
       "per_season": {}, "ablation": {}}

# 1. Season-by-season 5-fold CV
seasons = sorted(df["season"].unique())
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, s in enumerate(seasons):
    d = df[df["season"] == s]
    ys = le.transform(d["FTR"])
    proba = cross_val_predict(make_clf(), d[FEATS_ALL], ys, cv=5, method="predict_proba")
    ll = log_loss(ys, proba)
    acc = accuracy_score(ys, proba.argmax(axis=1))
    out["per_season"][s] = {"n": len(d), "log_loss": float(ll), "accuracy": float(acc)}
    axes[0].bar(i, acc)
    axes[1].bar(i, ll)
axes[0].set_xticks(range(len(seasons))); axes[0].set_xticklabels(seasons)
axes[0].set_title("CV accuracy by season"); axes[0].set_ylabel("accuracy")
axes[0].axhline(0.432, color="k", linestyle="--", alpha=0.5, label="always-home baseline")
axes[0].legend(fontsize=9)
axes[1].set_xticks(range(len(seasons))); axes[1].set_xticklabels(seasons)
axes[1].set_title("CV log-loss by season (lower is better)"); axes[1].set_ylabel("log-loss")
fig.suptitle("Does the signal hold up season by season? (multinomial logistic, 5-fold CV)")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/expand-work/figs/epl_seasons.png", dpi=110)
plt.close(fig)

# 2. Time-based: train on first two seasons, test on 2025-26
tr = df[df["season"].isin(["2023-24", "2024-25"])]
te = df[df["season"] == "2025-26"]
clf = make_clf().fit(tr[FEATS_ALL], le.transform(tr["FTR"]))
proba = clf.predict_proba(te[FEATS_ALL])
yte = le.transform(te["FTR"])
out["forward_test"] = {"train_n": len(tr), "test_n": len(te),
                       "log_loss": float(log_loss(yte, proba)),
                       "accuracy": float(accuracy_score(yte, proba.argmax(axis=1)))}

# 3. Ablation on the full set
abl = {}
for name, feats in GROUPS.items():
    proba = cross_val_predict(make_clf(), df[feats], y, cv=cv, method="predict_proba")
    abl[name] = {"n_features": len(feats),
                 "log_loss": float(log_loss(y, proba)),
                 "accuracy": float(accuracy_score(y, proba.argmax(axis=1)))}
for bname, strat in [("always home win", "most_frequent"), ("class priors", "stratified")]:
    proba = cross_val_predict(DummyClassifier(strategy=strat, random_state=SEED),
                              df[FEATS_ALL], y, cv=cv, method="predict_proba")
    abl[bname] = {"n_features": 0,
                  "log_loss": float(log_loss(y, proba)),
                  "accuracy": float(accuracy_score(y, proba.argmax(axis=1)))}
out["ablation"] = abl

names = list(abl.keys())
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(names, [abl[n]["accuracy"] for n in names])
axes[0].set_title("5-fold CV accuracy by feature set")
axes[1].barh(names, [abl[n]["log_loss"] for n in names])
axes[1].set_title("5-fold CV log-loss by feature set (lower is better)")
fig.suptitle("Ablation: which engineered features carry the signal?")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/expand-work/figs/epl_ablation.png", dpi=110)
plt.close(fig)

with open("/home/hatch/workspace/expand-work/epl_metrics.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
