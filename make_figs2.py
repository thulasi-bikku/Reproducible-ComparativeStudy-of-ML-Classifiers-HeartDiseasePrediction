import pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from hdlib import *
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
fig, ax = plt.subplots(figsize=(7.6, 3.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
W, H = 22, 15
xs = [1, 26, 51, 76]
def box(x, y_, t, fc="#e8eef6", ec=C1):
    ax.add_patch(FancyBboxPatch((x, y_), W, H, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1.1))
    ax.text(x + W / 2, y_ + H / 2, t, ha="center", va="center", fontsize=7.2, linespacing=1.35)
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#333333"))
top, bot = 32, 6
row1 = ["1. Data sources\nCleveland, n = 303\n(development);\nHungarian, n = 294\n(external only)",
        "2. Data audit\nexact and near-duplicate\ncheck, code\nharmonisation,\nmissing-value flags",
        "3. Preprocessing\n(training folds only)\nimputation, scaling,\none-hot encoding,\noptional IQR exclusion",
        "4. Repeated stratified\n10-fold CV\n10 repeats,\n100 train/validation\nfolds"]
row2 = ["8. Secondary analyses\nnested tuning, feature\nscenarios, sex-stratified\nresults, importance,\nexternal validation",
        "7. Statistical inference\ncorrected resampled\n$t$-test, Holm correction,\neffect sizes, 95% CIs",
        "6. Evaluation\nAUC from probabilities,\naccuracy, precision,\nrecall, specificity, F1,\nBrier score, calibration",
        "5. Eight classifiers\nLR, NB, SVM, KNN,\nDT, RF, GB, MLP"]
for i, t in enumerate(row1):
    box(xs[i], top, t, fc="#fdf2e0" if i == 2 else "#e8eef6", ec="#b36b00" if i == 2 else C1)
for i, t in enumerate(row2):
    box(xs[i], bot, t)
for i in range(3):
    arr(xs[i] + W + 0.4, top + H / 2, xs[i + 1] - 0.4, top + H / 2)
arr(xs[3] + W / 2, top - 0.4, xs[3] + W / 2, bot + H + 0.4)
for i in range(3, 0, -1):
    arr(xs[i] - 0.4, bot + H / 2, xs[i - 1] + W + 0.4, bot + H / 2)
fig.savefig("figs/workflow_framework.png", bbox_inches="tight"); plt.close(fig)

# replication-inflation figure
main = pickle.load(open("results/main.pkl", "rb")); misc = pickle.load(open("results/misc.pkl", "rb"))
rf = main["RF"][0]
rep_acc = rf.groupby("repeat").acc.mean(); rep_auc = rf.groupby("repeat").auc.mean()
sim = misc["dup_sim"]
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
for a, (k, lab) in zip(ax, [("acc", "Accuracy"), ("auc", "ROC--AUC")]):
    vals = [(rep_acc if k == "acc" else rep_auc).mean(), sim[k].mean()]
    errs = [(rep_acc if k == "acc" else rep_auc).std(ddof=1), sim[k].std(ddof=1)]
    a.bar(["303 unique\npatients", "1,025 records\n(722 repeated)"], vals, yerr=errs, color=[C1, C2], width=0.55, capsize=3)
    for i, v in enumerate(vals): a.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=8)
    a.set_ylim(0.5, 1.08); a.set_ylabel(lab); a.tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig("figs/duplication_effect.png", bbox_inches="tight"); plt.close(fig)
print(rep_acc.mean(), rep_acc.std(ddof=1), sim.acc.mean(), sim.acc.std(ddof=1))
