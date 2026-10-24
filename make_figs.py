import pickle, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from hdlib import *

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300})
X, y, raw = load_cleveland()
main = pickle.load(open("results/main.pkl", "rb"))
ex = pickle.load(open("results/explain.pkl", "rb"))
NAMES = {"LR": "Logistic\nregression", "NB": "Naive\nBayes", "SVM": "SVM\n(RBF)", "KNN": "KNN\n($k$=7)",
         "DT": "Decision\ntree", "RF": "Random\nforest", "GB": "Gradient\nboosting", "NN": "Neural\nnetwork"}
NAMES1 = {k: v.replace("\n", " ") for k, v in NAMES.items()}
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"

# ---------------- Fig 1: workflow ----------------
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
def box(x, y_, w, h, text, fc="#e8eef6", ec=C1, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y_ + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#333333"))
box(2, 46, 28, 12, "UCI Cleveland cohort\n$n$ = 303 patients, 13 predictors\n(development data)", bold=False)
box(2, 28, 28, 12, "UCI Hungarian cohort\n$n$ = 294 patients\n(external validation only)")
box(37, 46, 28, 12, "Data audit\nexact / near-duplicate check,\ncode harmonisation, missing flags")
box(37, 28, 28, 12, "Repeated stratified 10-fold CV\n(10 repeats, 100 folds)", fc="#fdf2e0", ec="#b36b00")
box(70, 46, 28, 12, "Inside each training fold only:\nimputation, scaling, encoding,\n(optional) IQR exclusion", fc="#fdf2e0", ec="#b36b00")
box(70, 28, 28, 12, "Eight classifiers\nLR, NB, SVM, KNN, DT,\nRF, GB, MLP")
box(2, 8, 28, 14, "Evaluation on held-out folds\nAUC (from probabilities), accuracy,\nprecision, recall, specificity, F1,\nBrier score, calibration slope")
box(37, 8, 28, 14, "Inference\ncorrected resampled $t$-test,\nHolm correction, effect sizes,\n95% confidence intervals")
box(70, 8, 28, 14, "Secondary analyses\nnested tuning, feature scenarios,\nsex-stratified results, importance,\nexternal validation")
arrow(30, 52, 37, 52); arrow(65, 52, 70, 52)
arrow(84, 46, 84, 40); arrow(70, 34, 65, 34)
arrow(51, 46, 51, 40)
arrow(37, 32, 16, 22); arrow(30, 34, 37, 34)
arrow(30, 16, 37, 16); arrow(65, 16, 70, 16)
arrow(16, 28, 16, 22)
fig.savefig("figs/workflow_framework.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 2: descriptive ----------------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
cnt = pd.Series(y).value_counts().sort_index()
ax[0].bar(["Absent", "Present"], cnt.values, color=[C3, C2], width=0.6)
for i, v in enumerate(cnt.values):
    ax[0].text(i, v + 3, f"{v}\n({100*v/len(y):.1f}%)", ha="center", va="bottom", fontsize=8)
ax[0].set_ylim(0, 215); ax[0].set_ylabel("Patients"); ax[0].set_title("(a) Disease status", fontsize=9)
bins = np.arange(25, 82, 5)
ax[1].hist([X.age[y == 0], X.age[y == 1]], bins=bins, color=[C3, C2], label=["Absent", "Present"], stacked=False)
ax[1].set_xlabel("Age (years)"); ax[1].set_ylabel("Patients"); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("(b) Age by disease status", fontsize=9)
sx = pd.crosstab(X.sex, y)
ax[2].bar(["Female", "Male"], sx.loc[[0, 1], 0], color=C3, width=0.6, label="Absent")
ax[2].bar(["Female", "Male"], sx.loc[[0, 1], 1], bottom=sx.loc[[0, 1], 0], color=C2, width=0.6, label="Present")
ax[2].set_ylabel("Patients"); ax[2].set_title("(c) Sex and disease status", fontsize=9); ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figs/class_age_sex.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 3: correlation heatmap ----------------
cm = ex["corr"].copy()
lab = {"age": "age", "trestbps": "trestbps", "chol": "chol", "thalach": "thalach", "oldpeak": "oldpeak", "ca": "ca", "num": "disease"}
cm = cm.rename(index=lab, columns=lab)
fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_yticks(range(len(cm)))
ax.set_xticklabels(cm.columns, rotation=45, ha="right"); ax.set_yticklabels(cm.index)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(cm.values[i, j]) > 0.6 else "black")
for s in ax.spines.values(): s.set_visible(False)
fig.colorbar(im, fraction=0.046, pad=0.04, label="Pearson $r$")
fig.savefig("figs/correlation_heatmap.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 5: per-fold AUC distribution ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.0))
order = ex["rank_auc"]
data = [main[m][0].auc.values for m in order]
bp = ax.boxplot(data, labels=[NAMES1[m] for m in order], patch_artist=True, widths=0.55,
                medianprops=dict(color="black"), flierprops=dict(markersize=2.5))
for b in bp["boxes"]: b.set_facecolor("#cfdcec"); b.set_edgecolor(C1)
ax.set_ylabel("Per-fold ROC--AUC"); ax.set_ylim(0.35, 1.03)
plt.setp(ax.get_xticklabels(), fontsize=8)
fig.tight_layout(); fig.savefig("figs/auc_distribution.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 6: confusion matrices (pooled out-of-fold, repeat 1) ----------------
fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.0))
for a, m in zip(axs.ravel(), MODELS):
    p = main[m][1][0]
    pred = (p >= 0.5).astype(int)
    cmx = np.array([[((y == 0) & (pred == 0)).sum(), ((y == 0) & (pred == 1)).sum()],
                    [((y == 1) & (pred == 0)).sum(), ((y == 1) & (pred == 1)).sum()]])
    a.imshow(cmx, cmap="Blues", vmin=0, vmax=170)
    for i in range(2):
        for j in range(2):
            a.text(j, i, str(cmx[i, j]), ha="center", va="center", fontsize=10,
                   color="white" if cmx[i, j] > 100 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["Abs.", "Pres."], fontsize=7)
    a.set_yticklabels(["Abs.", "Pres."], fontsize=7, rotation=90, va="center")
    a.set_title(NAMES1[m], fontsize=8)
    for s in a.spines.values(): s.set_visible(False)
for a in axs[:, 0]: a.set_ylabel("True class", fontsize=8)
for a in axs[1]: a.set_xlabel("Predicted class", fontsize=8)
fig.tight_layout(); fig.savefig("figs/confusion_matrices.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 7: calibration curves ----------------
fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.0), sharex=True, sharey=True)
for a, m in zip(axs.ravel(), MODELS):
    c = ex["calibration"][m]
    a.plot([0, 1], [0, 1], "--", color="#999999", lw=0.9)
    a.plot(c["curve"][0], c["curve"][1], "o-", color=C1, ms=3.5, lw=1.1)
    a.set_title(NAMES1[m], fontsize=8)
    a.text(0.04, 0.80, f"slope {c['slope']:.2f}\nBrier {c['brier']:.3f}", fontsize=7, transform=a.transAxes)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
for a in axs[:, 0]: a.set_ylabel("Observed fraction", fontsize=8)
for a in axs[1]: a.set_xlabel("Predicted probability", fontsize=8)
fig.tight_layout(); fig.savefig("figs/calibration_curves.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 8: importance ----------------
def barpanel(a, d, title, color):
    items = sorted(d.items(), key=lambda t: t[1])
    a.barh([k for k, _ in items], [v for _, v in items], color=color, height=0.65)
    a.set_title(title, fontsize=8.5); a.tick_params(labelsize=8)
fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.6))
rf = ex["importance"]["RF"]; lr = ex["importance"]["LR"]
barpanel(axs[0, 0], ex["mdi"], "(a) RF: mean decrease in impurity", C3)
barpanel(axs[0, 1], rf["perm_mean"], "(b) RF: permutation (AUC drop)", C1)
barpanel(axs[0, 2], rf["shap_abs"], "(c) RF: mean |Shapley value|", C2)
barpanel(axs[1, 0], lr["perm_mean"], "(d) LR: permutation (AUC drop)", C1)
barpanel(axs[1, 1], lr["shap_abs"], "(e) LR: mean |Shapley value|", C2)
axs[1, 2].axis("off")
fig.tight_layout(); fig.savefig("figs/feature_importance.png", bbox_inches="tight"); plt.close(fig)
print("figs 1-3,5-8 ok")
