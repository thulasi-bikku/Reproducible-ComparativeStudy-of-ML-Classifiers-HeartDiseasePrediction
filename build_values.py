"""Computes every number and LaTeX table used in the manuscript from results/*.pkl."""
import pickle, hashlib, platform, sys
import numpy as np, pandas as pd
from scipy import stats
from hdlib import *

main = pickle.load(open("results/main.pkl", "rb"))
var = pickle.load(open("results/variants.pkl", "rb"))
misc = pickle.load(open("results/misc.pkl", "rb"))
ex = pickle.load(open("results/explain.pkl", "rb"))
try:
    nest = pickle.load(open("results/nested.pkl", "rb"))
except Exception:
    nest = {}
X, y, raw = load_cleveland()
Xh, yh, rawh = load_hungarian()

FULL = {"LR": "Logistic regression", "NB": "Naive Bayes", "SVM": "SVM (RBF kernel)", "KNN": "$k$-nearest neighbours ($k=7$)",
        "DT": "Decision tree", "RF": "Random forest", "GB": "Gradient boosting", "NN": "Neural network (MLP)"}
SHORT = {"LR": "LR", "NB": "NB", "SVM": "SVM", "KNN": "KNN", "DT": "DT", "RF": "RF", "GB": "GB", "NN": "MLP"}
BEST = ex["best"]
ORDER = MODELS


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def ms(m, s, d=3): return f"{m:.{d}f} $\\pm$ {s:.{d}f}"
def pct(x, d=1): return f"{100 * x:.{d}f}"
def pfmt(p):
    if p < 0.001: return "$<0.001$"
    return f"{p:.3f}"
def ci(lo, hi, d=3): return f"[{lo:.{d}f}, {hi:.{d}f}]"
def sgn(x):
    t = f"{x:+.3f}"
    return "0.000" if t in ("+0.000", "-0.000") else t
def bold_if(s, cond): return f"\\textbf{{{s}}}" if cond else s

V = {}
T = {}

# ---------------- dataset ----------------
n = len(y)
V["n"] = n
V["n_pos"] = int(y.sum()); V["n_neg"] = int((1 - y).sum())
V["prev"] = pct(y.mean())
V["prev_neg"] = pct(1 - y.mean())
V["n_male"] = int(X.sex.sum()); V["n_female"] = int(n - X.sex.sum())
V["pct_male"] = pct(X.sex.mean()); V["pct_female"] = pct(1 - X.sex.mean())
V["age_mean"] = f"{X.age.mean():.1f}"; V["age_sd"] = f"{X.age.std():.1f}"
V["age_min"] = int(X.age.min()); V["age_max"] = int(X.age.max())
V["miss_ca"] = int(X.ca.isna().sum()); V["miss_thal"] = int(X.thal.isna().sum())
V["n_enc"] = ex["n_encoded"]
V["hn"] = len(yh); V["hpos"] = int(yh.sum()); V["hprev"] = pct(yh.mean())
V["hmale"] = int(Xh.sex.sum()); V["hpct_male"] = pct(Xh.sex.mean())
V["hage_mean"] = f"{Xh.age.mean():.1f}"; V["hage_sd"] = f"{Xh.age.std():.1f}"
V["n_outl"] = misc["iqr_outliers_total"]; V["pct_outl"] = pct(misc["iqr_outliers_total"] / n)
V["outl_trestbps"] = misc["iqr_outliers_by_col"]["trestbps"]; V["outl_chol"] = misc["iqr_outliers_by_col"]["chol"]
V["outl_thalach"] = misc["iqr_outliers_by_col"]["thalach"]; V["outl_oldpeak"] = misc["iqr_outliers_by_col"]["oldpeak"]
V["n_exact_dup"] = misc["n_exact_dups"]; V["n_near_dup"] = misc["n_near_dup_pairs"]
sim = misc["dup_sim"]
V["sim_acc"] = f"{sim.acc.mean():.3f}"; V["sim_acc_sd"] = f"{sim.acc.std(ddof=1):.3f}"
V["sim_auc"] = f"{sim.auc.mean():.3f}"
rf_rep = main["RF"][0].groupby("repeat").acc.mean()
V["rf_acc_unique"] = f"{rf_rep.mean():.3f}"
V["sim_acc_min"] = f"{sim.acc.min():.3f}"; V["sim_acc_max"] = f"{sim.acc.max():.3f}"

# ---------------- main performance ----------------
agg = ex["agg"]
for m in MODELS:
    for k in ["acc", "prec", "rec", "spec", "f1", "auc", "brier"]:
        V[f"{k}_{m}"] = f"{agg[m][k][0]:.3f}"
        V[f"{k}sd_{m}"] = f"{agg[m][k][1]:.3f}"
V["best"] = SHORT[BEST]; V["best_full"] = FULL[BEST]
rank = ex["rank_auc"]
V["rank2"] = SHORT[rank[1]]; V["rank3"] = SHORT[rank[2]]; V["rank4"] = SHORT[rank[3]]
V["worst"] = SHORT[rank[-1]]

rows = []
for m in rank:
    best_auc = (m == BEST)
    r = [FULL[m]]
    for k in ["acc", "prec", "rec", "spec", "f1", "auc", "brier"]:
        s = ms(*agg[m][k])
        r.append(bold_if(s, best_auc and k != "brier") if best_auc else s)
    rows.append(" & ".join(r) + " \\\\")
T["cv"] = "\n".join(rows)

# ---------------- statistical comparison ----------------
cdf = ex["comparison"].set_index("model")
rows = []
for m in [x for x in rank if x != BEST]:
    c = cdf.loc[m]
    rows.append(" & ".join([FULL[m],
        f"{c.auc_diff:.3f} {ci(c.auc_lo, c.auc_hi)}", f"{c.auc_dz:.2f}", pfmt(c.auc_p), pfmt(c.auc_p_holm),
        f"{c.acc_diff:.3f} {ci(c.acc_lo, c.acc_hi)}", f"{c.acc_dz:.2f}", pfmt(c.acc_p_holm)]) + " \\\\")
T["stats"] = "\n".join(rows)
V["friedman_chi"] = f"{ex['friedman'][0]:.1f}"
V["friedman_p"] = "$<0.001$" if ex["friedman"][1] < 0.001 else f"{ex['friedman'][1]:.3f}"
sig_auc = [SHORT[m] for m in cdf.index if cdf.loc[m, "auc_p_holm"] < 0.05]
V["sig_auc_models"] = ", ".join(sig_auc)
nonsig_auc = [SHORT[m] for m in rank if m != BEST and m not in [k for k in cdf.index if cdf.loc[k, "auc_p_holm"] < 0.05]]
V["nonsig_auc_models"] = ", ".join(nonsig_auc)
for m in cdf.index:
    V[f"dauc_{m}"] = f"{cdf.loc[m, 'auc_diff']:.3f}"
    V[f"dauc_lo_{m}"] = f"{cdf.loc[m, 'auc_lo']:.3f}"; V[f"dauc_hi_{m}"] = f"{cdf.loc[m, 'auc_hi']:.3f}"
    V[f"pholm_auc_{m}"] = pfmt(cdf.loc[m, "auc_p_holm"])
    V[f"dz_auc_{m}"] = f"{cdf.loc[m, 'auc_dz']:.2f}"
    V[f"pholm_acc_{m}"] = pfmt(cdf.loc[m, "acc_p_holm"])
    V[f"dacc_{m}"] = f"{cdf.loc[m, 'acc_diff']:.3f}"
    V[f"dacc_lo_{m}"] = f"{cdf.loc[m, 'acc_lo']:.3f}"; V[f"dacc_hi_{m}"] = f"{cdf.loc[m, 'acc_hi']:.3f}"
V["wil_RF"] = pfmt(ex["wilcoxon_naive"]["RF"]) if "RF" in ex["wilcoxon_naive"] else ""
V["n_wil_sig"] = sum(1 for v in ex["wilcoxon_naive"].values() if v < 0.05)

# ---------------- MLP seeds & convergence ----------------
s = misc["mlp_seeds"]
V["mlp_auc_mean"] = f"{s.auc.mean():.3f}"; V["mlp_auc_sd"] = f"{s.auc.std(ddof=1):.3f}"
V["mlp_auc_min"] = f"{s.auc.min():.3f}"; V["mlp_auc_max"] = f"{s.auc.max():.3f}"
V["mlp_acc_mean"] = f"{s.acc.mean():.3f}"; V["mlp_acc_sd"] = f"{s.acc.std(ddof=1):.3f}"
V["mlp_acc_min"] = f"{s.acc.min():.3f}"; V["mlp_acc_max"] = f"{s.acc.max():.3f}"
V["mlp_frac_limit"] = pct(ex["nn_iter"]["frac_at_limit"], 0)

# ---------------- nested tuning ----------------
rows = []
if nest and all(m in nest for m in MODELS):
    for m in MODELS:
        d0 = main[m][0][main[m][0].repeat < 2].reset_index(drop=True)
        d1 = nest[m][0].reset_index(drop=True)
        diff = d1.auc.values - d0.auc.values
        ratio = (d0.n_test / d0.n_train).mean()
        mean, se, t, p, c = corrected_resampled_t(diff, 1.0, ratio)
        dacc = d1.acc.values - d0.acc.values
        rows.append((m, d0.auc.mean(), d0.auc.std(ddof=1), d1.auc.mean(), d1.auc.std(ddof=1), mean, c, p,
                     d0.acc.mean(), d1.acc.mean(), dacc.mean()))
    T["nested"] = "\n".join(" & ".join([FULL[m], ms(a, b), ms(c, d), f"{sgn(e)} {ci(*cc)}", pfmt(p),
                                          f3(a0), f3(a1)]) + " \\\\" for m, a, b, c, d, e, cc, p, a0, a1, _ in rows)
    nd = pd.DataFrame([dict(m=m, d0=a, d1=c, diff=e, p=p) for m, a, b, c, d, e, cc, p, a0, a1, _ in rows]).set_index("m")
    V["nest_best"] = SHORT[nd.d1.idxmax()]; V["nest_best_auc"] = f"{nd.d1.max():.3f}"
    V["nest_max_gain"] = sgn(nd['diff'].max()); V["nest_max_gain_model"] = SHORT[nd["diff"].idxmax()]
    V["nest_min_gain"] = sgn(nd['diff'].min()); V["nest_min_gain_model"] = SHORT[nd["diff"].idxmin()]
    V["nest_n_sig"] = int((nd.p < 0.05).sum())
    V["nest_sig_models"] = ", ".join(SHORT[m] for m in nd.index[nd.p < 0.05])
    for m in MODELS:
        V[f"nest_{m}"] = f"{nd.loc[m,'d1']:.3f}"; V[f"nestd_{m}"] = sgn(nd.loc[m,'diff'])
        V[f"nestdef_{m}"] = f"{nd.loc[m,'d0']:.3f}"
    # range of tuned AUC among models
    V["nest_range"] = f"{nd.d1.min():.3f}--{nd.d1.max():.3f}"
    V["nest_nodt_range"] = f"{nd.d1.drop('DT').min():.3f}--{nd.d1.max():.3f}"

# ---------------- outlier sensitivity ----------------
rows = []
for m in MODELS:
    a = main[m][0]; g = var[("outlier", "global", m)][0]; i = var[("outlier", "infold", m)][0]
    rows.append(" & ".join([FULL[m], f3(a.auc.mean()), f3(i.auc.mean()), f3(g.auc.mean()),
                            f3(a.acc.mean()), f3(i.acc.mean()), f3(g.acc.mean())]) + " \\\\")
T["outlier"] = "\n".join(rows)
V["n_global_removed"] = var[("outlier", "global", "LR")][2]
V["pct_global_removed"] = pct(var[("outlier", "global", "LR")][2] / n)
dd = [abs(var[("outlier", "global", m)][0].auc.mean() - main[m][0].auc.mean()) for m in MODELS]
V["outl_max_dauc"] = f"{max(dd):.3f}"
di = [abs(var[("outlier", "infold", m)][0].auc.mean() - main[m][0].auc.mean()) for m in MODELS]
V["outl_infold_max_dauc"] = f"{max(di):.3f}"
da = [abs(var[("outlier", "global", m)][0].acc.mean() - main[m][0].acc.mean()) for m in MODELS]
V["outl_max_dacc"] = f"{max(da):.3f}"

# ---------------- feature scenarios ----------------
def scen(m, key):
    if key == "F13_all": return main[m][0]
    return var[("fs", key, m)][0]
rows = []
for m in MODELS:
    rows.append(" & ".join([FULL[m]] + [ms(scen(m, k).auc.mean(), scen(m, k).auc.std(ddof=1)) for k in
                           ["F13_all", "F12_no_ca", "F11_no_ca_thal", "F7_resting"]]) + " \\\\")
T["scen"] = "\n".join(rows)
for k, tag in [("F13_all", "f13"), ("F12_no_ca", "f12"), ("F11_no_ca_thal", "f11"), ("F7_resting", "f7"), ("F10_common", "f10")]:
    a = {m: scen(m, k).auc.mean() for m in MODELS}
    bm = max(a, key=a.get)
    V[f"scen_{tag}_best"] = SHORT[bm]; V[f"scen_{tag}_best_auc"] = f"{a[bm]:.3f}"
    V[f"scen_{tag}_min"] = f"{min(a.values()):.3f}"; V[f"scen_{tag}_max"] = f"{max(a.values()):.3f}"
    for m in MODELS: V[f"scen_{tag}_{m}"] = f"{a[m]:.3f}"
    acc_ = {m: scen(m, k).acc.mean() for m in MODELS}
    V[f"scen_{tag}_acc_LR"] = f"{acc_['LR']:.3f}"
    V[f"scen_{tag}_sens_LR"] = f"{scen('LR', k).rec.mean():.3f}"
    V[f"scen_{tag}_spec_LR"] = f"{scen('LR', k).spec.mean():.3f}"
# mean loss in AUC (all models) when removing ca / ca+thal / resting-only
def meanauc(k): return np.mean([scen(m, k).auc.mean() for m in MODELS])
V["scen_mean_f13"] = f"{meanauc('F13_all'):.3f}"
V["scen_drop_f12"] = f"{meanauc('F13_all') - meanauc('F12_no_ca'):.3f}"
V["scen_drop_f11"] = f"{meanauc('F13_all') - meanauc('F11_no_ca_thal'):.3f}"
V["scen_drop_f7"] = f"{meanauc('F13_all') - meanauc('F7_resting'):.3f}"
V["scen_mean_f7"] = f"{meanauc('F7_resting'):.3f}"
V["scen_mean_f11"] = f"{meanauc('F11_no_ca_thal'):.3f}"

# ---------------- calibration ----------------
cal = ex["calibration"]
rows = []
for m in rank:
    c = cal[m]
    rows.append(" & ".join([FULL[m], ms(c["brier"], c["brier_sd"]), f2(c["slope"]), f"{c['intercept']:+.2f}"]) + " \\\\")
T["cal"] = "\n".join(rows)
for m in MODELS:
    V[f"slope_{m}"] = f2(cal[m]["slope"]); V[f"icpt_{m}"] = f"{cal[m]['intercept']:+.2f}"
    V[f"brier_{m}"] = f3(cal[m]["brier"])

# ---------------- sex-stratified ----------------
fair = ex["fairness"]
rows = []
def one(m, sx):
    f = fair[m][sx]; e, lo, hi = f["est"], f["lo"], f["hi"]
    prev = ex["sex_prev"][sx]
    return " & ".join([SHORT[m], sx, f"{f['n']}", f"{pct(prev, 0)}\\%",
                       f"{e[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}]", f"{e[1]:.2f} [{lo[1]:.2f}, {hi[1]:.2f}]",
                       f"{e[2]:.2f} [{lo[2]:.2f}, {hi[2]:.2f}]", f"{e[3]:.2f}", f"{e[4]:.3f}", f"{f['slope']:.2f}"]) + " \\\\"
fair_models = [BEST, "RF", "GB"] if BEST != "RF" else ["RF", "LR", "GB"]
for m in fair_models:
    for sx in ["Male", "Female"]:
        rows.append(one(m, sx))
    rows.append("\\midrule")
T["fair"] = "\n".join(rows[:-1])
V["n_f"] = fair[BEST]["Female"]["n"]; V["npos_f"] = fair[BEST]["Female"]["n_pos"]
V["n_m"] = fair[BEST]["Male"]["n"]; V["npos_m"] = fair[BEST]["Male"]["n_pos"]
V["prev_f"] = pct(ex["sex_prev"]["Female"], 1); V["prev_m"] = pct(ex["sex_prev"]["Male"], 1)
for m in MODELS:
    for sx, tg in [("Male", "m"), ("Female", "f")]:
        f = fair[m][sx]
        V[f"fa_auc_{tg}_{m}"] = f"{f['est'][0]:.2f}"; V[f"fa_sens_{tg}_{m}"] = f"{f['est'][1]:.2f}"
        V[f"fa_spec_{tg}_{m}"] = f"{f['est'][2]:.2f}"; V[f"fa_fnr_{tg}_{m}"] = f"{f['est'][3]:.2f}"
        V[f"fa_sens_lo_{tg}_{m}"] = f"{f['lo'][1]:.2f}"; V[f"fa_sens_hi_{tg}_{m}"] = f"{f['hi'][1]:.2f}"
        V[f"fa_spec_lo_{tg}_{m}"] = f"{f['lo'][2]:.2f}"; V[f"fa_spec_hi_{tg}_{m}"] = f"{f['hi'][2]:.2f}"
        V[f"fa_auc_lo_{tg}_{m}"] = f"{f['lo'][0]:.2f}"; V[f"fa_auc_hi_{tg}_{m}"] = f"{f['hi'][0]:.2f}"
        V[f"fa_slope_{tg}_{m}"] = f"{f['slope']:.2f}"

# ---------------- external validation ----------------
e = misc["external"]
rows = []
for m in MODELS:
    a = e[(e.fs == "F10_common") & (e.model == m)].iloc[0]
    b = e[(e.fs == "F7_resting") & (e.model == m)].iloc[0]
    inter = scen(m, "F10_common").auc.mean()
    rows.append(" & ".join([FULL[m], f3(inter), f"{a.auc:.3f} [{a.auc_lo:.3f}, {a.auc_hi:.3f}]", f3(a.acc), f3(a.rec), f3(a.spec), f3(a.brier),
                            f"{b.auc:.3f} [{b.auc_lo:.3f}, {b.auc_hi:.3f}]"]) + " \\\\")
T["ext"] = "\n".join(rows)
ea = e[e.fs == "F10_common"].set_index("model"); eb = e[e.fs == "F7_resting"].set_index("model")
V["ext_best"] = SHORT[ea.auc.idxmax()]; V["ext_best_auc"] = f3(ea.auc.max())
V["ext_min"] = f3(ea.auc.min()); V["ext_min_model"] = SHORT[ea.auc.idxmin()]
V["ext_min_nb_dt"] = f3(ea.drop(["NB", "DT"]).auc.min()); V["ext_max_nb_dt"] = f3(ea.drop(["NB", "DT"]).auc.max())
for m in MODELS:
    V[f"ext_{m}"] = f3(ea.loc[m, "auc"]); V[f"ext7_{m}"] = f3(eb.loc[m, "auc"])
    V[f"ext_acc_{m}"] = f3(ea.loc[m, "acc"]); V[f"ext_sens_{m}"] = f3(ea.loc[m, "rec"]); V[f"ext_spec_{m}"] = f3(ea.loc[m, "spec"])
    V[f"ext_lo_{m}"] = f3(ea.loc[m, "auc_lo"]); V[f"ext_hi_{m}"] = f3(ea.loc[m, "auc_hi"])
V["ext7_min"] = f3(eb.auc.min()); V["ext7_max"] = f3(eb.auc.max())
V["ext7_min_nbdt"] = f3(eb.drop(["NB", "DT"]).auc.min()); V["ext7_max_nbdt"] = f3(eb.drop(["NB", "DT"]).auc.max())
V["h_miss_chol"] = int(Xh.chol.isna().sum()); V["h_miss_slope"] = int(Xh.slope.isna().sum())
V["h_miss_ca"] = int(Xh.ca.isna().sum()); V["h_miss_thal"] = int(Xh.thal.isna().sum())
gap = {m: scen(m, "F10_common").auc.mean() - ea.loc[m, "auc"] for m in MODELS}
V["ext_gap_LR"] = f3(gap["LR"]); V["ext_gap_RF"] = f3(gap["RF"])
V["ext_gap_min"] = f3(min(gap.values())); V["ext_gap_max"] = f3(max(gap.values()))

# ---------------- importance ----------------
imp = ex["importance"]; mdi = ex["mdi"]
feats_sorted = sorted(ALL_FEATS, key=lambda f: -imp["RF"]["shap_abs"][f])
rows = []
for f in feats_sorted:
    rows.append(" & ".join([f"\\texttt{{{f}}}", f3(mdi[f]), f"{imp['RF']['perm_mean'][f]:.3f}", f3(imp["RF"]["shap_abs"][f]),
                            f"{imp['LR']['perm_mean'][f]:.3f}", f3(imp["LR"]["shap_abs"][f])]) + " \\\\")
T["imp"] = "\n".join(rows)
def top(d, k=3): return [f for f, _ in sorted(d.items(), key=lambda t: -t[1])[:k]]
V["mdi_top3"] = ", ".join(f"\\texttt{{{f}}}" for f in top(mdi))
V["rfperm_top3"] = ", ".join(f"\\texttt{{{f}}}" for f in top(imp["RF"]["perm_mean"]))
V["rfshap_top3"] = ", ".join(f"\\texttt{{{f}}}" for f in top(imp["RF"]["shap_abs"]))
V["lrperm_top3"] = ", ".join(f"\\texttt{{{f}}}" for f in top(imp["LR"]["perm_mean"]))
V["lrshap_top3"] = ", ".join(f"\\texttt{{{f}}}" for f in top(imp["LR"]["shap_abs"]))
V["mdi_ca_rank"] = sorted(mdi, key=lambda f: -mdi[f]).index("ca") + 1
V["mdi_thalach_rank"] = sorted(mdi, key=lambda f: -mdi[f]).index("thalach") + 1
V["rfperm_ca"] = f"{imp['RF']['perm_mean']['ca']:.3f}"; V["rfperm_thal"] = f"{imp['RF']['perm_mean']['thal']:.3f}"
V["lrperm_ca"] = f"{imp['LR']['perm_mean']['ca']:.3f}"
V["rfperm_thalach"] = f"{imp['RF']['perm_mean']['thalach']:.3f}"
V["rfperm_chol"] = f"{imp['RF']['perm_mean']['chol']:.3f}"
V["rfperm_trestbps"] = f"{imp['RF']['perm_mean']['trestbps']:.3f}"
V["mdi_chol"] = f3(mdi["chol"]); V["mdi_trestbps"] = f3(mdi["trestbps"])
V["mdi_thalach"] = f3(mdi["thalach"])

# ---------------- collinearity ----------------
a, b, r_ = ex["max_abs_r"]
V["max_r"] = f"{r_:.2f}"; V["max_r_pair"] = f"\\texttt{{{a}}}--\\texttt{{{b}}}"
vif = ex["vif"]
vm = max(vif, key=vif.get)
V["vif_max"] = f"{vif[vm]:.1f}"; V["vif_max_name"] = vm.replace("_", "\\_")
V["vif_cont_max"] = f"{max(vif[c] for c in ['age','trestbps','chol','thalach','oldpeak']):.1f}"
V["n_vif_gt5"] = sum(1 for v in vif.values() if v > 5)
V["corr_oldpeak_slope"] = ""  # not used

# ---------------- runtime ----------------
rt = ex["runtime"]
T["runtime"] = "\n".join(" & ".join([FULL[m], f"{1000*rt[m]['fit_mean']:.1f} $\\pm$ {1000*rt[m]['fit_sd']:.1f}",
                                      f"{rt[m]['pred_mean']:.2f} $\\pm$ {rt[m]['pred_sd']:.2f}"]) + " \\\\" for m in MODELS)
V["rt_max_fit_ms"] = f"{1000*max(r['fit_mean'] for r in rt.values()):.0f}"
V["rt_max_pred"] = f"{max(r['pred_mean'] for r in rt.values()):.2f}"

# ---------------- environment ----------------
import sklearn, scipy, matplotlib
V["py"] = platform.python_version(); V["skl"] = sklearn.__version__; V["np"] = np.__version__
V["pd"] = pd.__version__; V["sp"] = scipy.__version__; V["mpl"] = matplotlib.__version__
cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
V["cpu"] = cpu.replace("Processor @", "processor at").replace("  ", " ")
for fn, tag in [("cleveland.csv", "sha_cle"), ("hungarian.csv", "sha_hun")]:
    V[tag] = hashlib.sha256(open(DATA + fn, "rb").read()).hexdigest()

# ---------------- descriptive tables ----------------
def grp(df, yy, col):
    return int((df[col] == 1).sum())
T["desc"] = None
# Table I rows (Cleveland vs Hungarian)
def desc_row(label, a, b): return f"{label} & {a} & {b} \\\\"
rows = [
    desc_row("Patients", n, len(yh)),
    desc_row("Disease present (angiographic stenosis $>50\\%$ in any major vessel)", f"{V['n_pos']} ({V['prev']}\\%)", f"{V['hpos']} ({V['hprev']}\\%)"),
    desc_row("Disease absent", f"{V['n_neg']} ({V['prev_neg']}\\%)", f"{len(yh)-V['hpos']} ({pct(1-yh.mean())}\\%)"),
    desc_row("Male sex", f"{V['n_male']} ({V['pct_male']}\\%)", f"{V['hmale']} ({V['hpct_male']}\\%)"),
    desc_row("Age, years (mean $\\pm$ SD)", f"{V['age_mean']} $\\pm$ {V['age_sd']}", f"{V['hage_mean']} $\\pm$ {V['hage_sd']}"),
    desc_row("Missing values, \\texttt{ca}", V["miss_ca"], V["h_miss_ca"]),
    desc_row("Missing values, \\texttt{thal}", V["miss_thal"], V["h_miss_thal"]),
    desc_row("Missing values, \\texttt{slope}", int(X.slope.isna().sum()), V["h_miss_slope"]),
    desc_row("Missing values, \\texttt{chol}", int(X.chol.isna().sum()), V["h_miss_chol"]),
    desc_row("Exact duplicate records", V["n_exact_dup"], int(Xh.assign(y=yh).duplicated().sum())),
]
T["desc"] = "\n".join(rows)
V["h_dups"] = int(Xh.assign(y=yh).duplicated().sum())

if __name__ == "__main__":
    for k, v in V.items(): print(k, "=", v)
