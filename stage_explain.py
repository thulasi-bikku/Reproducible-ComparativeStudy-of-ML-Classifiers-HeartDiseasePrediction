import pickle, time
from sklearn.calibration import calibration_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from hdlib import *
X, y, raw = load_cleveland()
main = pickle.load(open("results/main.pkl", "rb"))
out = {}
t0 = time.time()

# ---------- aggregate performance ----------
agg = {}
for m in MODELS:
    df = main[m][0]
    agg[m] = {k: (df[k].mean(), df[k].std(ddof=1)) for k in ["acc", "prec", "rec", "spec", "f1", "auc", "brier"]}
out["agg"] = agg
rank = sorted(MODELS, key=lambda m: -agg[m]["auc"][0])
out["rank_auc"] = rank
best = rank[0]
out["best"] = best

# ---------- runtime ----------
rt = {}
for m in MODELS:
    df = main[m][0]
    rt[m] = dict(fit_mean=df.fit_s.mean(), fit_sd=df.fit_s.std(ddof=1),
                 pred_mean=df.pred_ms_per_patient.mean(), pred_sd=df.pred_ms_per_patient.std(ddof=1))
out["runtime"] = rt
nn = main["NN"][0]
out["nn_iter"] = dict(mean=float(nn.n_iter.mean()), min=int(nn.n_iter.min()), max=int(nn.n_iter.max()),
                      frac_at_limit=float((nn.n_iter >= 500).mean()))

# ---------- statistical comparison against the best model ----------
ratio = (main[best][0].n_test / main[best][0].n_train).mean()
comp = []
for metric in ["auc", "acc"]:
    pass
rows = []
for m in MODELS:
    if m == best:
        continue
    r = dict(model=m)
    for metric in ["auc", "acc"]:
        d = main[best][0][metric].values - main[m][0][metric].values
        mean, se, t, p, ci = corrected_resampled_t(d, 1.0, ratio)
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
        r.update({f"{metric}_diff": mean, f"{metric}_lo": ci[0], f"{metric}_hi": ci[1],
                  f"{metric}_p": p, f"{metric}_dz": dz})
    rows.append(r)
cdf = pd.DataFrame(rows)
cdf["auc_p_holm"] = holm(cdf.auc_p.values)
cdf["acc_p_holm"] = holm(cdf.acc_p.values)
out["comparison"] = cdf
# Friedman omnibus on repeat-level mean AUC
rep_auc = np.array([[main[m][0][main[m][0].repeat == r].auc.mean() for m in MODELS] for r in range(10)])
fr = stats.friedmanchisquare(*[rep_auc[:, j] for j in range(len(MODELS))])
out["friedman"] = (float(fr.statistic), float(fr.pvalue))
# Wilcoxon on 100 folds (naive, for reference)
out["wilcoxon_naive"] = {m: float(stats.wilcoxon(main[best][0].auc, main[m][0].auc).pvalue) for m in MODELS if m != best}

# ---------- calibration ----------
yrep = np.tile(y, 10)
cal = {}
for m in MODELS:
    oof = main[m][1]
    p = oof.flatten()
    s, i = calib_slope_intercept(yrep, p)
    fx, fy = calibration_curve(yrep, p, n_bins=8, strategy="quantile")
    cal[m] = dict(slope=s, intercept=i, brier=agg[m]["brier"][0], brier_sd=agg[m]["brier"][1], curve=(fx, fy))
out["calibration"] = cal

# ---------- sex-stratified performance ----------
sex = X["sex"].values
rng = np.random.default_rng(5)
fair = {}
def metrics_sex(yy, pp):
    r = fold_metrics(yy, pp)
    return r["auc"], r["rec"], r["spec"], 1 - r["rec"], r["brier"], r["acc"]
for m in MODELS:
    pbar = main[m][1].mean(axis=0)
    fair[m] = {}
    for sx, lab in [(1, "Male"), (0, "Female")]:
        idx = np.where(sex == sx)[0]
        yy, pp = y[idx], pbar[idx]
        est = metrics_sex(yy, pp)
        bs = []
        for b in range(1000):
            ib = rng.integers(0, len(idx), len(idx))
            if len(np.unique(yy[ib])) < 2:
                continue
            bs.append(metrics_sex(yy[ib], pp[ib]))
        bs = np.array(bs)
        lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
        slope, icpt = calib_slope_intercept(yy, pp)
        fair[m][lab] = dict(n=len(idx), n_pos=int(yy.sum()), est=est, lo=lo, hi=hi, slope=slope, intercept=icpt)
out["fairness"] = fair
out["sex_prev"] = {lab: float(y[sex == sx].mean()) for sx, lab in [(1, "Male"), (0, "Female")]}

# ---------- feature importance: MDI, permutation (held-out), Monte-Carlo Shapley ----------
def group_mdi():
    pipe = make_pipeline("RF", ALL_FEATS, seed=0)
    pipe.fit(X[ALL_FEATS], y)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_
    g = {f: 0.0 for f in ALL_FEATS}
    for n_, v in zip(names, imp):
        base = n_.split("__", 1)[1]
        key = next(f for f in ALL_FEATS if base == f or base.startswith(f + "_"))
        g[key] += v
    return g, len(names)
mdi, n_enc = group_mdi()
out["mdi"] = mdi
out["n_encoded"] = n_enc

def perm_importance(name, n_rep=3, n_shuffle=5):
    cvp = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_rep, random_state=CV_SEED)
    rg = np.random.default_rng(3)
    res = []
    for i, (tr, te) in enumerate(cvp.split(X, y)):
        pipe = make_pipeline(name, ALL_FEATS, seed=i // 10)
        pipe.fit(X.iloc[tr][ALL_FEATS], y[tr])
        Xv, yv = X.iloc[te][ALL_FEATS], y[te]
        base = roc_auc_score(yv, pipe.predict_proba(Xv)[:, 1])
        drops = []
        for f in ALL_FEATS:
            dd = []
            for s in range(n_shuffle):
                Xp = Xv.copy()
                Xp[f] = rg.permutation(Xp[f].values)
                dd.append(base - roc_auc_score(yv, pipe.predict_proba(Xp)[:, 1]))
            drops.append(np.mean(dd))
        res.append(drops)
    res = np.array(res)
    return res.mean(axis=0), res.std(axis=0, ddof=1)

def mc_shapley(name, M=100):
    cvp = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=CV_SEED)
    rg = np.random.default_rng(9)
    phi = np.zeros((len(y), len(ALL_FEATS)))
    d = len(ALL_FEATS)
    for i, (tr, te) in enumerate(cvp.split(X, y)):
        pipe = make_pipeline(name, ALL_FEATS, seed=0)
        pipe.fit(X.iloc[tr][ALL_FEATS], y[tr])
        xt = X.iloc[tr][ALL_FEATS].values.astype(float)
        xv = X.iloc[te][ALL_FEATS].values.astype(float)
        nv = len(te)
        rows = np.empty((nv, M, d + 1, d))
        perms = np.empty((nv, M, d), dtype=int)
        for s in range(nv):
            for m_ in range(M):
                pm = rg.permutation(d)
                z = xt[rg.integers(len(xt))].copy()
                rows[s, m_, 0] = z
                cur = z.copy()
                for k in range(d):
                    cur[pm[k]] = xv[s, pm[k]]
                    rows[s, m_, k + 1] = cur
                perms[s, m_] = pm
        fx = pipe.predict_proba(pd.DataFrame(rows.reshape(-1, d), columns=ALL_FEATS))[:, 1].reshape(nv, M, d + 1)
        contrib = np.diff(fx, axis=2)                       # (nv, M, d) in permutation order
        for s in range(nv):
            for m_ in range(M):
                np.add.at(phi[te[s]], perms[s, m_], contrib[s, m_] / M)
    return phi

imp = {}
for nm in sorted(set(["RF", best])):
    pm, ps = perm_importance(nm)
    phi = mc_shapley(nm)
    imp[nm] = dict(perm_mean=dict(zip(ALL_FEATS, pm)), perm_sd=dict(zip(ALL_FEATS, ps)),
                   shap_abs=dict(zip(ALL_FEATS, np.abs(phi).mean(axis=0))), phi=phi)
    print("importance", nm, round(time.time() - t0), flush=True)
out["importance"] = imp

# ---------- correlation and collinearity ----------
num = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
cm = raw[num + ["num"]].assign(num=(raw.num > 0).astype(int)).corr()
out["corr"] = cm
pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:]]
out["max_abs_r"] = max(pairs, key=lambda t: abs(t[2]))
# VIF on drop-first encoded, imputed design matrix
Xi = X.copy()
for c in Xi.columns:
    Xi[c] = Xi[c].fillna(Xi[c].median() if c in CONT else Xi[c].mode().iloc[0])
D = pd.get_dummies(Xi, columns=CAT, drop_first=True).astype(float)
Z = ((D - D.mean()) / D.std()).values
vif = {}
for j, c in enumerate(D.columns):
    others = np.delete(Z, j, axis=1)
    A = np.c_[np.ones(len(Z)), others]
    beta, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
    resid = Z[:, j] - A @ beta
    r2 = 1 - resid.var() / Z[:, j].var()
    vif[c] = 1 / (1 - r2)
out["vif"] = vif

pickle.dump(out, open("results/explain.pkl", "wb"))
print("done", round(time.time() - t0), flush=True)
