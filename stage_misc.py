import pickle, time
from hdlib import *
from sklearn.model_selection import StratifiedKFold
X, y, raw = load_cleveland()
out = {}
t0 = time.time()

# ---- (a) MLP multi-seed analysis: 20 initialisation seeds on one fixed 10-fold partition
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=CV_SEED)
seed_rows = []
for s in range(20):
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        pipe = make_pipeline("NN", ALL_FEATS, seed=s)
        pipe.fit(X.iloc[tr][ALL_FEATS], y[tr])
        p = pipe.predict_proba(X.iloc[te][ALL_FEATS])[:, 1]
        r = fold_metrics(y[te], p)
        accs.append(r["acc"]); aucs.append(r["auc"])
    seed_rows.append(dict(seed=s, acc=np.mean(accs), auc=np.mean(aucs)))
    print("seed", s, round(np.mean(aucs),3), round(time.time()-t0), flush=True)
out["mlp_seeds"] = pd.DataFrame(seed_rows)

# ---- (b) duplicate audit and replication-inflation simulation
Xd = X.copy()
out["n_exact_dups"] = int(Xd.assign(y=y).duplicated().sum())
# near duplicates: record pairs agreeing exactly on >= 12 of the 13 attributes
A = Xd.values
n = len(A)
near = 0
for i in range(n):
    eq = (A[i + 1:] == A[i])            # NaN != NaN, so missing values never match
    near += int(((eq.sum(axis=1)) >= 12).sum())
out["n_near_dup_pairs"] = near
out["iqr_outliers_total"] = int(iqr_outlier_mask(X).sum())
out["iqr_outliers_by_col"] = {c: int(iqr_outlier_mask(X, [c]).sum()) for c in OUTLIER_COLS}

rng = np.random.default_rng(7)
sim = []
for rep in range(20):
    extra = rng.integers(0, len(X), size=1025 - len(X))          # 722 repeated draws
    idx = np.concatenate([np.arange(len(X)), extra])
    Xr, yr = X.iloc[idx].reset_index(drop=True), y[idx]
    n_unique = len(np.unique(idx))
    skf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=rep)
    accs, aucs = [], []
    for tr, te in skf2.split(Xr, yr):
        pipe = make_pipeline("RF", ALL_FEATS, seed=rep)
        pipe.fit(Xr.iloc[tr][ALL_FEATS], yr[tr])
        p = pipe.predict_proba(Xr.iloc[te][ALL_FEATS])[:, 1]
        r = fold_metrics(yr[te], p)
        accs.append(r["acc"]); aucs.append(r["auc"])
    # share of validation records whose exact copy sits in the training fold
    sim.append(dict(rep=rep, n_unique=n_unique, acc=np.mean(accs), auc=np.mean(aucs)))
    print("sim", rep, round(np.mean(accs),3), round(time.time()-t0), flush=True)
out["dup_sim"] = pd.DataFrame(sim)

# ---- (c) external validation on the Hungarian cohort
Xh, yh, rawh = load_hungarian()
ext = []
rngb = np.random.default_rng(11)
for fs in ["F10_common", "F7_resting"]:
    feats = FEATURE_SETS[fs]
    for m in MODELS:
        seeds = range(10) if m == "NN" else [42]
        ps = []
        for s in seeds:
            pipe = make_pipeline(m, feats, seed=s)
            pipe.fit(X[feats], y)
            ps.append(pipe.predict_proba(Xh[feats])[:, 1])
        p = np.mean(ps, axis=0)
        r = fold_metrics(yh, p)
        aucs = []
        sens = []; spec = []; accs = []
        for b in range(1000):
            ib = rngb.integers(0, len(yh), len(yh))
            if len(np.unique(yh[ib])) < 2:
                continue
            rb = fold_metrics(yh[ib], p[ib])
            aucs.append(rb["auc"]); sens.append(rb["rec"]); spec.append(rb["spec"]); accs.append(rb["acc"])
        r.update(fs=fs, model=m,
                 auc_lo=np.percentile(aucs, 2.5), auc_hi=np.percentile(aucs, 97.5),
                 acc_lo=np.percentile(accs, 2.5), acc_hi=np.percentile(accs, 97.5),
                 sens_lo=np.percentile(sens, 2.5), sens_hi=np.percentile(sens, 97.5),
                 spec_lo=np.percentile(spec, 2.5), spec_hi=np.percentile(spec, 97.5))
        ext.append(r)
        print("ext", fs, m, round(r["auc"],3), flush=True)
out["external"] = pd.DataFrame(ext)
out["hung_n"] = len(yh)
out["hung_prev"] = float(yh.mean())
pickle.dump(out, open("results/misc.pkl", "wb"))
print("done", round(time.time()-t0), flush=True)
