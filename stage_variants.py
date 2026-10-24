import pickle, time
from hdlib import *
X, y, raw = load_cleveland()
out = {}
t0 = time.time()
# outlier handling sensitivity (all 13 features)
for mode in ["global", "infold"]:
    for m in MODELS:
        df, oof, keep = cv_eval(m, X, y, ALL_FEATS, n_repeats=10, outlier=mode)
        out[("outlier", mode, m)] = (df, oof, int((~keep).sum()))
        print(mode, m, round(df.auc.mean(),3), round(time.time()-t0), flush=True)
# feature-set scenarios
for fs in ["F12_no_ca", "F11_no_ca_thal", "F10_common", "F7_resting"]:
    for m in MODELS:
        df, oof, _ = cv_eval(m, X, y, FEATURE_SETS[fs], n_repeats=10)
        out[("fs", fs, m)] = (df, oof, 0)
        print(fs, m, round(df.auc.mean(),3), round(time.time()-t0), flush=True)
pickle.dump(out, open("results/variants.pkl", "wb"))
print("done", flush=True)
