import pickle, time, sys
from hdlib import *
X, y, raw = load_cleveland()
res = {}
t0 = time.time()
for m in MODELS:
    df, oof, _ = cv_eval(m, X, y, ALL_FEATS, n_repeats=10, record_time=True)
    res[m] = (df, oof)
    print(m, round(df.auc.mean(),3), round(df.acc.mean(),3), "elapsed", round(time.time()-t0), flush=True)
pickle.dump(res, open("results/main.pkl", "wb"))
print("done", flush=True)
