"""Nested cross-validation with an identical tuning budget for every classifier.
Outer: repeated stratified 10-fold (first two repeats of the main design);
inner: stratified 5-fold random search, 15 candidates, ROC-AUC criterion."""
import pickle, time
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from hdlib import *
X, y, raw = load_cleveland()
SPACES = {
 "LR":  {"clf__C": loguniform(1e-3, 1e2)},
 "NB":  {"clf__var_smoothing": loguniform(1e-11, 1e-1)},
 "SVM": {"clf__C": loguniform(1e-2, 1e3), "clf__gamma": loguniform(1e-4, 1e0)},
 "KNN": {"clf__n_neighbors": randint(3, 31), "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
 "DT":  {"clf__max_depth": randint(2, 12), "clf__min_samples_leaf": randint(1, 20), "clf__criterion": ["gini", "entropy"]},
 "RF":  {"clf__n_estimators": randint(50, 201), "clf__max_depth": [None, 3, 5, 8, 12],
         "clf__min_samples_leaf": randint(1, 10), "clf__max_features": ["sqrt", "log2", 0.5]},
 "GB":  {"clf__n_estimators": randint(50, 301), "clf__learning_rate": loguniform(0.01, 0.3),
         "clf__max_depth": randint(1, 5), "clf__subsample": uniform(0.6, 0.4)},
 "NN":  {"clf__hidden_layer_sizes": [(5,), (11,), (20,), (32,), (16, 8)],
         "clf__alpha": loguniform(1e-5, 1e-1), "clf__learning_rate_init": loguniform(1e-4, 1e-2)},
}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=CV_SEED)
import os
out = pickle.load(open("results/nested.pkl","rb")) if os.path.exists("results/nested.pkl") else {}
t0 = time.time()
for m in MODELS:
    if m in out: continue
    rows, chosen = [], []
    for i, (tr, te) in enumerate(cv.split(X, y)):
        rep = i // 10
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        search = RandomizedSearchCV(make_pipeline(m, ALL_FEATS, seed=rep), SPACES[m], n_iter=15,
                                    scoring="roc_auc", cv=inner, random_state=i, refit=True, n_jobs=1)
        search.fit(X.iloc[tr][ALL_FEATS], y[tr])
        p = search.predict_proba(X.iloc[te][ALL_FEATS])[:, 1]
        r = fold_metrics(y[te], p); r.update(repeat=rep, fold=i % 10)
        rows.append(r); chosen.append(search.best_params_)
    out[m] = (pd.DataFrame(rows), chosen)
    print(m, round(out[m][0].auc.mean(), 3), round(out[m][0].acc.mean(), 3), "elapsed", round(time.time() - t0), flush=True)
    pickle.dump(out, open("results/nested.pkl", "wb"))
print("done", flush=True)
