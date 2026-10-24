"""Shared utilities for the revised heart-disease classification benchmark.

All data-dependent operations (imputation, scaling, one-hot encoding and, in the
sensitivity analysis, IQR-based outlier exclusion) are fitted on the training
portion of every fold only.
"""
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA = "/home/claude/work/data/"
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
ALL_FEATS = COLS[:-1]
CONT = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINORD = ["sex", "fbs", "exang", "ca"]
CAT = ["cp", "restecg", "slope", "thal"]
OUTLIER_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
MODELS = ["LR", "NB", "SVM", "KNN", "DT", "RF", "GB", "NN"]
SCALED = {"LR", "SVM", "KNN", "NN"}
CV_SEED = 2026

FEATURE_SETS = {
    "F13_all": ALL_FEATS,
    "F12_no_ca": [f for f in ALL_FEATS if f != "ca"],
    "F11_no_ca_thal": [f for f in ALL_FEATS if f not in ("ca", "thal")],
    "F10_common": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak"],
    "F7_resting": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"],
}


def load_cleveland():
    d = pd.read_csv(DATA + "cleveland.csv", header=None, names=COLS, na_values="?")
    y = (d["num"] > 0).astype(int).values
    return d[ALL_FEATS].copy(), y, d


def load_hungarian():
    d = pd.read_csv(DATA + "hungarian.csv", header=None, names=COLS, na_values="?")
    y = (d["num"] > 0).astype(int).values
    return d[ALL_FEATS].copy(), y, d


def build_pre(feats, scale):
    cont = [f for f in CONT if f in feats]
    binord = [f for f in BINORD if f in feats]
    cat = [f for f in CAT if f in feats]
    tr = []
    if cont:
        steps = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("sc", StandardScaler()))
        tr.append(("cont", Pipeline(steps), cont))
    if binord:
        steps = [("imp", SimpleImputer(strategy="most_frequent"))]
        if scale:
            steps.append(("sc", StandardScaler()))
        tr.append(("bin", Pipeline(steps), binord))
    if cat:
        tr.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(tr)


def make_estimator(name, seed):
    if name == "LR":
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
                                  random_state=seed)
    if name == "NB":
        return GaussianNB()
    if name == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=7, weights="uniform", metric="minkowski", p=2)
    if name == "DT":
        return DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    if name == "GB":
        return GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                          subsample=1.0, random_state=seed)
    if name == "NN":
        return MLPClassifier(hidden_layer_sizes=(11,), activation="relu", solver="adam",
                             max_iter=500, random_state=seed)
    raise ValueError(name)


def make_pipeline(name, feats, seed):
    return Pipeline([("pre", build_pre(feats, name in SCALED)),
                     ("clf", make_estimator(name, seed))])


def iqr_outlier_mask(X, cols=OUTLIER_COLS, k=1.5):
    """True for rows lying outside [Q1-k*IQR, Q3+k*IQR] on at least one column."""
    mask = np.zeros(len(X), dtype=bool)
    for c in cols:
        q1, q3 = np.nanpercentile(X[c], [25, 75])
        iqr = q3 - q1
        mask |= ((X[c] < q1 - k * iqr) | (X[c] > q3 + k * iqr)).values
    return mask


def fold_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    return dict(acc=acc, prec=prec, rec=rec, spec=spec, f1=f1, auc=auc, brier=brier,
                tp=tp, tn=tn, fp=fp, fn=fn)


def cv_eval(name, X, y, feats, n_repeats, outlier="none", record_time=False, seed_cv=CV_SEED):
    """Repeated stratified 10-fold CV.  Returns (DataFrame of fold metrics, OOF matrix).

    outlier: 'none' | 'global' | 'infold'.
      'global'  : IQR rule applied once to the whole data set before CV (original protocol).
      'infold'  : IQR rule applied to the training fold only; validation fold untouched.
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    keep = np.ones(len(X), dtype=bool)
    if outlier == "global":
        keep = ~iqr_outlier_mask(X)
        X = X[keep].reset_index(drop=True)
        y = y[keep]
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats, random_state=seed_cv)
    rows = []
    oof = np.full((n_repeats, len(y)), np.nan)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        rep = i // 10
        Xtr, ytr = X.iloc[tr], y[tr]
        if outlier == "infold":
            m = ~iqr_outlier_mask(Xtr)
            Xtr, ytr = Xtr[m], ytr[m]
        pipe = make_pipeline(name, feats, seed=rep)
        t0 = time.perf_counter()
        pipe.fit(Xtr[feats], ytr)
        t1 = time.perf_counter()
        p = pipe.predict_proba(X.iloc[te][feats])[:, 1]
        t2 = time.perf_counter()
        r = fold_metrics(y[te], p)
        r.update(repeat=rep, fold=i % 10, n_train=len(ytr), n_test=len(te))
        if record_time:
            r.update(fit_s=t1 - t0, pred_ms_per_patient=1000 * (t2 - t1) / len(te))
            if name == "NN":
                r.update(n_iter=pipe.named_steps["clf"].n_iter_)
        rows.append(r)
        oof[rep, te] = p
    return pd.DataFrame(rows), oof, keep


def corrected_resampled_t(diff, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test on per-fold paired differences."""
    k = len(diff)
    mean = float(np.mean(diff))
    var = float(np.var(diff, ddof=1))
    se = np.sqrt((1.0 / k + n_test / n_train) * var)
    if se == 0:
        return mean, 0.0, np.nan, 1.0, (mean, mean)
    t = mean / se
    df = k - 1
    p = 2 * stats.t.sf(abs(t), df)
    crit = stats.t.ppf(0.975, df)
    return mean, se, t, p, (mean - crit * se, mean + crit * se)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def calib_slope_intercept(y, p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])
