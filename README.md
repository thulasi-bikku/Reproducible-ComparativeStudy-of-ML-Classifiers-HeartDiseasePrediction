# Heart-disease classification benchmark (MO-D-26-00079, revision)

Run order (single core, about 30 minutes in total):
1. `python stage_main.py`      repeated 10x10-fold CV, eight classifiers, default settings
2. `python stage_variants.py`  outlier handling and predictor-set scenarios
3. `python stage_misc.py`      MLP seeds, duplicate audit and replication experiment, external validation
4. `python stage_nested.py`    nested CV with random search (resumable)
5. `python stage_explain.py`   statistics, calibration, sex-stratified results, importance, VIF
6. `python make_figs.py && python make_figs2.py`   figures
`build_values.py` computes every number and table used in the manuscript from `results/*.pkl`.

Data: `data/cleveland.csv` (303 rows) and `data/hungarian.csv` (294 rows) are the UCI files
`processed.cleveland.data` and `processed.hungarian.data` (UCI Heart Disease, doi:10.24432/C52P4X),
header-free, missing values coded as "?". SHA-256 checksums are in the manuscript.
Seeds: cross-validation 2026; stochastic models use the repeat index; random search uses the fold index.
`results_logs/` holds the console logs of the run reported in the paper.
