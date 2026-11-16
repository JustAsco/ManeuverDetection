"""Train and evaluate classifiers for maneuver detection.

Data split:
    - Test: the flights in ``test_data_ids`` (untouched here, see ``test.py``).
    - Train / validation: k-fold cross-validation over the training flights.
      Folds are formed from whole flights (StratifiedGroupKFold), because the
      windows overlap by 50% and neighbouring windows are nearly identical.
      Splitting individual windows would leak validation data into training.

The model variant with the best cross-validated score is refit on all
training flights and saved.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold

from maneuver_detection.config import get_default_config
from maneuver_detection.evaluate import evaluate
from maneuver_detection.evaluation_metrics import EvaluationMetrics
from maneuver_detection.helpers import (
    extract_features_and_labels,
    save_classifiers,
    save_open_figures,
)

N_FOLDS = 5
RANDOM_STATE = 42
# Metric used to pick the best model variant (attribute of EvaluationMetrics)
SELECTION_METRIC = "maneuver_recall_only_foreground"
METRIC_NAMES = [
    "maneuver_recall",
    "maneuver_recall_only_foreground",
    "precision",
    "accuracy",
]

config = get_default_config()

# Candidate LDA variants compared via cross-validation
candidates: dict[str, BaseEstimator] = {
    "LDA (svd)": LinearDiscriminantAnalysis(solver="svd"),
    "LDA (lsqr, auto shrinkage)": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
}

df_train = pd.read_parquet(config.train_data_path)
metadata = pd.read_csv(config.data_path / "train_window_metadata.csv")
if len(metadata) != len(df_train):
    raise ValueError("Window metadata does not match train data. Re-run preprocessing.")

x_train, y_train = extract_features_and_labels(df_train)
groups = metadata["Flight_ID"].to_numpy()

cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds = list(cv.split(x_train, y_train, groups=groups))

print(f"{len(np.unique(groups))} training flights, {len(y_train)} windows, {N_FOLDS} folds")
for i, (_, val_idx) in enumerate(folds):
    print(f"  fold {i}: {len(np.unique(groups[val_idx]))} validation flights, {len(val_idx)} windows")

results = {}
oof_predictions = {}

for name, candidate in candidates.items():
    print(f"\nCross-validating {name} ...")
    fold_scores = []
    y_oof = np.empty_like(y_train)

    for i, (train_idx, val_idx) in enumerate(folds):
        model = clone(candidate).fit(x_train[train_idx], y_train[train_idx])
        # val_idx is sorted, so windows stay in temporal order per flight,
        # which the maneuver-level metrics rely on.
        y_pred = model.predict(x_train[val_idx])
        y_oof[val_idx] = y_pred

        metrics = EvaluationMetrics(y_train[val_idx], y_pred, config)
        fold_scores.append({m: getattr(metrics, m) for m in METRIC_NAMES})

    scores = pd.DataFrame(fold_scores)
    results[name] = scores
    oof_predictions[name] = y_oof
    print(scores.agg(["mean", "std"]).round(3).to_string())

summary = pd.DataFrame({name: s.mean() for name, s in results.items()}).T
print("\nMean cross-validation scores:")
print(summary.round(3).to_string())

best_name = summary[SELECTION_METRIC].idxmax()
print(f"\nSelected: {best_name} (best mean {SELECTION_METRIC})")

# Refit on all training flights and save (before plotting, so a plotting
# problem cannot lose the trained model)
print(f"Fitting {best_name} on all training flights ...")
final_model = clone(candidates[best_name]).fit(x_train, y_train)
save_classifiers([final_model], config.classifiers_path)

# Plots for the selected model, based on out-of-fold validation predictions
evaluate(f"{best_name} – cross-validation", y_train, oof_predictions[best_name], config)
save_open_figures(config.project_root / "plots", prefix="train_cv")

plt.show()
