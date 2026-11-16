"""Script to test trained classifiers on the test dataset and evaluate their performance."""

import matplotlib.pyplot as plt
import pandas as pd

from maneuver_detection.config import get_default_config
from maneuver_detection.evaluate import evaluate
from maneuver_detection.helpers import (
    extract_features_and_labels,
    load_classifiers,
    save_open_figures,
)

config = get_default_config()
classifiers = load_classifiers(config.classifiers_path)

if not config.test_data_path.exists():
    raise FileNotFoundError(
        f"{config.test_data_path} not found. "
        "Run `python -m maneuver_detection.preprocess --include-test` first."
    )

df_test = pd.read_parquet(config.test_data_path)
x_test, y_test = extract_features_and_labels(df_test)

for classifier in classifiers:
    classifier_name = type(classifier).__name__
    print(f"Testing {classifier_name} ...")

    y_pred = classifier.predict(x_test)
    metrics = evaluate(f"{classifier_name} – test", y_test, y_pred, config)

    print(f"  Maneuver recall:                {metrics.maneuver_recall:.3f}")
    print(f"  Maneuver recall (excl. no man.): {metrics.maneuver_recall_only_foreground:.3f}")
    print(f"  Precision:                      {metrics.precision:.3f}")
    print(f"  Accuracy:                       {metrics.accuracy:.3f}")

save_open_figures(config.project_root / "plots", prefix="test")
plt.show()
