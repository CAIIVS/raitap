"""Generate the artifacts for the tree/XGBoost SHAP smoke config (#246).

Trains a tiny XGBoost classifier on deterministic synthetic data (mirrors the
acceptance test in ``test_e2e_integration.py``) and writes:

- ``artifacts/model.ubj``    — fitted XGBClassifier in XGBoost's native format
- ``artifacts/features.csv`` — the (N, 6) feature matrix the assessment explains

Run once before the assessment (artifacts are also committed, so this is only
needed to regenerate them):

    uv run --extra xgboost python contributor-configs/tree-xgboost-shap/build_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost

FEATURE_COUNT = 6
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(64, FEATURE_COUNT)).astype(np.float32)
    # Label depends on the feature sum, so SHAP attributions are meaningful.
    labels = (features.sum(axis=1) > 0).astype(int)

    clf = xgboost.XGBClassifier(n_estimators=16, max_depth=3)
    clf.fit(features, labels)
    clf.save_model(ARTIFACTS / "model.ubj")

    # The assessment explains a readable subset of rows.
    header = ",".join(f"f{i}" for i in range(FEATURE_COUNT))
    np.savetxt(
        ARTIFACTS / "features.csv",
        features[:16],
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6f",
    )
    print(f"Wrote {ARTIFACTS / 'model.ubj'} and {ARTIFACTS / 'features.csv'}")


if __name__ == "__main__":
    main()
