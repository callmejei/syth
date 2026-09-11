"""Utility evaluation: does the synthetic data actually work as a substitute?

Marginal and correlation comparisons (report.py) answer "does it look like the
real data". That is necessary but not sufficient: a model can match every
marginal and still be useless for the thing people want synthetic data FOR,
which is building and testing things without touching production records.

Two tests here, and they measure opposite failure modes:

  TSTR (train on synthetic, test on real)
      Train a predictor on synthetic data, evaluate it on held-out REAL data.
      Compare against the same predictor trained on real data. The ratio is the
      number that matters: 0.95 means a model built on synthetic data is almost
      as good as one built on the real thing. This is the headline utility
      metric and the one to quote to a stakeholder.

  Detection (real vs synthetic discriminator)
      Train a classifier to tell real rows from synthetic ones. AUC near 0.5
      means indistinguishable; near 1.0 means the synthetic data has a tell.
      High detection AUC with high TSTR is possible and worth knowing about --
      it usually means some column has an artefact (a placeholder pattern, a
      clipped range) that does not affect the prediction task.

Both are deliberately built on small, fast, well-understood models. The point is
to compare two datasets, not to win a modelling competition, and a complex
learner would make the comparison harder to interpret.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Columns with more distinct values than this are treated as identifiers and
# never used as a prediction target or feature.
MAX_FEATURE_CARDINALITY = 50
MIN_ROWS_FOR_EVALUATION = 100


def _prepare(
    frame: pd.DataFrame, feature_columns: list[str], reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Encode features consistently across the real and synthetic frames.

    Categories are aligned to the reference frame so that both sides produce the
    same columns -- otherwise the two models would not be comparable.
    """
    out = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        series = frame[column] if column in frame.columns else pd.Series(index=frame.index)

        # Choose the encoding from the REFERENCE column, and apply it to both
        # sides. Branching on each frame's own dtype silently encodes the same
        # value two different ways whenever the dtypes differ -- a nullable
        # integer column is `object` in the source and `float64` after a
        # round-trip, so the value 4 becomes a category index on one side and
        # the number 4.0 on the other. A detection classifier then scores AUC
        # 1.0 against an artefact of this function rather than a real
        # difference in the data.
        decider = (
            reference[column]
            if reference is not None and column in reference.columns
            else series
        )

        if pd.api.types.is_datetime64_any_dtype(decider):
            out[column] = pd.to_datetime(series, errors="coerce").astype("int64") / 1e9
        elif pd.api.types.is_numeric_dtype(decider) and not pd.api.types.is_bool_dtype(
            decider
        ):
            out[column] = pd.to_numeric(series, errors="coerce")
        else:
            # Judge numeric-ness on the NON-NULL values only. Including nulls
            # means a column that is 10% null fails the test and falls through
            # to string encoding, where "4" (object column) and "4.0" (same
            # column after a float round-trip) share no categories at all and
            # every synthetic row encodes to the unknown bucket.
            non_null_reference = pd.Series(decider).dropna()
            numeric_reference = pd.to_numeric(non_null_reference, errors="coerce")
            if len(non_null_reference) and numeric_reference.notna().mean() > 0.95:
                # Numbers held in an object column: compare them as numbers.
                out[column] = pd.to_numeric(series, errors="coerce")
            else:
                categories = (
                    pd.Series(decider).astype(str).value_counts().index[
                        :MAX_FEATURE_CARDINALITY
                    ]
                )
                mapping = {value: index for index, value in enumerate(categories)}
                out[column] = series.astype(str).map(mapping).fillna(-1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(-999)


def choose_targets(
    frame: pd.DataFrame,
    explicit: list[str] | None = None,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Pick columns worth predicting.

    Prefers explicitly configured targets (schema.yaml has `target_columns` for
    exactly this), falling back to any low-cardinality categorical -- the kind
    of column a downstream classifier would actually be built on.

    Keys are excluded. Predicting a surrogate key is meaningless, and because
    synthetic keys are minted in a different range from real ones, including
    them produces spectacular nonsense (R^2 of -10^9) that says nothing about
    data quality.
    """
    targets: list[dict[str, Any]] = []
    exclude = exclude or set()
    candidates = explicit or list(frame.columns)

    for column in candidates:
        if column not in frame.columns or column in exclude:
            continue
        series = frame[column].dropna()
        if series.empty:
            continue
        n_unique = series.nunique()

        if pd.api.types.is_bool_dtype(series) or (
            not pd.api.types.is_numeric_dtype(series) and 2 <= n_unique <= 20
        ):
            counts = series.value_counts(normalize=True)
            if counts.iloc[0] > 0.98:
                continue  # degenerate: almost one class
            targets.append({"column": column, "task": "classification"})
        elif pd.api.types.is_numeric_dtype(series) and n_unique > MAX_FEATURE_CARDINALITY:
            targets.append({"column": column, "task": "regression"})
        elif pd.api.types.is_numeric_dtype(series) and 2 <= n_unique <= 20 and explicit:
            targets.append({"column": column, "task": "classification"})

    return targets[:4]


def _feature_columns(frame: pd.DataFrame, target: str, exclude: set[str]) -> list[str]:
    columns = []
    for column in frame.columns:
        if column == target or column in exclude:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            columns.append(column)
        elif series.nunique() <= MAX_FEATURE_CARDINALITY:
            columns.append(column)
    return columns


def _score_classification(
    train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series
) -> float | None:
    if train_y.nunique() < 2 or test_y.nunique() < 2:
        return None
    model = RandomForestClassifier(
        n_estimators=60, max_depth=8, random_state=0, n_jobs=-1
    )
    model.fit(train_x, train_y)

    classes = list(model.classes_)
    probabilities = model.predict_proba(test_x)
    try:
        if len(classes) == 2:
            positive = classes[1]
            return float(roc_auc_score((test_y == positive).astype(int), probabilities[:, 1]))
        # Multiclass: restrict to classes the model actually saw.
        mask = test_y.isin(classes)
        if mask.sum() < 10 or test_y[mask].nunique() < 2:
            return None
        return float(
            roc_auc_score(
                test_y[mask], probabilities[mask.to_numpy()], multi_class="ovr",
                average="macro", labels=classes,
            )
        )
    except ValueError as exc:
        logger.info("classification scoring skipped: %s", exc)
        return None


def _score_regression(
    train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series
) -> float | None:
    model = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=0, n_jobs=-1)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    return float(r2_score(test_y, predictions))


def run_tstr(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    targets: list[str] | None = None,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Train on real vs train on synthetic; both evaluated on held-out real."""
    exclude = exclude or set()
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for target in choose_targets(real_train, targets, exclude=exclude):
        column, task = target["column"], target["task"]
        if column not in synthetic.columns:
            continue

        features = _feature_columns(real_train, column, exclude)
        if not features:
            continue

        train_real_x = _prepare(real_train, features, real_train)
        test_x = _prepare(real_test, features, real_train)
        train_synth_x = _prepare(synthetic, features, real_train)

        if task == "classification":
            real_y = real_train[column].astype(str)
            synth_y = synthetic[column].astype(str)
            test_y = real_test[column].astype(str)
            scorer = _score_classification
            metric = "roc_auc"
        else:
            real_y = pd.to_numeric(real_train[column], errors="coerce")
            synth_y = pd.to_numeric(synthetic[column], errors="coerce")
            test_y = pd.to_numeric(real_test[column], errors="coerce")
            valid = real_y.notna()
            real_y, train_real_x = real_y[valid], train_real_x[valid]
            valid_synth = synth_y.notna()
            synth_y, train_synth_x = synth_y[valid_synth], train_synth_x[valid_synth]
            valid_test = test_y.notna()
            test_y, test_x = test_y[valid_test], test_x[valid_test]
            scorer = _score_regression
            metric = "r2"

        if len(real_y) < MIN_ROWS_FOR_EVALUATION or len(synth_y) < MIN_ROWS_FOR_EVALUATION:
            continue

        try:
            baseline = scorer(train_real_x, real_y, test_x, test_y)
            tstr = scorer(train_synth_x, synth_y, test_x, test_y)
        except Exception as exc:  # noqa: BLE001
            logger.info("TSTR failed for %s: %s", column, exc)
            continue

        # A scorer can legitimately return NaN (e.g. a constant target in the
        # test split). JSON has no NaN literal, so letting one through breaks
        # persistence of the whole report.
        if baseline is None or tstr is None:
            continue
        if not np.isfinite(baseline) or not np.isfinite(tstr):
            skipped.append(
                {
                    "target": column,
                    "metric": metric,
                    "train_on_real": None,
                    "reason": "scoring produced a non-finite value",
                }
            )
            continue

        # A target the REAL data cannot predict is not a utility test. If a
        # model trained on real rows scores at chance, then "synthetic matches
        # real" only means both are useless, and the ratio divides one noise
        # figure by another. Report these separately instead of scoring them.
        if metric == "roc_auc" and baseline < 0.55:
            skipped.append(
                {
                    "target": column,
                    "metric": metric,
                    "train_on_real": round(baseline, 4),
                    "reason": (
                        "target is not predictable from the real data "
                        "(baseline AUC at chance), so it cannot measure utility"
                    ),
                }
            )
            continue
        if metric == "r2" and baseline < 0.05:
            skipped.append(
                {
                    "target": column,
                    "metric": metric,
                    "train_on_real": round(baseline, 4),
                    "reason": (
                        "target is not predictable from the real data "
                        "(baseline R2 at or below zero), so it cannot measure utility"
                    ),
                }
            )
            continue

        # For AUC, 0.5 is the no-skill floor; ratio against skill above chance
        # rather than against the raw number, which would flatter both models.
        if metric == "roc_auc":
            baseline_skill = max(baseline - 0.5, 1e-6)
            tstr_skill = max(tstr - 0.5, 0.0)
            ratio = tstr_skill / baseline_skill
        else:
            ratio = tstr / baseline if baseline > 0 else 0.0

        results.append(
            {
                "target": column,
                "task": task,
                "metric": metric,
                "train_on_real": round(baseline, 4),
                "train_on_synthetic": round(tstr, 4),
                "utility_ratio": round(float(np.clip(ratio, 0.0, 2.0)), 4),
                "n_real_train": int(len(real_y)),
                "n_synthetic_train": int(len(synth_y)),
                "n_test": int(len(test_y)),
            }
        )

    return {"scored": results, "not_measurable": skipped}


def run_detection(
    real: pd.DataFrame, synthetic: pd.DataFrame, *, exclude: set[str] | None = None
) -> dict[str, Any]:
    """Can a classifier tell real rows from synthetic ones? 0.5 AUC is ideal."""
    exclude = exclude or set()
    features = [
        column
        for column in real.columns
        if column in synthetic.columns and column not in exclude
        and (
            pd.api.types.is_numeric_dtype(real[column])
            or pd.api.types.is_datetime64_any_dtype(real[column])
            or real[column].nunique() <= MAX_FEATURE_CARDINALITY
        )
    ]
    if not features:
        return {"available": False, "reason": "no comparable feature columns"}

    size = min(len(real), len(synthetic), 20_000)
    if size < MIN_ROWS_FOR_EVALUATION:
        return {"available": False, "reason": "not enough rows"}

    real_sample = real.sample(size, random_state=0)
    synth_sample = synthetic.sample(size, random_state=0)

    x = pd.concat(
        [_prepare(real_sample, features, real), _prepare(synth_sample, features, real)],
        ignore_index=True,
    )
    y = np.concatenate([np.ones(size), np.zeros(size)])

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, random_state=0, stratify=y
    )
    model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=0, n_jobs=-1)
    model.fit(train_x, train_y)
    auc = float(roc_auc_score(test_y, model.predict_proba(test_x)[:, 1]))

    importances = sorted(
        zip(features, model.feature_importances_), key=lambda item: -item[1]
    )[:5]

    return {
        "available": True,
        "detection_auc": round(auc, 4),
        # 1.0 = indistinguishable, 0.0 = trivially separable.
        "indistinguishability": round(float(max(0.0, 1.0 - 2 * abs(auc - 0.5))), 4),
        "most_revealing_columns": [
            {"column": name, "importance": round(float(value), 4)}
            for name, value in importances
        ],
        "interpretation": (
            "AUC near 0.5 means a classifier cannot separate real from synthetic. "
            "Well above 0.5 means the synthetic data has a detectable signature; "
            "the listed columns are where it is strongest."
        ),
    }


def evaluate_tables(
    real_train: dict[str, pd.DataFrame],
    real_test: dict[str, pd.DataFrame],
    synthetic: dict[str, pd.DataFrame],
    *,
    protected: dict[str, set[str]] | None = None,
    targets: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Run TSTR and detection for every table."""
    protected = protected or {}
    targets = targets or {}
    tables: dict[str, Any] = {}
    ratios: list[float] = []
    aucs: list[float] = []

    for name, train_frame in real_train.items():
        test_frame = real_test.get(name)
        synth_frame = synthetic.get(name)
        if test_frame is None or synth_frame is None:
            continue
        # Belt and braces: drop anything that looks like a surrogate key even
        # if the caller did not list it. Synthetic keys are minted in a
        # different numeric range from real ones, which makes the detection
        # test trivially winnable and tells you nothing about the data.
        auto_keys = {
            column
            for column in train_frame.columns
            if train_frame[column].is_unique and len(train_frame) > MIN_ROWS_FOR_EVALUATION
        }
        protected[name] = set(protected.get(name, set())) | auto_keys
        if len(train_frame) < MIN_ROWS_FOR_EVALUATION or len(synth_frame) < MIN_ROWS_FOR_EVALUATION:
            tables[name] = {"skipped": "not enough rows to evaluate"}
            continue

        # Protected columns are placeholders by construction; including them
        # would make detection trivially easy and TSTR meaningless.
        exclude = set(protected.get(name, set()))

        tstr = run_tstr(
            train_frame, test_frame, synth_frame,
            targets=targets.get(name), exclude=exclude,
        )
        detection = run_detection(test_frame, synth_frame, exclude=exclude)

        tables[name] = {
            "tstr": tstr["scored"],
            "not_measurable": tstr["not_measurable"],
            "detection": detection,
        }
        ratios.extend(entry["utility_ratio"] for entry in tstr["scored"])
        if detection.get("available"):
            aucs.append(detection["detection_auc"])

    return {
        "tables": tables,
        "mean_utility_ratio": round(float(np.mean(ratios)), 4) if ratios else None,
        "mean_detection_auc": round(float(np.mean(aucs)), 4) if aucs else None,
        "targets_evaluated": len(ratios),
        "interpretation": (
            "utility_ratio is the headline number: 1.0 means a model trained on "
            "synthetic data performs as well on real data as one trained on real "
            "data. detection_auc near 0.5 means the synthetic rows are not "
            "distinguishable from real ones."
        ),
    }


__all__ = ["choose_targets", "evaluate_tables", "run_detection", "run_tstr"]
