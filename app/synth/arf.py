"""Adversarial Random Forests for tabular density estimation and generation.

Implements FORDE/FORGE (Watson et al., 2023, "Adversarial Random Forests for
Density Estimation and Generative Modeling") on scikit-learn, rather than taking
a dependency on `arfpy`. That library is MIT and would have been fine to use;
writing it out keeps the dependency surface at zero and leaves the internals
open for the DP work that would be needed to make this engine private.

The method, and why it suits a single wide table better than the SPN:

  1. Generate naive synthetic data by drawing each column independently from its
     own marginal. This deliberately destroys every interaction.
  2. Train a random forest to tell real rows from those synthetic rows. Wherever
     it succeeds, it has found an interaction the naive model missed -- the tree
     splits ARE the discovered structure.
  3. Resample synthetic data from within the forest's leaves, where the local
     independence assumption is much closer to true, and repeat.
  4. Stop when the forest can no longer separate the two (OOB accuracy at
     chance). That is a real convergence signal, measured on the same quantity a
     reviewer would check afterwards.

Contrast with the SPN, whose partition comes from KMeans plus chi-square tests
and whose `beta` is a blind knob: set it above the row count and the model
silently degenerates to independent columns. ARF has no equivalent trap.

  FORDE  after convergence, estimate a density per column within each leaf:
         truncated normal for continuous, multinomial for categorical.
  FORGE  to sample: pick a tree, pick a leaf by weight, draw each column from
         its within-leaf density.

DIFFERENTIAL PRIVACY IS NOT SUPPORTED HERE. The SPN's epsilon accounting works
because every learned parameter is a count. ARF learns split thresholds, which
are data-dependent values needing the exponential mechanism, and the adversarial
rounds multiply the budget. `SynthEngine.supports_dp` is False and the pipeline
refuses a private ARF run rather than quietly producing a non-private model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.synth.preprocess import CATEGORICAL, NON_STD, NULL_CODE, ColumnSpec

# Below this many rows in a leaf, the within-leaf marginals are noise; the leaf
# falls back to the column's global marginal.
MIN_LEAF_ROWS = 5


def _truncated_normal(
    mean: float, std: float, low: float, high: float, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Truncated normal by inverse CDF.

    scipy.stats.truncnorm.rvs is exact but carries enough per-call overhead to
    dominate here: a forest has thousands of leaves and most hold only a handful
    of rows, so this is called constantly with a tiny `count`. ndtr/ndtri are the
    same mathematics without the rv_continuous machinery.
    """
    from scipy.special import ndtr, ndtri

    lower = ndtr((low - mean) / std)
    upper = ndtr((high - mean) / std)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.clip(rng.normal(mean, std, count), low, high)
    uniform = rng.uniform(lower, upper, count)
    return mean + std * ndtri(np.clip(uniform, 1e-12, 1 - 1e-12))


@dataclass
class ARFParams:
    n_trees: int = 60
    max_rounds: int = 5
    # Convergence when the discriminator's OOB accuracy is within delta of 0.5.
    delta: float = 0.05
    # Leaf size drives both cost and resolution. Conditional sampling scores
    # every row against every leaf, so leaf count is the dominant term: on the
    # orders table, raising this from 20 to 60 cut leaves from 24,783 to 8,358
    # and conditional sampling from 22.9s to 6.7s for 20k rows, with no
    # measurable change in output quality. Lower it for small tables where
    # resolution matters more than speed.
    min_node_size: int = 20
    max_depth: int | None = None
    random_state: int = 0
    # Laplace-style smoothing on within-leaf category counts, so a category
    # absent from a leaf is unlikely rather than impossible.
    alpha: float = 0.5


@dataclass
class LeafDensity:
    """Per-column density parameters inside one leaf of one tree."""

    weight: float
    # column index -> parameters
    numeric: dict[int, tuple[float, float, float, float, float]] = field(default_factory=dict)
    categorical: dict[int, tuple[list[float], list[float], float]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "w": self.weight,
            "n": {str(k): list(v) for k, v in self.numeric.items()},
            "c": {str(k): [list(v[0]), list(v[1]), v[2]] for k, v in self.categorical.items()},
        }

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> LeafDensity:
        return cls(
            weight=blob["w"],
            numeric={int(k): tuple(v) for k, v in blob["n"].items()},
            categorical={
                int(k): (list(v[0]), list(v[1]), v[2]) for k, v in blob["c"].items()
            },
        )


class ARF:
    """Adversarial Random Forest density model over an encoded matrix."""

    def __init__(
        self,
        specs: list[ColumnSpec],
        params: ARFParams,
        leaves: list[dict[int, LeafDensity]] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ):
        self.specs = specs
        self.params = params
        # One dict per tree: leaf id -> LeafDensity
        self.leaves = leaves or []
        self.diagnostics = diagnostics or {}
        # Lazily built vectorised view; see _ensure_arrays.
        self._arrays: dict | None = None

    # -- helpers --

    @staticmethod
    def _impute(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
        """Fill nulls so the forest can be fitted.

        Nulls are modelled separately as a per-leaf, per-column probability, so
        this imputation only affects where a row lands in the tree, never the
        value that is eventually generated.
        """
        filled = matrix.copy()
        for column in range(filled.shape[1]):
            mask = np.isnan(filled[:, column])
            if mask.any():
                filled[mask, column] = medians[column]
        return filled

    @staticmethod
    def _marginal_sample(
        matrix: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Independent draw from each column's own marginal.

        This is the round-zero synthetic data. Sampling with replacement from
        the observed values keeps every marginal exactly right while destroying
        all joint structure, which is precisely what the discriminator should be
        able to detect.
        """
        out = np.empty((n, matrix.shape[1]), dtype=float)
        for column in range(matrix.shape[1]):
            out[:, column] = rng.choice(matrix[:, column], size=n, replace=True)
        return out

    # -- fit --

    @classmethod
    def fit(cls, matrix: np.ndarray, specs: list[ColumnSpec], params: ARFParams) -> ARF:
        rng = np.random.default_rng(params.random_state)
        n_rows, n_columns = matrix.shape
        if n_rows < 2 * MIN_LEAF_ROWS:
            raise ValueError(f"ARF needs more than {2 * MIN_LEAF_ROWS} rows, got {n_rows}")

        medians = np.nanmedian(np.where(np.isnan(matrix), np.nan, matrix), axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        real_filled = cls._impute(matrix, medians)

        # Round zero: every column drawn independently from its own marginal.
        synthetic = cls._marginal_sample(matrix, n_rows, rng)

        forest = None
        history: list[float] = []

        for round_index in range(max(params.max_rounds, 1)):
            synthetic_filled = cls._impute(synthetic, medians)
            x = np.vstack([real_filled, synthetic_filled])
            y = np.concatenate([np.ones(n_rows), np.zeros(n_rows)])

            forest = RandomForestClassifier(
                n_estimators=params.n_trees,
                min_samples_leaf=params.min_node_size,
                max_depth=params.max_depth,
                oob_score=True,
                bootstrap=True,
                n_jobs=-1,
                random_state=params.random_state + round_index,
            )
            forest.fit(x, y)
            accuracy = float(getattr(forest, "oob_score_", 0.5))
            history.append(accuracy)

            # Converged: the discriminator is guessing, so the synthetic data is
            # indistinguishable at the resolution this forest can see.
            if accuracy <= 0.5 + params.delta:
                break

            # Otherwise redraw the synthetic set from the leaves just learned,
            # where local independence is a far better approximation.
            interim = cls(specs, params)
            interim.leaves = interim._estimate_leaves(forest, real_filled, matrix)
            if not any(interim.leaves):
                break
            synthetic = interim.sample(n_rows, seed=int(rng.integers(0, 2**31 - 1)))

        model = cls(specs, params)
        model.leaves = model._estimate_leaves(forest, real_filled, matrix)
        model.diagnostics = {
            "rounds": len(history),
            "discriminator_oob_accuracy": history,
            "converged": bool(history and history[-1] <= 0.5 + params.delta),
            "final_accuracy": history[-1] if history else None,
            "trees": params.n_trees,
            "leaves_total": int(sum(len(tree) for tree in model.leaves)),
            "note": (
                "OOB accuracy near 0.5 means the discriminator cannot separate real "
                "from synthetic. Above ~0.6 after max_rounds means it has found "
                "structure the model still is not reproducing."
            ),
        }
        return model

    def _estimate_leaves(
        self, forest: RandomForestClassifier, real_filled: np.ndarray, raw: np.ndarray
    ) -> list[dict[int, LeafDensity]]:
        """FORDE: fit per-column densities inside every leaf, from REAL rows only."""
        per_tree: list[dict[int, LeafDensity]] = []
        n_rows = raw.shape[0]

        for estimator in forest.estimators_:
            leaf_ids = estimator.apply(real_filled)
            densities: dict[int, LeafDensity] = {}

            for leaf in np.unique(leaf_ids):
                rows = np.flatnonzero(leaf_ids == leaf)
                if rows.size == 0:
                    continue
                # Too few rows to estimate anything locally: fall back to the
                # whole column, which is worse but not noise.
                source = rows if rows.size >= MIN_LEAF_ROWS else np.arange(n_rows)
                density = LeafDensity(weight=float(rows.size) / n_rows)

                for index, spec in enumerate(self.specs):
                    values = raw[source, index]
                    if spec.kind == NON_STD:
                        continue
                    null_rate = float(np.mean(np.isnan(values)))
                    valid = values[~np.isnan(values)]

                    if spec.kind == CATEGORICAL:
                        codes = np.where(np.isnan(values), float(NULL_CODE), values)
                        support, counts = np.unique(codes, return_counts=True)
                        smoothed = counts.astype(float) + self.params.alpha
                        probabilities = smoothed / smoothed.sum()
                        density.categorical[index] = (
                            [float(v) for v in support],
                            [float(p) for p in probabilities],
                            null_rate,
                        )
                    else:
                        if valid.size == 0:
                            density.numeric[index] = (0.0, 0.0, 0.0, 0.0, 1.0)
                            continue
                        mean = float(np.mean(valid))
                        std = float(np.std(valid)) if valid.size > 1 else 0.0
                        density.numeric[index] = (
                            mean,
                            std,
                            float(np.min(valid)),
                            float(np.max(valid)),
                            null_rate,
                        )

                densities[int(leaf)] = density

            per_tree.append(densities)
        return per_tree

    # -- vectorised view over the leaves --
    #
    # The dict-of-LeafDensity form is convenient to build and to serialise, but
    # sampling from it in Python is far too slow: a conditional draw scores every
    # leaf against the evidence, and 34,000 child rows against 40,000 leaves is
    # hundreds of millions of interpreter operations. These arrays are built once
    # and let both sampling paths work in numpy.

    def _ensure_arrays(self) -> None:
        if getattr(self, "_arrays", None) is not None:
            return

        keys: list[tuple[int, int]] = []
        weights: list[float] = []
        for tree_index, tree in enumerate(self.leaves):
            for leaf, density in tree.items():
                keys.append((tree_index, leaf))
                weights.append(density.weight)
        if not keys:
            raise ValueError("ARF model has no leaves; was it fitted?")

        n_leaves = len(keys)
        n_columns = len(self.specs)

        # Numeric: (mean, std, low, high, null_rate) per leaf per column.
        numeric = np.full((n_leaves, n_columns, 5), np.nan, dtype=float)
        # Categorical: dense probability per declared category, plus a null slot
        # at index 0, so an evidence lookup is a single column index.
        widths = [
            (len(spec.categories) + 1) if spec.kind == CATEGORICAL else 0
            for spec in self.specs
        ]
        categorical = [
            np.zeros((n_leaves, width), dtype=float) if width else None
            for width in widths
        ]

        for position, (tree_index, leaf) in enumerate(keys):
            density = self.leaves[tree_index][leaf]
            for index, values in density.numeric.items():
                numeric[position, index, :] = values
            for index, (support, probabilities, null_rate) in density.categorical.items():
                table = categorical[index]
                if table is None:
                    continue
                for code, probability in zip(support, probabilities):
                    slot = 0 if int(round(code)) == NULL_CODE else int(round(code)) + 1
                    if 0 <= slot < table.shape[1]:
                        table[position, slot] += probability
                _ = null_rate

        weight_array = np.asarray(weights, dtype=float)
        self._arrays = {
            "keys": keys,
            "tree_of": np.array([k[0] for k in keys], dtype=int),
            "weights": weight_array / weight_array.sum(),
            "numeric": numeric,
            "categorical": categorical,
        }

    def _leaf_table(self) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Flat (tree, leaf) index with normalised selection weights."""
        keys: list[tuple[int, int]] = []
        weights: list[float] = []
        for tree_index, tree in enumerate(self.leaves):
            for leaf, density in tree.items():
                keys.append((tree_index, leaf))
                weights.append(density.weight)
        if not keys:
            raise ValueError("ARF model has no leaves; was it fitted?")
        array = np.asarray(weights, dtype=float)
        # Each tree contributes equally, so a tree with many leaves does not
        # dominate simply by being deeper.
        return keys, array / array.sum()

    def _draw_column(
        self, density: LeafDensity, index: int, rng: np.random.Generator
    ) -> float:
        spec = self.specs[index]
        if spec.kind == NON_STD:
            return 0.0

        if index in density.categorical:
            support, probabilities, null_rate = density.categorical[index]
            if null_rate > 0 and rng.random() < null_rate:
                return float(NULL_CODE)
            return float(rng.choice(support, p=probabilities))

        if index in density.numeric:
            mean, std, low, high, null_rate = density.numeric[index]
            if null_rate > 0 and rng.random() < null_rate:
                return float("nan")
            if std <= 0 or high <= low:
                return float(mean)
            # Truncated normal by rejection, bounded by the leaf's own range so a
            # generated value never leaves the region the leaf represents.
            for _ in range(12):
                value = rng.normal(mean, std)
                if low <= value <= high:
                    return float(value)
            return float(rng.uniform(low, high))

        return float("nan")

    def _draw_rows(
        self,
        picks: np.ndarray,
        rng: np.random.Generator,
        out: np.ndarray,
        skip: set[int] | None = None,
    ) -> None:
        """Fill `out` for every row, grouping rows that landed in the same leaf.

        Drawing row by row spends all its time in the interpreter. Rows sharing a
        leaf share a density, so they can be drawn in one numpy call each.
        """
        skip = skip or set()
        self._ensure_arrays()
        keys = self._arrays["keys"]

        order = np.argsort(picks, kind="stable")
        boundaries = np.flatnonzero(np.diff(picks[order])) + 1
        for rows in np.split(order, boundaries):
            if rows.size == 0:
                continue
            tree_index, leaf = keys[picks[rows[0]]]
            density = self.leaves[tree_index][leaf]
            count = rows.size

            for index, spec in enumerate(self.specs):
                if index in skip:
                    continue
                if spec.kind == NON_STD:
                    out[rows, index] = 0.0
                    continue

                if index in density.categorical:
                    support, probabilities, null_rate = density.categorical[index]
                    values = rng.choice(support, size=count, p=probabilities)
                    if null_rate > 0:
                        nulls = rng.random(count) < null_rate
                        values = np.where(nulls, float(NULL_CODE), values)
                    out[rows, index] = values
                    continue

                if index in density.numeric:
                    mean, std, low, high, null_rate = density.numeric[index]
                    if std <= 0 or high <= low:
                        values = np.full(count, mean, dtype=float)
                    else:
                        values = _truncated_normal(mean, std, low, high, count, rng)
                    if null_rate > 0:
                        nulls = rng.random(count) < null_rate
                        values = np.where(nulls, np.nan, values)
                    out[rows, index] = values

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """FORGE: pick a leaf by weight, draw each column from its density."""
        rng = np.random.default_rng(
            seed if seed is not None else self.params.random_state
        )
        keys, weights = self._leaf_table()
        out = np.full((n, len(self.specs)), np.nan, dtype=float)
        if n == 0:
            return out

        picks = rng.choice(len(keys), size=n, p=weights)
        self._draw_rows(picks, rng, out)
        return out

    # -- conditional sampling, so ARF can also be used for child tables --

    def _evidence_logprob(
        self, density: LeafDensity, evidence_row: np.ndarray, columns: list[int]
    ) -> float:
        total = 0.0
        for index in columns:
            if index in density.categorical:
                support, probabilities, null_rate = density.categorical[index]
                value = evidence_row[index]
                target = float(NULL_CODE) if np.isnan(value) else float(value)
                probability = 1e-12
                for candidate, p in zip(support, probabilities):
                    if abs(candidate - target) < 1e-9:
                        probability = p * (1.0 - null_rate) if target != NULL_CODE else max(null_rate, p)
                        break
                total += float(np.log(max(probability, 1e-12)))
            elif index in density.numeric:
                mean, std, low, high, null_rate = density.numeric[index]
                value = evidence_row[index]
                if np.isnan(value):
                    total += float(np.log(max(null_rate, 1e-12)))
                    continue
                if std <= 0:
                    total += 0.0 if abs(value - mean) < 1e-6 else -30.0
                    continue
                # Gaussian kernel, plus a hard penalty outside the leaf's range.
                z = (value - mean) / std
                total += float(-0.5 * z * z - np.log(std))
                if not (low <= value <= high):
                    total -= 10.0
        return total

    def _evidence_logprob_matrix(
        self, evidence: np.ndarray, columns: list[int]
    ) -> np.ndarray:
        """log P(evidence | leaf) for every row against every leaf.

        Shape (n_rows, n_leaves), computed column by column in numpy rather than
        leaf by leaf in Python.
        """
        self._ensure_arrays()
        numeric = self._arrays["numeric"]
        categorical = self._arrays["categorical"]
        n_rows = evidence.shape[0]
        n_leaves = numeric.shape[0]
        total = np.zeros((n_rows, n_leaves), dtype=float)

        for index in columns:
            spec = self.specs[index]
            values = evidence[:, index]

            if spec.kind == CATEGORICAL and categorical[index] is not None:
                table = categorical[index]
                slots = np.where(
                    np.isnan(values), 0, np.clip(values.astype(int) + 1, 0, table.shape[1] - 1)
                )
                total += np.log(np.maximum(table[:, slots].T, 1e-12))
                continue

            mean = numeric[:, index, 0]
            std = numeric[:, index, 1]
            low = numeric[:, index, 2]
            high = numeric[:, index, 3]
            null_rate = numeric[:, index, 4]

            usable = np.isfinite(mean)
            safe_std = np.where((std > 0) & usable, std, 1.0)

            z = (values[:, None] - mean[None, :]) / safe_std[None, :]
            density = -0.5 * z * z - np.log(safe_std)[None, :]
            # A leaf whose range excludes the value is penalised, not excluded:
            # the leaf boundaries come from the training rows, so a slightly
            # out-of-range value should be unlikely rather than impossible.
            outside = (values[:, None] < low[None, :]) | (values[:, None] > high[None, :])
            density = np.where(outside, density - 10.0, density)
            # Degenerate leaves (single value) match only near that value.
            degenerate = ~(std > 0) & usable
            exact = np.abs(values[:, None] - mean[None, :]) < 1e-6
            density = np.where(
                degenerate[None, :], np.where(exact, 0.0, -30.0), density
            )
            density = np.where(usable[None, :], density, -30.0)
            # Nulls in the evidence are scored against the leaf's null rate.
            null_score = np.log(np.maximum(null_rate, 1e-12))[None, :]
            density = np.where(np.isnan(values)[:, None], null_score, density)
            total += density

        return total

    def sample_conditional(
        self, evidence: np.ndarray, evidence_columns: list[int], seed: int | None = None
    ) -> np.ndarray:
        """Sample given fixed values for `evidence_columns`.

        Leaves are reweighted by how well they explain the evidence, the ARF
        analogue of the SPN's posterior over mixture components.
        """
        rng = np.random.default_rng(
            seed if seed is not None else self.params.random_state
        )
        n = evidence.shape[0]
        out = np.full((n, len(self.specs)), np.nan, dtype=float)
        if n == 0:
            return out

        columns = [c for c in evidence_columns if 0 <= c < len(self.specs)]
        if not columns:
            return self.sample(n, seed=seed)

        self._ensure_arrays()
        keys = self._arrays["keys"]
        prior = self._arrays["weights"]
        log_prior = np.log(np.maximum(prior, 1e-12))

        # Chunked so the (rows x leaves) score matrix stays a sensible size.
        chunk = max(1, min(n, int(4_000_000 / max(len(keys), 1)) or 1))
        picks = np.empty(n, dtype=int)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            scores = self._evidence_logprob_matrix(evidence[start:stop], columns)
            scores += log_prior[None, :]
            scores -= scores.max(axis=1, keepdims=True)
            posterior = np.exp(scores)
            totals = posterior.sum(axis=1, keepdims=True)
            posterior = np.where(totals > 0, posterior / np.where(totals > 0, totals, 1.0), prior)
            # Inverse-CDF draw per row, vectorised.
            cumulative = np.cumsum(posterior, axis=1)
            draws = rng.random(stop - start)[:, None] * cumulative[:, -1:]
            picks[start:stop] = (draws > cumulative).sum(axis=1).clip(0, len(keys) - 1)

        evidence_set = set(columns)
        self._draw_rows(picks, rng, out, skip=evidence_set)
        for index in columns:
            out[:, index] = evidence[:, index]
        return out

    # -- persistence --

    def to_json(self) -> dict[str, Any]:
        return {
            "engine": "ARF",
            "version": 1,
            "params": self.params.__dict__,
            "specs": [s.to_json() for s in self.specs],
            "diagnostics": self.diagnostics,
            "leaves": [
                {str(leaf): density.to_json() for leaf, density in tree.items()}
                for tree in self.leaves
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle)

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> ARF:
        specs = [ColumnSpec.from_json(s) for s in blob["specs"]]
        params = ARFParams(**blob["params"])
        leaves = [
            {int(leaf): LeafDensity.from_json(d) for leaf, d in tree.items()}
            for tree in blob["leaves"]
        ]
        return cls(specs, params, leaves=leaves, diagnostics=blob.get("diagnostics", {}))

    @classmethod
    def load(cls, path: str) -> ARF:
        with open(path, encoding="utf-8") as handle:
            return cls.from_json(json.load(handle))

    # -- parity with the SPN's reporting surface --

    @property
    def privacy(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "mechanism": "none",
            "reason": (
                "ARF learns data-dependent split thresholds across adversarial "
                "rounds; the count-based Laplace accounting used for the SPN does "
                "not apply. Use the SPN engine when differential privacy is "
                "required."
            ),
        }

    def count_nodes(self) -> dict[str, int]:
        return {
            "trees": len(self.leaves),
            "leaves": int(sum(len(tree) for tree in self.leaves)),
        }


__all__ = ["ARF", "ARFParams", "LeafDensity"]
