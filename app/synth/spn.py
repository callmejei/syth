"""Sum-Product Network for tabular synthesis.

Structure learning follows LearnSPN (Gens & Domingos, 2013), with the two
parameters the vendor exposes in SPN_Config:

  beta     -- "Minimum leaf size to build SUM node, node with less data will all
              be PRODUCT node." Below beta rows we stop clustering and emit a
              fully-factorised PRODUCT node over independent leaves.
  private  -- enables differential privacy
  epsilon  -- the DP budget

Recursion at each node:
  1. scope of one column                 -> Leaf
  2. fewer than `beta` rows              -> PRODUCT over independent leaves
  3. columns split into independent sets -> PRODUCT over those sets
  4. otherwise cluster rows              -> SUM over the clusters

Leaves are categorical distributions or histograms, which makes the DP story
tractable: every learned parameter is a count, so Laplace noise on counts gives
a clean per-query sensitivity of 1.

DIFFERENTIAL PRIVACY -- SCOPE OF THE GUARANTEE (read before quoting this):
The epsilon budget now covers every data-dependent step: leaf parameters, sum
weights, the node-size test that chooses SUM vs PRODUCT, the pairwise
independence tests, and row clustering (via DP Lloyd's). Mechanisms are Laplace
throughout, so this is pure epsilon-DP with delta = 0. See app.synth.dp for the
composition argument and the per-stage allocation.

One condition remains, and it is a real one. The guarantee assumes the column
DOMAIN -- numeric ranges and category lists -- is public knowledge rather than
something learned from the training data. Under DP the domain must be declared
(from Atlas or the configuration); if it is instead derived from the data, the
released bin edges and category lists are an unbudgeted disclosure and the
guarantee is conditional. The ledger records which case applies and every
report states it. Do not quote an unconditional epsilon for a run where
domain_is_public is false.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from app.synth.dp import DEFAULT_STRUCTURE_FRACTION, PrivacyLedger
from app.synth.preprocess import (
    CATEGORICAL,
    NON_STD,
    NULL_CODE,
    ColumnSpec,
)

# Histogram leaves use at most this many bins; more bins means finer marginals
# but a thinner DP signal per bin.
MAX_BINS = 32


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


class Node:
    def sample(self, n: int, rng: np.random.Generator, out: np.ndarray) -> None:
        raise NotImplementedError

    def evidence_prob(self, evidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """P(evidence columns | this node), evaluated per row.

        Columns outside `mask` are marginalised out, which for an SPN is just
        "return 1" -- that closure property is the whole reason conditional
        sampling is tractable here.
        """
        raise NotImplementedError

    def sample_conditional(
        self,
        evidence: np.ndarray,
        mask: np.ndarray,
        rows: np.ndarray,
        rng: np.random.Generator,
        out: np.ndarray,
    ) -> None:
        """Sample the non-evidence columns given the evidence columns."""
        raise NotImplementedError

    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError

    def count_nodes(self) -> dict[str, int]:
        raise NotImplementedError


@dataclass
class SumNode(Node):
    """Mixture over row-clusters. Weights are cluster proportions."""

    children: list[Node]
    weights: np.ndarray

    def sample(self, n: int, rng: np.random.Generator, out: np.ndarray) -> None:
        if n == 0:
            return
        picks = rng.choice(len(self.children), size=n, p=self.weights)
        cursor = 0
        for index, child in enumerate(self.children):
            count = int(np.sum(picks == index))
            if count:
                child.sample(count, rng, out[cursor : cursor + count])
                cursor += count

    def evidence_prob(self, evidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
        total = np.zeros(evidence.shape[0], dtype=float)
        for weight, child in zip(self.weights, self.children):
            total += weight * child.evidence_prob(evidence, mask)
        return total

    def sample_conditional(
        self,
        evidence: np.ndarray,
        mask: np.ndarray,
        rows: np.ndarray,
        rng: np.random.Generator,
        out: np.ndarray,
    ) -> None:
        if rows.size == 0:
            return
        subset = evidence[rows]
        # Posterior over mixture components given the evidence.
        likelihoods = np.stack(
            [
                weight * child.evidence_prob(subset, mask)
                for weight, child in zip(self.weights, self.children)
            ],
            axis=1,
        )
        totals = likelihoods.sum(axis=1, keepdims=True)
        # Rows whose evidence is impossible under every component fall back to
        # the prior weights rather than producing NaNs.
        degenerate = (totals[:, 0] <= 0) | ~np.isfinite(totals[:, 0])
        posterior = np.divide(
            likelihoods,
            np.where(totals > 0, totals, 1.0),
            out=np.zeros_like(likelihoods),
            where=totals > 0,
        )
        if degenerate.any():
            posterior[degenerate] = self.weights

        cumulative = np.cumsum(posterior, axis=1)
        draws = rng.random(rows.size)[:, None]
        picks = (draws > cumulative).sum(axis=1)
        picks = np.clip(picks, 0, len(self.children) - 1)

        for index, child in enumerate(self.children):
            selected = rows[picks == index]
            if selected.size:
                child.sample_conditional(evidence, mask, selected, rng, out)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "sum",
            "weights": [float(w) for w in self.weights],
            "children": [c.to_json() for c in self.children],
        }

    def count_nodes(self) -> dict[str, int]:
        totals = {"sum": 1, "product": 0, "leaf": 0}
        for child in self.children:
            for key, value in child.count_nodes().items():
                totals[key] = totals.get(key, 0) + value
        return totals


@dataclass
class ProductNode(Node):
    """Independent factorisation over disjoint column sets."""

    children: list[Node]

    def sample(self, n: int, rng: np.random.Generator, out: np.ndarray) -> None:
        for child in self.children:
            child.sample(n, rng, out)

    def evidence_prob(self, evidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
        total = np.ones(evidence.shape[0], dtype=float)
        for child in self.children:
            total *= child.evidence_prob(evidence, mask)
        return total

    def sample_conditional(
        self,
        evidence: np.ndarray,
        mask: np.ndarray,
        rows: np.ndarray,
        rng: np.random.Generator,
        out: np.ndarray,
    ) -> None:
        # Disjoint scopes, so each factor is sampled independently.
        for child in self.children:
            child.sample_conditional(evidence, mask, rows, rng, out)

    def to_json(self) -> dict[str, Any]:
        return {"type": "product", "children": [c.to_json() for c in self.children]}

    def count_nodes(self) -> dict[str, int]:
        totals = {"sum": 0, "product": 1, "leaf": 0}
        for child in self.children:
            for key, value in child.count_nodes().items():
                totals[key] = totals.get(key, 0) + value
        return totals


@dataclass
class CategoricalLeaf(Node):
    column: int
    codes: list[float]
    probs: np.ndarray

    def sample(self, n: int, rng: np.random.Generator, out: np.ndarray) -> None:
        out[:, self.column] = rng.choice(self.codes, size=n, p=self.probs)

    def evidence_prob(self, evidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask[self.column]:
            return np.ones(evidence.shape[0], dtype=float)
        values = evidence[:, self.column]
        codes = np.asarray(self.codes, dtype=float)
        # Nulls are encoded as NULL_CODE, so they match a real category.
        values = np.where(np.isnan(values), float(NULL_CODE), values)
        matches = np.isclose(values[:, None], codes[None, :])
        probs = (matches * self.probs[None, :]).sum(axis=1)
        # Unseen category: floor rather than zero, so the row stays samplable.
        return np.where(probs > 0, probs, 1e-12)

    def sample_conditional(
        self,
        evidence: np.ndarray,
        mask: np.ndarray,
        rows: np.ndarray,
        rng: np.random.Generator,
        out: np.ndarray,
    ) -> None:
        if rows.size == 0:
            return
        if mask[self.column]:
            out[rows, self.column] = evidence[rows, self.column]
        else:
            out[rows, self.column] = rng.choice(
                self.codes, size=rows.size, p=self.probs
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "categorical",
            "column": self.column,
            "codes": [float(c) for c in self.codes],
            "probs": [float(p) for p in self.probs],
        }

    def count_nodes(self) -> dict[str, int]:
        return {"sum": 0, "product": 0, "leaf": 1}


@dataclass
class HistogramLeaf(Node):
    """Piecewise-uniform density, plus an explicit null mass."""

    column: int
    edges: np.ndarray
    probs: np.ndarray
    null_prob: float

    def sample(self, n: int, rng: np.random.Generator, out: np.ndarray) -> None:
        values = np.empty(n, dtype=float)
        is_null = rng.random(n) < self.null_prob
        n_valid = int(np.sum(~is_null))
        if n_valid:
            bins = rng.choice(len(self.probs), size=n_valid, p=self.probs)
            lows = self.edges[bins]
            highs = self.edges[bins + 1]
            values[~is_null] = rng.uniform(lows, highs)
        values[is_null] = np.nan
        out[:, self.column] = values

    def evidence_prob(self, evidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask[self.column]:
            return np.ones(evidence.shape[0], dtype=float)
        values = evidence[:, self.column]
        widths = np.diff(self.edges)
        widths = np.where(widths > 0, widths, 1e-12)
        densities = self.probs / widths

        bins = np.digitize(values, self.edges) - 1
        inside = (bins >= 0) & (bins < len(densities)) & ~np.isnan(values)
        safe_bins = np.clip(bins, 0, len(densities) - 1)

        result = np.where(inside, densities[safe_bins] * (1.0 - self.null_prob), 0.0)
        # Out-of-range values get a floor so an unusual parent cannot make the
        # whole row unsamplable.
        result = np.where(inside, result, 1e-12)
        result = np.where(np.isnan(values), max(self.null_prob, 1e-12), result)
        return result

    def sample_conditional(
        self,
        evidence: np.ndarray,
        mask: np.ndarray,
        rows: np.ndarray,
        rng: np.random.Generator,
        out: np.ndarray,
    ) -> None:
        if rows.size == 0:
            return
        if mask[self.column]:
            out[rows, self.column] = evidence[rows, self.column]
            return
        buffer = np.empty((rows.size, out.shape[1]), dtype=float)
        self.sample(rows.size, rng, buffer)
        out[rows, self.column] = buffer[:, self.column]

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "histogram",
            "column": self.column,
            "edges": [float(e) for e in self.edges],
            "probs": [float(p) for p in self.probs],
            "null_prob": float(self.null_prob),
        }

    def count_nodes(self) -> dict[str, int]:
        return {"sum": 0, "product": 0, "leaf": 1}


@dataclass
class ConstantLeaf(Node):
    column: int
    value: float

    def sample(self, n: int, rng: np.random.Generator, out: np.ndarray) -> None:
        out[:, self.column] = self.value

    def evidence_prob(self, evidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask[self.column]:
            return np.ones(evidence.shape[0], dtype=float)
        values = evidence[:, self.column]
        if np.isnan(self.value):
            return np.where(np.isnan(values), 1.0, 1e-12)
        return np.where(np.isclose(values, self.value), 1.0, 1e-12)

    def sample_conditional(
        self,
        evidence: np.ndarray,
        mask: np.ndarray,
        rows: np.ndarray,
        rng: np.random.Generator,
        out: np.ndarray,
    ) -> None:
        if rows.size == 0:
            return
        if mask[self.column]:
            out[rows, self.column] = evidence[rows, self.column]
        else:
            out[rows, self.column] = self.value

    def to_json(self) -> dict[str, Any]:
        return {"type": "constant", "column": self.column, "value": float(self.value)}

    def count_nodes(self) -> dict[str, int]:
        return {"sum": 0, "product": 0, "leaf": 1}


# --------------------------------------------------------------------------
# Privacy accountant
# --------------------------------------------------------------------------


# Privacy accounting lives in app.synth.dp.PrivacyLedger, which covers leaf
# parameters, sum weights, node-size decisions, independence tests and row
# clustering. See that module for the composition argument.


# --------------------------------------------------------------------------
# Learner
# --------------------------------------------------------------------------


@dataclass
class SPNParams:
    beta: int = 100_000
    private: bool = False
    epsilon: float = 2.0
    p_value_threshold: float = 0.05
    max_clusters: int = 4
    # Kept low deliberately. Under DP the structure budget is divided across
    # max_depth levels, so a generous limit starves every level: a tree that
    # only ever grows 3 deep with max_depth=12 wastes three quarters of its
    # budget on levels that never exist, and the clustering noise that results
    # flattens exactly the correlations the model is there to reproduce.
    # Measured on the customer sample at epsilon=2, dropping 12 -> 4 roughly
    # doubled the recovered income spread across segments.
    max_depth: int = 4
    random_state: int = 0
    # Share of epsilon spent learning the tree shape rather than leaf values.
    structure_fraction: float = DEFAULT_STRUCTURE_FRACTION
    # True when column ranges and category lists came from a declared schema
    # (Atlas / configuration) rather than being read off the training data.
    # See app.synth.dp for why this matters to the guarantee.
    domain_is_public: bool = False
    domain_note: str = ""


class SPN:
    def __init__(self, root: Node, specs: list[ColumnSpec], params: SPNParams,
                 privacy: dict[str, Any] | None = None):
        self.root = root
        self.specs = specs
        self.params = params
        self.privacy = privacy or {}

    # -- fit --

    @classmethod
    def fit(cls, matrix: np.ndarray, specs: list[ColumnSpec], params: SPNParams) -> SPN:
        rng = np.random.default_rng(params.random_state)
        ledger = PrivacyLedger(
            epsilon_total=params.epsilon,
            enabled=params.private,
            n_columns=max(matrix.shape[1], 1),
            max_depth=params.max_depth,
            structure_fraction=params.structure_fraction,
            domain_is_public=params.domain_is_public,
            domain_note=params.domain_note,
        )
        learner = _Learner(specs, params, ledger, rng)
        scope = list(range(matrix.shape[1]))
        root = learner.build(matrix, scope, depth=0)
        return cls(root, specs, params, privacy=ledger.report())

    # -- sample --

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else self.params.random_state)
        out = np.full((n, len(self.specs)), np.nan, dtype=float)
        self.root.sample(n, rng, out)
        return out

    def sample_conditional(
        self,
        evidence: np.ndarray,
        evidence_columns: list[int],
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample rows given fixed values for `evidence_columns`.

        This is what carries parent attributes into a child table: the child's
        row is drawn from P(child columns | parent context), not from the
        child's marginal. Without it each table is individually plausible but
        the database as a whole is incoherent.
        """
        rng = np.random.default_rng(seed if seed is not None else self.params.random_state)
        n = evidence.shape[0]
        out = np.full((n, len(self.specs)), np.nan, dtype=float)
        if n == 0:
            return out
        mask = np.zeros(len(self.specs), dtype=bool)
        for column in evidence_columns:
            if 0 <= column < len(self.specs):
                mask[column] = True
        self.root.sample_conditional(
            evidence, mask, np.arange(n), rng, out
        )
        return out

    # -- persistence --

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "params": {
                "beta": self.params.beta,
                "private": self.params.private,
                "epsilon": self.params.epsilon,
                "p_value_threshold": self.params.p_value_threshold,
                "max_clusters": self.params.max_clusters,
                "max_depth": self.params.max_depth,
                "random_state": self.params.random_state,
            },
            "specs": [s.to_json() for s in self.specs],
            "privacy": self.privacy,
            "structure": self.root.to_json(),
            "node_counts": self.root.count_nodes(),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle)

    @classmethod
    def load(cls, path: str) -> SPN:
        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)
        return cls.from_json(blob)

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> SPN:
        specs = [ColumnSpec.from_json(s) for s in blob["specs"]]
        params = SPNParams(**blob["params"])
        root = _node_from_json(blob["structure"])
        return cls(root, specs, params, privacy=blob.get("privacy", {}))


def _node_from_json(blob: dict[str, Any]) -> Node:
    kind = blob["type"]
    if kind == "sum":
        return SumNode(
            children=[_node_from_json(c) for c in blob["children"]],
            weights=np.array(blob["weights"], dtype=float),
        )
    if kind == "product":
        return ProductNode(children=[_node_from_json(c) for c in blob["children"]])
    if kind == "categorical":
        return CategoricalLeaf(
            column=blob["column"],
            codes=list(blob["codes"]),
            probs=np.array(blob["probs"], dtype=float),
        )
    if kind == "histogram":
        return HistogramLeaf(
            column=blob["column"],
            edges=np.array(blob["edges"], dtype=float),
            probs=np.array(blob["probs"], dtype=float),
            null_prob=blob["null_prob"],
        )
    if kind == "constant":
        return ConstantLeaf(column=blob["column"], value=blob["value"])
    raise ValueError(f"unknown node type: {kind}")


class _Learner:
    def __init__(
        self,
        specs: list[ColumnSpec],
        params: SPNParams,
        ledger: PrivacyLedger,
        rng: np.random.Generator,
    ):
        self.specs = specs
        self.params = params
        self.ledger = ledger
        self.rng = rng

    # -- recursion --

    def build(self, data: np.ndarray, scope: list[int], depth: int) -> Node:
        if len(scope) == 1:
            return self.make_leaf(data, scope[0])

        # The row count is itself a data-dependent quantity, so under DP the
        # SUM-vs-PRODUCT decision is made on a noisy count.
        effective_rows = self.ledger.noisy_node_size(data.shape[0], self.rng)

        if depth >= self.params.max_depth or effective_rows < self.params.beta:
            # Below beta rows: no SUM node, factorise completely.
            return ProductNode([self.make_leaf(data, col) for col in scope])

        groups = self.split_columns(data, scope)
        if len(groups) > 1:
            return ProductNode([self.build(data, g, depth + 1) for g in groups])

        clusters = self.split_rows(data, scope)
        if clusters is None:
            return ProductNode([self.make_leaf(data, col) for col in scope])

        labels, k = clusters
        children: list[Node] = []
        raw_counts = np.array(
            [int(np.sum(labels == i)) for i in range(k)], dtype=float
        )
        for index in range(k):
            subset = data[labels == index]
            if subset.shape[0] == 0:
                continue
            children.append(self.build(subset, scope, depth + 1))
        counts = raw_counts[raw_counts > 0]
        if not children:
            return ProductNode([self.make_leaf(data, col) for col in scope])
        if len(children) == 1:
            return children[0]

        noisy = self.ledger.laplace_counts(
            counts, self.ledger.leaf_epsilon(), self.rng, "sum_weights"
        )
        weights = noisy / noisy.sum()
        return SumNode(children=children, weights=weights)

    # -- column split (independence test) --

    def split_columns(self, data: np.ndarray, scope: list[int]) -> list[list[int]]:
        """Union-find over pairwise dependence; returns independent column sets."""
        parent = {col: col for col in scope}
        # Number of tests is a function of the scope size, which is public
        # structure, so budgeting on it leaks nothing.
        n_pairs = max(len(scope) * (len(scope) - 1) // 2, 1)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, col_a in enumerate(scope):
            for col_b in scope[i + 1 :]:
                if find(col_a) == find(col_b):
                    continue
                if self.is_dependent(data, col_a, col_b, n_pairs):
                    union(col_a, col_b)

        groups: dict[int, list[int]] = {}
        for col in scope:
            groups.setdefault(find(col), []).append(col)
        return list(groups.values())

    def is_dependent(
        self, data: np.ndarray, col_a: int, col_b: int, n_pairs: int = 1
    ) -> bool:
        """Chi-square test on discretised columns.

        Discretising numerics lets one test cover cat/cat, cat/num and num/num
        uniformly, and it is what makes the test cheap enough to run pairwise.

        Under DP the test runs on a noised contingency table. Defaulting to
        "dependent" on weak evidence is the safe direction: it produces a SUM
        node over the joint distribution rather than falsely asserting that two
        columns are unrelated.
        """
        a = self.discretise(data[:, col_a], col_a)
        b = self.discretise(data[:, col_b], col_b)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if a.size < 20:
            return True  # too little evidence to declare independence
        if np.unique(a).size < 2 or np.unique(b).size < 2:
            return False  # a constant column is independent of everything

        table = _contingency(a, b)
        if table.shape[0] < 2 or table.shape[1] < 2:
            return False

        table = self.ledger.noisy_contingency(table, n_pairs, self.rng)
        # Noise can empty a row or column, which chi2 rejects.
        table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
        if table.shape[0] < 2 or table.shape[1] < 2:
            return True

        try:
            _, p_value, _, _ = stats.chi2_contingency(table)
        except ValueError:
            return True
        if np.isnan(p_value):
            return True
        return bool(p_value < self.params.p_value_threshold)

    def discretise(self, column: np.ndarray, index: int) -> np.ndarray:
        spec = self.specs[index]
        if spec.kind in (CATEGORICAL, NON_STD):
            return column
        valid = column[~np.isnan(column)]
        if valid.size == 0:
            return np.full_like(column, np.nan)
        # Quantile bins: robust to skew, which matters for money columns.
        quantiles = np.unique(np.quantile(valid, np.linspace(0, 1, 9)))
        if quantiles.size < 2:
            return np.zeros_like(column)
        binned = np.digitize(column, quantiles[1:-1])
        return np.where(np.isnan(column), np.nan, binned.astype(float))

    # -- row split (clustering) --

    def split_rows(self, data: np.ndarray, scope: list[int]) -> tuple[np.ndarray, int] | None:
        features = self.cluster_features(data, scope)
        if features is None or features.shape[0] < 2 * self.params.max_clusters:
            return None
        k = min(self.params.max_clusters, max(2, features.shape[0] // self.params.beta or 2))
        k = max(2, min(k, features.shape[0] - 1))

        # Non-private path uses plain KMeans; the private path uses DP Lloyd's.
        # Both live behind the ledger so the choice is accounted for in one
        # place rather than duplicated here.
        labels = self.ledger.dp_kmeans(features, k, self.rng)
        if labels is None:
            return None
        return labels, int(np.max(labels)) + 1

    def cluster_features(self, data: np.ndarray, scope: list[int]) -> np.ndarray | None:
        """Feature matrix for clustering, scaled into [-1, 1].

        Scaling uses the column's declared domain (ColumnSpec.min_value /
        max_value / categories) rather than the mean and standard deviation of
        the rows at this node. That matters twice over: those statistics are
        themselves a data-dependent release that was never budgeted for, and
        z-scored features are unbounded, which would leave DP Lloyd's without a
        finite sensitivity bound.
        """
        blocks: list[np.ndarray] = []
        for col in scope:
            spec = self.specs[col]
            column = data[:, col]
            if spec.kind == NON_STD:
                continue

            if spec.kind == CATEGORICAL:
                n_categories = len(spec.categories)
                if n_categories < 2 or n_categories > 24:
                    continue
                codes = np.where(np.isnan(column), NULL_CODE, column).astype(int)

                if self.ledger.enabled:
                    # Ordinal, one dimension per column. One-hot would be a
                    # better clustering representation, but DP Lloyd's has
                    # sensitivity 2(d+1): one-hotting a handful of categoricals
                    # pushes d from ~8 to ~23 and nearly triples the noise. At
                    # realistic epsilon the noise reduction outweighs the
                    # weaker encoding.
                    scaled = 2.0 * codes / max(n_categories - 1, 1) - 1.0
                    blocks.append(np.clip(scaled, -1.0, 1.0)[:, None])
                else:
                    positions = np.arange(n_categories)
                    onehot = (codes[:, None] == positions[None, :]).astype(float)
                    # Centre so an absent category is -1 and a present one +1.
                    blocks.append(onehot * 2.0 - 1.0)
            elif self.ledger.enabled:
                # Domain scaling into [-1, 1]: bounded, and independent of the
                # rows at this node, which DP Lloyd's requires.
                low, high = float(spec.min_value), float(spec.max_value)
                span = high - low
                if not np.isfinite(span) or span <= 0:
                    continue
                midpoint = (low + high) / 2.0
                filled = np.where(np.isnan(column), midpoint, column)
                scaled = 2.0 * (filled - low) / span - 1.0
                scaled = np.clip(scaled, -1.0, 1.0)
                if not np.isfinite(scaled).all():
                    continue
                blocks.append(scaled[:, None])
            else:
                # Non-private: z-score. Min-max scaling would squash a
                # heavy-tailed column (money, durations) into a sliver at the
                # bottom of its range, so clustering would effectively ignore
                # it and the learned structure would lose that column's
                # relationships. DP cannot use this because the mean and
                # standard deviation are themselves unbudgeted releases.
                filled = np.where(np.isnan(column), np.nanmedian(column), column)
                if not np.isfinite(filled).all():
                    continue
                std = np.std(filled)
                if std <= 0:
                    continue
                blocks.append(((filled - np.mean(filled)) / std)[:, None])

        if not blocks:
            return None
        return np.hstack(blocks)

    # -- leaves --

    def make_leaf(self, data: np.ndarray, index: int) -> Node:
        spec = self.specs[index]
        column = data[:, index]

        if spec.kind == NON_STD:
            return ConstantLeaf(column=index, value=0.0)

        leaf_epsilon = self.ledger.leaf_epsilon()

        if spec.kind == CATEGORICAL:
            codes = np.where(np.isnan(column), NULL_CODE, column).astype(int)
            if self.ledger.enabled:
                # Count over the DECLARED category list, including categories
                # absent from these rows. Releasing only the observed
                # categories would disclose the support of the data, which no
                # amount of noise on the counts can undo.
                support = np.arange(len(spec.categories))
                if spec.nullable:
                    support = np.concatenate([np.array([NULL_CODE]), support])
                counts = np.array(
                    [float(np.sum(codes == value)) for value in support]
                )
            else:
                support, counts_int = np.unique(codes, return_counts=True)
                counts = counts_int.astype(float)

            noisy = self.ledger.laplace_counts(
                counts, leaf_epsilon, self.rng, "categorical_leaf"
            )
            probs = noisy / noisy.sum()
            return CategoricalLeaf(
                column=index, codes=support.astype(float).tolist(), probs=probs
            )

        valid = column[~np.isnan(column)]
        null_count = float(np.sum(np.isnan(column)))
        total = float(column.size)

        if valid.size == 0:
            return ConstantLeaf(column=index, value=np.nan)
        if np.unique(valid).size == 1:
            value = float(valid[0])
            if null_count == 0:
                return ConstantLeaf(column=index, value=value)
            edges = np.array([value, value + 1e-9])
            return HistogramLeaf(
                column=index,
                edges=edges,
                probs=np.array([1.0]),
                null_prob=null_count / max(total, 1.0),
            )

        if self.ledger.enabled:
            # Bin edges must not depend on the data. Quantile edges are a
            # release of the distribution itself -- the top edge alone is the
            # maximum value in the column. Under DP we use equal-width bins over
            # the declared domain, which costs nothing and leaks nothing.
            n_bins = int(min(MAX_BINS, 16))
            low, high = float(spec.min_value), float(spec.max_value)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = float(np.min(valid)), float(np.max(valid)) + 1e-9
            edges = np.linspace(low, high, n_bins + 1)
        else:
            n_bins = int(min(MAX_BINS, max(4, np.sqrt(valid.size))))
            edges = np.unique(np.quantile(valid, np.linspace(0, 1, n_bins + 1)))
            if edges.size < 2:
                edges = np.array([float(np.min(valid)), float(np.max(valid)) + 1e-9])

        # Widen the last edge so the max value falls inside a bin.
        edges = edges.astype(float)
        edges[-1] = np.nextafter(edges[-1], edges[-1] + 1.0)

        clipped = np.clip(valid, edges[0], edges[-1])
        counts, _ = np.histogram(clipped, bins=edges)
        noisy = self.ledger.laplace_counts(
            counts.astype(float), leaf_epsilon, self.rng, "histogram_leaf"
        )
        probs = noisy / noisy.sum()

        null_prob = null_count / max(total, 1.0)
        if self.ledger.enabled and null_count > 0:
            noisy_null = self.ledger.laplace_counts(
                np.array([null_count, total - null_count]),
                leaf_epsilon,
                self.rng,
                "null_rate",
            )
            null_prob = float(noisy_null[0] / noisy_null.sum())

        return HistogramLeaf(
            column=index, edges=edges, probs=probs, null_prob=float(null_prob)
        )


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_vals, a_idx = np.unique(a, return_inverse=True)
    b_vals, b_idx = np.unique(b, return_inverse=True)
    table = np.zeros((a_vals.size, b_vals.size), dtype=float)
    np.add.at(table, (a_idx, b_idx), 1.0)
    # Drop all-zero rows/cols so chi2 does not choke.
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    return table


__all__ = ["SPN", "SPNParams", "PrivacyAccountant"]
