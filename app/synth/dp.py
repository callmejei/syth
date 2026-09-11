"""Differential privacy accounting and mechanisms for the SPN.

The previous build privatised only leaf parameters and sum weights, and said so.
Closing that gap surfaced three data-dependent steps, not one:

  1. STRUCTURE   clustering (KMeans) and the pairwise independence tests both
                 read raw rows to decide the shape of the tree.
  2. NODE SIZE   the `n_rows < beta` test that decides SUM vs PRODUCT.
  3. DOMAIN      histogram bin edges came from data quantiles, and category
                 lists from observed values. This is the subtle one: releasing
                 "the categories present in this column" can directly disclose
                 that a rare value exists at all, which is exactly the kind of
                 leak DP is supposed to prevent. It is not fixed by adding
                 noise to counts, because the *support* itself is the leak.

All three are handled here. (3) is handled by requiring the domain to be public
input rather than learned -- which is where the Atlas integration pays off a
second time, since the catalog already declares column types, ranges and
allowed values. When a domain is not declared, the caller must explicitly
acknowledge that it will be derived from data, and the report records the
guarantee as conditional.

Composition
-----------
* PARALLEL across siblings: nodes at the same depth own disjoint row subsets, so
  they share that level's budget rather than each consuming it.
* SEQUENTIAL across depth: a row passes through one node per level, so levels
  add up.
* SEQUENTIAL across columns for leaf parameters: every row contributes to every
  column's leaf, so the per-column budget divides.
* PARALLEL across leaves of one column: leaves partition the rows.

Total: epsilon_structure + epsilon_parameters = epsilon_total, with
epsilon_structure spread over max_depth levels and epsilon_parameters over the
column count. Pure epsilon-DP throughout (Laplace only, delta = 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Fraction of the budget spent on learning the tree shape. The remainder pays
# for leaf parameters. Structure is cheap to get approximately right and
# expensive to get exactly right, so it takes the smaller share.
DEFAULT_STRUCTURE_FRACTION = 0.35

# Data-independent constants, safe to hard-code (they leak nothing).
KMEANS_ITERATIONS = 3
CLUSTER_FEATURE_CLIP = 1.0


@dataclass
class LedgerEntry:
    stage: str
    mechanism: str
    epsilon: float
    count: int = 0


@dataclass
class PrivacyLedger:
    """Tracks and enforces the epsilon budget across every data-dependent step."""

    epsilon_total: float
    enabled: bool
    n_columns: int
    max_depth: int
    structure_fraction: float = DEFAULT_STRUCTURE_FRACTION
    domain_is_public: bool = False
    domain_note: str = ""
    entries: dict[str, LedgerEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_columns = max(int(self.n_columns), 1)
        self.max_depth = max(int(self.max_depth), 1)

    # -- allocation --

    @property
    def epsilon_structure(self) -> float:
        return self.epsilon_total * self.structure_fraction if self.enabled else 0.0

    @property
    def epsilon_parameters(self) -> float:
        return self.epsilon_total - self.epsilon_structure if self.enabled else 0.0

    @property
    def epsilon_per_level(self) -> float:
        """Structure budget for one depth level (parallel across siblings)."""
        return self.epsilon_structure / self.max_depth

    @property
    def epsilon_per_column(self) -> float:
        """Leaf-parameter budget for one column (parallel across its leaves)."""
        return self.epsilon_parameters / self.n_columns

    def _level_share(self, share: float) -> float:
        """Split a level's budget between the three structure decisions.

        At one node we ask: how many rows are here, which columns are
        dependent, and how do the rows cluster. Those queries hit the same rows,
        so they compose sequentially and must divide the level budget.
        """
        return self.epsilon_per_level * share

    # -- bookkeeping --

    def _record(self, stage: str, mechanism: str, epsilon: float) -> None:
        entry = self.entries.get(stage)
        if entry is None:
            self.entries[stage] = LedgerEntry(stage, mechanism, epsilon, 1)
        else:
            entry.count += 1
            # Parallel composition within a stage: the epsilon does not grow
            # with the number of disjoint queries, so keep the max rather than
            # summing.
            entry.epsilon = max(entry.epsilon, epsilon)

    # -- mechanisms --

    def laplace_counts(
        self,
        counts: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
        stage: str,
        sensitivity: float = 1.0,
    ) -> np.ndarray:
        """Laplace mechanism on a histogram of counts."""
        counts = np.asarray(counts, dtype=float)
        if not self.enabled or epsilon <= 0:
            return counts
        noisy = counts + rng.laplace(0.0, sensitivity / epsilon, size=counts.shape)
        # Clipping is post-processing and does not weaken the guarantee.
        noisy = np.clip(noisy, 0.0, None)
        self._record(stage, f"Laplace(sensitivity={sensitivity})", epsilon)
        if noisy.sum() <= 0:
            noisy = np.ones_like(noisy)
        return noisy

    def noisy_node_size(
        self, n_rows: int, rng: np.random.Generator
    ) -> float:
        """Privatised row count, for the `n_rows < beta` decision.

        Adding or removing one row changes the count by 1.
        """
        if not self.enabled:
            return float(n_rows)
        epsilon = self._level_share(0.2)
        if epsilon <= 0:
            return float(n_rows)
        noisy = float(n_rows) + rng.laplace(0.0, 1.0 / epsilon)
        self._record("node_size", "Laplace(sensitivity=1)", epsilon)
        return max(noisy, 0.0)

    def noisy_contingency(
        self, table: np.ndarray, n_pairs: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Privatised contingency table for an independence test.

        Replacing one row moves it out of one cell and into another, so the L1
        sensitivity of the table is 2.
        """
        table = np.asarray(table, dtype=float)
        if not self.enabled:
            return table
        epsilon = self._level_share(0.4) / max(n_pairs, 1)
        if epsilon <= 0:
            return table
        noisy = table + rng.laplace(0.0, 2.0 / epsilon, size=table.shape)
        noisy = np.clip(noisy, 0.0, None)
        self._record("independence_test", "Laplace(sensitivity=2)", epsilon)
        return noisy

    def dp_kmeans(
        self,
        features: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Differentially private Lloyd's algorithm.

        Features must be clipped to [-1, 1]^d so that sensitivity is bounded.
        Each iteration releases, per cluster, a d-vector of coordinate sums plus
        a count. Moving one row between clusters changes two clusters' sums by
        at most |x|_1 <= d each and two counts by 1 each, giving an L1
        sensitivity of 2*(d + 1) for the whole release.

        Centroid initialisation is uniform random, which is data-independent and
        therefore free.
        """
        n_rows, n_features = features.shape
        if n_rows < k or k < 2:
            return None

        if not self.enabled:
            from sklearn.cluster import KMeans

            try:
                return KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(features)
            except Exception:  # noqa: BLE001
                return None

        epsilon = self._level_share(0.4)
        if epsilon <= 0:
            return None
        epsilon_per_iteration = epsilon / KMEANS_ITERATIONS
        sensitivity = 2.0 * (n_features + 1)
        scale = sensitivity / epsilon_per_iteration

        clipped = np.clip(features, -CLUSTER_FEATURE_CLIP, CLUSTER_FEATURE_CLIP)
        centroids = rng.uniform(-1.0, 1.0, size=(k, n_features))

        labels = np.zeros(n_rows, dtype=int)
        for _ in range(KMEANS_ITERATIONS):
            distances = ((clipped[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(distances, axis=1)

            for cluster in range(k):
                mask = labels == cluster
                noisy_sum = clipped[mask].sum(axis=0) + rng.laplace(0.0, scale, n_features)
                noisy_count = float(mask.sum()) + rng.laplace(0.0, scale)
                if noisy_count < 1.0:
                    # Too few points to trust; re-seed this centroid at random.
                    centroids[cluster] = rng.uniform(-1.0, 1.0, n_features)
                else:
                    centroids[cluster] = np.clip(noisy_sum / noisy_count, -1.0, 1.0)

            self._record("clustering", f"Laplace(sensitivity={sensitivity:.0f})", epsilon)

        if np.unique(labels).size < 2:
            return None
        return labels

    def leaf_epsilon(self) -> float:
        return self.epsilon_per_column

    # -- reporting --

    def report(self) -> dict[str, Any]:
        spent_structure = sum(
            entry.epsilon
            for stage, entry in self.entries.items()
            if stage in ("node_size", "independence_test", "clustering")
        ) * (self.max_depth if self.enabled else 0)
        return {
            "enabled": self.enabled,
            "mechanism": "Laplace only (pure epsilon-DP, delta = 0)",
            "epsilon_total": self.epsilon_total if self.enabled else 0.0,
            "allocation": {
                "structure": self.epsilon_structure,
                "parameters": self.epsilon_parameters,
                "structure_per_level": self.epsilon_per_level,
                "parameters_per_column": self.epsilon_per_column,
            },
            "covers": [
                "leaf parameters (categorical and histogram counts)",
                "sum-node weights",
                "node size decisions (SUM vs PRODUCT)",
                "independence tests (contingency tables)",
                "row clustering (DP Lloyd's)",
            ],
            "stages": {
                stage: {
                    "mechanism": entry.mechanism,
                    "epsilon_per_query": entry.epsilon,
                    "queries": entry.count,
                }
                for stage, entry in self.entries.items()
            },
            "structure_epsilon_accounted": min(spent_structure, self.epsilon_structure)
            if self.enabled
            else 0.0,
            "domain_is_public": self.domain_is_public,
            "domain_note": self.domain_note,
            "composition": (
                "Parallel composition across siblings at a depth and across the "
                "leaves of one column; sequential across depth levels and across "
                "columns. Budget is allocated up front, so the stated epsilon is "
                "an upper bound regardless of the tree that is learned."
            ),
            "caveat": (
                ""
                if self.domain_is_public
                else (
                    "CONDITIONAL GUARANTEE: column ranges and category lists were "
                    "derived from the training data rather than declared, so the "
                    "released domain itself is not covered by the epsilon budget. "
                    "Declare the domain (from Atlas or the configuration) for an "
                    "unconditional guarantee."
                )
            ),
        }


__all__ = ["DEFAULT_STRUCTURE_FRACTION", "LedgerEntry", "PrivacyLedger"]
