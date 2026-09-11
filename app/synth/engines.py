"""Engine registry.

The relational layer needs one thing from a density model: fit it on an encoded
matrix, then sample from it, optionally conditioned on some columns. Anything
providing that can be a table model, so the choice of engine is a lookup rather
than a branch scattered through the pipeline.

Two engines:

    SPN   Sum-Product Network. Supports differential privacy, because every
          learned parameter is a count.
    ARF   Adversarial Random Forest. Better at interactions in a single wide
          table, and it knows when it has converged. No DP.

The `supports_dp` flag is load-bearing: the pipeline refuses a private run on an
engine that cannot deliver it, rather than silently returning a non-private model
from a request that asked for privacy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from app.synth.arf import ARF, ARFParams
from app.synth.preprocess import ColumnSpec
from app.synth.spn import SPN, SPNParams


class TableModel(Protocol):
    """What the relational layer requires of a per-table density model."""

    def sample(self, n: int, seed: int | None = ...) -> np.ndarray: ...

    def sample_conditional(
        self, evidence: np.ndarray, evidence_columns: list[int], seed: int | None = ...
    ) -> np.ndarray: ...

    def save(self, path: str) -> None: ...


@dataclass
class Engine:
    name: str
    supports_dp: bool
    supports_conditional: bool
    description: str

    def build_params(self, config: dict[str, Any], random_state: int = 0) -> Any:
        raise NotImplementedError

    def fit(self, matrix: np.ndarray, specs: list[ColumnSpec], params: Any) -> TableModel:
        raise NotImplementedError

    def load(self, path: str) -> TableModel:
        raise NotImplementedError

    def privacy_of(self, model: Any) -> dict[str, Any]:
        return getattr(model, "privacy", {"enabled": False})

    def stats_of(self, model: Any) -> dict[str, Any]:
        counter = getattr(model, "count_nodes", None)
        if callable(counter):
            return counter()
        root = getattr(model, "root", None)
        return root.count_nodes() if root is not None else {}


class SPNEngine(Engine):
    def __init__(self) -> None:
        super().__init__(
            name="SPN",
            supports_dp=True,
            supports_conditional=True,
            description=(
                "Sum-Product Network. Differential privacy supported; `beta` must "
                "be well below the row count or the model degenerates to "
                "independent columns."
            ),
        )

    def build_params(self, config: dict[str, Any], random_state: int = 0) -> SPNParams:
        return SPNParams(
            beta=int(config.get("beta", 100_000)),
            private=bool(config.get("private", False)),
            epsilon=float(config.get("epsilon", 2.0)),
            p_value_threshold=float(config.get("p_value_threshold", 0.05)),
            max_clusters=int(config.get("max_clusters", 4)),
            max_depth=int(config.get("max_depth", 4)),
            random_state=int(config.get("random_state", random_state)),
        )

    def fit(self, matrix: np.ndarray, specs: list[ColumnSpec], params: SPNParams) -> SPN:
        return SPN.fit(matrix, specs, params)

    def load(self, path: str) -> SPN:
        return SPN.load(path)


class ARFEngine(Engine):
    def __init__(self) -> None:
        super().__init__(
            name="ARF",
            supports_dp=False,
            supports_conditional=True,
            description=(
                "Adversarial Random Forest. Stronger on interactions within a "
                "single wide table and reports its own convergence; no "
                "differential privacy."
            ),
        )

    def build_params(self, config: dict[str, Any], random_state: int = 0) -> ARFParams:
        return ARFParams(
            n_trees=int(config.get("n_trees", 60)),
            max_rounds=int(config.get("max_rounds", 5)),
            delta=float(config.get("delta", 0.05)),
            min_node_size=int(config.get("min_node_size", 20)),
            max_depth=config.get("max_depth"),
            random_state=int(config.get("random_state", random_state)),
            alpha=float(config.get("alpha", 0.5)),
        )

    def fit(self, matrix: np.ndarray, specs: list[ColumnSpec], params: ARFParams) -> ARF:
        return ARF.fit(matrix, specs, params)

    def load(self, path: str) -> ARF:
        return ARF.load(path)


_REGISTRY: dict[str, Engine] = {
    "SPN": SPNEngine(),
    "ARF": ARFEngine(),
    # The vendor's screen lists ARF2; treat it as an alias so a configuration
    # copied from their UI resolves rather than failing.
    "ARF2": ARFEngine(),
}


def get_engine(name: str | None) -> Engine:
    key = (name or "SPN").upper()
    engine = _REGISTRY.get(key)
    if engine is None:
        available = ", ".join(sorted({e.name for e in _REGISTRY.values()}))
        raise ValueError(f"unknown engine '{name}'. Available: {available}")
    return engine


def engine_names() -> list[str]:
    return sorted({engine.name for engine in _REGISTRY.values()})


def config_key(engine_name: str) -> str:
    """Where this engine's parameters live in training_params."""
    return "arf_config" if get_engine(engine_name).name == "ARF" else "spn_config"


__all__ = ["ARFEngine", "Engine", "SPNEngine", "config_key", "engine_names", "get_engine"]
