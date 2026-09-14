"""Column typing and encode/decode, driven by the schema.yaml parameters.

The SPN works on a purely numeric matrix. This module is the boundary: it
decides what each column *is* (categorical / numerical / datetime / timedelta /
non-std), encodes to that matrix, and inverts the transform after sampling.

Type inference mirrors the "Column Type Inference" group in schema.yaml
(`unique_threshold`, `max_n_category`, `force_min_category`), and can be
overridden per column by the configuration -- or, in our extension, by the
classification Apache Atlas holds for that column.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

CATEGORICAL = "categorical"
NUMERICAL = "numerical"
DATETIME = "datetime"
TIMEDELTA = "timedelta"
NON_STD = "non_std"

# Sentinel code reserved for "value was null" in categorical encodings.
NULL_CODE = -1

# Internal marker for the low-frequency tail of a high-cardinality column.
# Never emitted: decoding resamples from ColumnSpec.other_values instead.
OTHER_CATEGORY = "__other__"

# Format-preserving placeholder generators for columns the policy refuses to
# model. The value carries no information from the source; only its shape is
# preserved, so systems consuming the synthetic data still parse it.
_PLACEHOLDER_HINTS = (
    ("email", "email"),
    ("mail", "email"),
    ("phone", "phone"),
    ("mobile", "phone"),
    ("contact", "phone"),
    ("name", "name"),
    ("nric", "national_id"),
    ("national", "national_id"),
    ("passport", "national_id"),
    ("ic_no", "national_id"),
    ("address", "address"),
    ("addr", "address"),
)

_FAKE_FIRST = (
    "Aarav", "Priya", "Wei", "Siti", "John", "Mei", "Rahul", "Nurul", "David",
    "Ananya", "Hafiz", "Grace", "Kumar", "Lina", "Tan", "Farah", "Omar", "Yuki",
)
_FAKE_LAST = (
    "Rahman", "Chen", "Sharma", "Lim", "Tan", "Abdullah", "Nair", "Wong", "Lee",
    "Ibrahim", "Krishnan", "Menon", "Yusof", "Goh", "Das", "Ong", "Baker", "Sato",
)
_FAKE_STREET = ("Jalan Merdeka", "Orchard Road", "Park Avenue", "Bukit Timah", "High Street")


def stable_seed(*parts: Any) -> int:
    """Deterministic seed from a name.

    Python's built-in hash() is salted per process, so using it to seed an RNG
    means the same model with the same seed produces different output in every
    new process. CRC32 over the name is stable across processes and runs, which
    is what "regenerate with seed 7" has to mean.
    """
    payload = "␟".join(str(part) for part in parts).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def _placeholder_style_for(column_name: str) -> str:
    lowered = column_name.lower()
    for hint, style in _PLACEHOLDER_HINTS:
        if hint in lowered:
            return style
    return "token"


def _make_placeholders(spec: ColumnSpec, count: int, seed: int = 0) -> list[str]:
    """Generate unlinkable but realistically-shaped values."""
    rng = np.random.default_rng(seed)
    style = spec.placeholder_style

    if style == "name":
        return [
            f"{rng.choice(_FAKE_FIRST)} {rng.choice(_FAKE_LAST)}" for _ in range(count)
        ]
    if style == "email":
        # An email is an identifier, not a vocabulary word: a coincidental
        # collision with a real address could be mistaken for a real record.
        # A name-plus-small-number scheme gives only ~3M combinations, which
        # collides with a few thousand real addresses by birthday alone, so the
        # local part carries a 64-bit random token instead.
        return [
            f"{rng.choice(_FAKE_FIRST).lower()}.{rng.choice(_FAKE_LAST).lower()}"
            f".{rng.integers(0, 2**63):016x}@example.invalid"
            for _ in range(count)
        ]
    if style == "phone":
        return [f"+60{rng.integers(10**8, 10**9 - 1)}" for _ in range(count)]
    if style == "national_id":
        # Shaped like a Malaysian NRIC (YYMMDD-PB-NNNN) so downstream parsers
        # and validators still accept it. The digits carry nothing from the
        # source.
        return [
            f"{rng.integers(50, 99):02d}{rng.integers(1, 13):02d}{rng.integers(1, 29):02d}"
            f"-{rng.integers(1, 15):02d}-{rng.integers(1000, 9999)}"
            for _ in range(count)
        ]
    if style == "address":
        return [
            f"{rng.integers(1, 999)} {rng.choice(_FAKE_STREET)}, {rng.integers(10000, 99999)}"
            for _ in range(count)
        ]
    if style == "mask" and spec.placeholder_masks:
        return _make_masked(spec, count, rng)

    return [f"{spec.placeholder_prefix}{i}" for i in range(count)]


# Positions whose observed alphabet is a single character are reproduced
# literally. That is what carries a fixed prefix ("ACC...") and a leading zero
# across to the synthetic value, and it is deliberate: the format is the part we
# are preserving. It is the same trade-off the NRIC style makes by hardcoding
# YYMMDD-PB-NNNN, except measured instead of assumed.
_MASK_CLASSES = ((str.isdigit, "D"), (str.isupper, "A"), (str.islower, "a"))
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


def _mask_of(value: str) -> str:
    out = []
    for ch in value:
        for test, symbol in _MASK_CLASSES:
            if test(ch):
                out.append(symbol)
                break
        else:
            out.append(ch)
    return "".join(out)


def _learn_masks(series: Any, sample_size: int = 2000) -> list[dict[str, Any]]:
    """Describe a column's surface format: which characters occur where.

    Returns one entry per distinct mask, each holding the observed alphabet at
    every position. Generating from it preserves length, character classes, any
    constant prefix, and -- because a position that never held "0" cannot draw
    one -- the digit width of a numeric identifier once it is cast back to int.
    """
    try:
        import pandas as pd  # noqa: F401

        sample = series.dropna().astype(str).head(sample_size)
    except Exception:  # noqa: BLE001
        return []
    sample = [v for v in sample.tolist() if v]
    if not sample:
        return []

    by_mask: dict[str, list[str]] = {}
    for value in sample:
        by_mask.setdefault(_mask_of(value), []).append(value)

    # Two regimes, and the difference is what may be copied literally.
    #
    # A REGULAR column (few distinct masks: ACC0000000, a 10-digit account) has
    # a genuine format. A position that never varies is part of that format --
    # the "ACC" prefix, a leading zero -- and is reproduced.
    #
    # An IRREGULAR column (a name, an address, free text) has no format, only a
    # shape. Copying its constant positions would copy the source's own
    # characters back out, so nothing is held fixed except separators: every
    # letter and digit is redrawn from its full class. That is the enterprise
    # default -- random characters in the shape of the original -- and it does
    # not care what the column is called.
    regular = len(by_mask) <= max(8, len(sample) // 20)

    masks: list[dict[str, Any]] = []
    for mask, values in by_mask.items():
        alphabets = []
        for index, symbol in enumerate(mask):
            if symbol not in ("D", "A", "a"):
                # Separators and punctuation are structure, not content: a name
                # keeps its space, an email its "@".
                alphabets.append(symbol)
                continue
            if not regular:
                alphabets.append(
                    {"D": "0123456789", "A": _UPPER, "a": _LOWER}[symbol]
                )
                continue
            # Positions that vary get the full character class, not just the
            # characters seen there: a wider draw both enlarges the space the
            # value is drawn from and avoids copying the source's own alphabet
            # back out. A digit position that never held "0" keeps that
            # constraint, which is what preserves the width of an integer id.
            observed = {v[index] for v in values}
            # "Constant" must mean constant across the column, not across the
            # three rows that happen to share a rare shape -- otherwise those
            # rows' own characters are emitted verbatim. Below the support
            # floor the position is redrawn like any other.
            if len(observed) == 1 and len(values) >= max(20, 0.01 * len(sample)):
                alphabets.append("".join(observed))
                continue
            full = {"D": "0123456789", "A": _UPPER, "a": _LOWER}[symbol]
            if symbol == "D" and "0" not in observed:
                full = full[1:]
            alphabets.append(full)
        masks.append({"weight": len(values) / len(sample), "alphabets": alphabets})
    masks.sort(key=lambda m: -m["weight"])
    return masks


def _make_masked(spec: ColumnSpec, count: int, rng) -> list[str]:
    masks = spec.placeholder_masks
    weights = np.array([m["weight"] for m in masks], dtype=float)
    weights = weights / weights.sum()

    seen: set[str] = set()
    values: list[str] = []
    for _ in range(count):
        alphabets = masks[int(rng.choice(len(masks), p=weights))]["alphabets"]
        # Identifiers are normally unique, so retry on a collision rather than
        # emitting a duplicate. A narrow format can exhaust its space; give up
        # after a bounded number of tries instead of looping forever.
        for _attempt in range(20):
            candidate = "".join(
                alphabet
                if len(alphabet) == 1
                else alphabet[int(rng.integers(0, len(alphabet)))]
                for alphabet in alphabets
            )
            if candidate not in seen:
                break
        seen.add(candidate)
        values.append(candidate)
    return values


def _restore_placeholder_dtype(spec: "ColumnSpec", values: list[str]) -> Any:
    """Put a placeholder back in the source column's type.

    A protected int column (an account number) must still be an int in the
    output, or anything loading the synthetic CSV against the real schema fails
    on that column alone. Only integer dtypes are restored: a float or datetime
    identifier is not a format we generate, and a silent cast there would be a
    guess rather than a restoration.
    """
    import pandas as pd

    dtype = (spec.dtype_str or "").lower()
    if dtype.startswith(("int", "uint")) and all(v.isdigit() for v in values):
        try:
            return pd.Series(values, dtype="int64")
        except (ValueError, OverflowError):
            # Wider than int64 -- keep the digits as text rather than lose them.
            return values
    return values


@dataclass
class ColumnSpec:
    name: str
    kind: str
    categories: list[Any] = field(default_factory=list)
    null_ratio: float = 0.0
    # numerical / datetime
    min_value: float = 0.0
    max_value: float = 1.0
    dtype_str: str = ""
    datetime_format: str | None = None
    # non_std columns are replaced by a placeholder, per the `non_std_columns`
    # documentation in schema.yaml. We keep the *shape* of the original value
    # (an email still looks like an email) so downstream systems under test
    # still parse it, while the value itself carries no real information.
    placeholder_prefix: str = ""
    placeholder_style: str = "token"
    # Surface format of the source column, learned when no name hint applies:
    # one entry per distinct mask, each with the observed alphabet per position.
    # Without it an account number came back as "account_number_0" -- a string
    # where the source was an int, of the wrong width, in row order.
    placeholder_masks: list[dict[str, Any]] = field(default_factory=list)
    # "declared" when the range/categories came from the catalog or config,
    # "data" when they were measured from the training rows. Under DP only
    # "declared" gives an unconditional guarantee.
    domain_source: str = "data"
    # Granularity of the source column, preserved on output. Without this the
    # synthetic data carries a signature that any classifier can find: real
    # timestamps land on day boundaries while sampled ones have arbitrary
    # sub-second precision, and a source rounded to 2 decimals comes back with
    # 15. It is a fidelity issue rather than a privacy one, but it is the
    # single biggest giveaway in a real-vs-synthetic detection test.
    datetime_granularity_seconds: float = 0.0
    decimal_places: int | None = None
    # Declared nullability. Under DP the categorical support must be declared,
    # and including a null category in a column that is never null lets noise
    # invent nulls that do not exist in the source.
    nullable: bool = True
    # "none" or "log". A log transform is a declared modelling choice about the
    # column's shape, not a measurement of it, so it is safe under DP -- and it
    # is what makes DP usable on heavy-tailed money columns. Equal-width bins
    # over a raw log-normal range put nearly all mass in the first bin, and the
    # Laplace noise then scatters weight into tail bins spanning huge value
    # ranges, which moves the median by orders of magnitude.
    transform: str = "none"
    # Categories beyond max_n_category collapse into an "other" bucket; we keep
    # the tail so it can be resampled rather than emitting a literal marker.
    other_values: list[Any] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "categories": [_jsonable(c) for c in self.categories],
            "null_ratio": self.null_ratio,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "dtype_str": self.dtype_str,
            "datetime_format": self.datetime_format,
            "placeholder_prefix": self.placeholder_prefix,
            "placeholder_style": self.placeholder_style,
            "placeholder_masks": self.placeholder_masks,
            "other_values": [_jsonable(v) for v in self.other_values],
            "domain_source": self.domain_source,
            "transform": self.transform,
            "datetime_granularity_seconds": self.datetime_granularity_seconds,
            "decimal_places": self.decimal_places,
            "nullable": self.nullable,
        }

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> ColumnSpec:
        return cls(
            name=blob["name"],
            kind=blob["kind"],
            categories=blob.get("categories", []),
            null_ratio=blob.get("null_ratio", 0.0),
            min_value=blob.get("min_value", 0.0),
            max_value=blob.get("max_value", 1.0),
            dtype_str=blob.get("dtype_str", ""),
            datetime_format=blob.get("datetime_format"),
            placeholder_prefix=blob.get("placeholder_prefix", ""),
            placeholder_style=blob.get("placeholder_style", "token"),
            placeholder_masks=blob.get("placeholder_masks", []),
            other_values=blob.get("other_values", []),
            domain_source=blob.get("domain_source", "data"),
            transform=blob.get("transform", "none"),
            datetime_granularity_seconds=blob.get("datetime_granularity_seconds", 0.0),
            decimal_places=blob.get("decimal_places"),
            nullable=blob.get("nullable", True),
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def infer_column_kind(
    series: pd.Series,
    *,
    unique_threshold: float = 0.05,
    max_n_category: int = 50,
    force_min_category: int = 2,
) -> str:
    """Decide a column's type, following the Column Type Inference rules."""
    non_null = series.dropna()
    if non_null.empty:
        return CATEGORICAL

    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME
    if pd.api.types.is_timedelta64_dtype(series):
        return TIMEDELTA
    if pd.api.types.is_bool_dtype(series):
        return CATEGORICAL

    n_unique = int(non_null.nunique())
    n_rows = int(len(non_null))

    if pd.api.types.is_numeric_dtype(series):
        # Low-cardinality numerics (flags, codes, segment ids) behave as
        # categories, not as continuous quantities.
        if n_unique <= max(force_min_category, min(max_n_category, 20)) and (
            n_unique / max(n_rows, 1) < unique_threshold or n_unique <= 10
        ):
            return CATEGORICAL
        return NUMERICAL

    # Object dtype. Try datetime, but ONLY for genuinely string-like values.
    #
    # pd.to_datetime happily reads a bare integer as a nanosecond epoch, so a
    # nullable integer column -- which pandas stores as object as soon as it
    # contains a null -- would be classified as a datetime and come back as
    # 1970-01-01 00:00:00.000000003. Nullable integer columns are extremely
    # common in real extracts, so this check matters.
    string_like = non_null.map(lambda v: isinstance(v, str))
    if string_like.mean() > 0.95:
        candidates = non_null[string_like]
        # A string of pure digits is an identifier or a code, not a date.
        if not candidates.str.fullmatch(r"\s*\d+\s*").fillna(False).all():
            parsed = pd.to_datetime(candidates, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.95:
                return DATETIME
    else:
        # Mixed or numeric objects: fall back to the numeric rules.
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().mean() > 0.95:
            if n_unique <= max(force_min_category, min(max_n_category, 20)) and (
                n_unique / max(n_rows, 1) < unique_threshold or n_unique <= 10
            ):
                return CATEGORICAL
            return NUMERICAL

    if n_unique > max_n_category and (n_unique / max(n_rows, 1)) > 0.5:
        # High-cardinality free text / identifiers: no statistics to learn.
        return NON_STD
    return CATEGORICAL


class TableEncoder:
    """Encodes a DataFrame to a float matrix and back."""

    def __init__(self, specs: list[ColumnSpec]):
        self.specs = specs

    @property
    def columns(self) -> list[str]:
        return [s.name for s in self.specs]

    # -- fit --

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        overrides: dict[str, str] | None = None,
        placeholder_styles: dict[str, str] | None = None,
        unique_threshold: float = 0.05,
        max_n_category: int = 50,
        force_min_category: int = 2,
        declared_domain: dict[str, dict[str, Any]] | None = None,
        apply_declared_transform: bool = False,
    ) -> TableEncoder:
        """Fit encoders.

        `declared_domain` supplies column ranges and category lists from a
        source that is public knowledge (the data catalog or the configuration),
        keyed by column name:

            {"balance": {"min": 0, "max": 1_000_000},
             "segment": {"categories": ["MASS", "AFFLUENT", ...]}}

        Anything not declared is measured from the data instead. That is fine
        for ordinary runs, but under differential privacy a measured domain is
        an unbudgeted release -- see app.synth.dp. The caller is told which
        columns fell back, via `domain_sources`.
        """
        overrides = overrides or {}
        placeholder_styles = placeholder_styles or {}
        declared_domain = declared_domain or {}
        specs: list[ColumnSpec] = []
        for name in frame.columns:
            series = frame[name]
            kind = overrides.get(name) or infer_column_kind(
                series,
                unique_threshold=unique_threshold,
                max_n_category=max_n_category,
                force_min_category=force_min_category,
            )
            spec = ColumnSpec(name=name, kind=kind, dtype_str=str(series.dtype))
            spec.null_ratio = float(series.isna().mean())

            declared = declared_domain.get(name) or {}
            if declared.get("nullable") is not None:
                spec.nullable = bool(declared["nullable"])
            else:
                spec.nullable = bool(series.isna().any())

            if kind == CATEGORICAL:
                declared_categories = declared.get("categories")
                if declared_categories:
                    spec.categories = list(declared_categories)
                    spec.domain_source = "declared"
                else:
                    cats = series.dropna().unique().tolist()
                    # Cap runaway cardinality; the tail becomes an "other"
                    # bucket. The tail values are retained so that decoding can
                    # resample from them -- emitting the literal "__other__"
                    # marker into the output would be a visible defect.
                    if len(cats) > max_n_category:
                        counts = series.dropna().value_counts()
                        cats = counts.index[: max_n_category - 1].tolist()
                        spec.other_values = counts.index[max_n_category - 1 :].tolist()
                        cats = cats + [OTHER_CATEGORY]
                    spec.categories = cats
                    spec.domain_source = "data"
            elif kind in (NUMERICAL, DATETIME, TIMEDELTA):
                if (
                    apply_declared_transform
                    and declared.get("transform") == "log"
                    and kind == NUMERICAL
                ):
                    # Only used under DP. The transform exists so that
                    # equal-width bins over a declared range can represent a
                    # heavy tail. Quantile bins (the non-private path) are
                    # already scale-free, so applying it there buys nothing and
                    # costs accuracy: sampling uniformly inside a log-space bin
                    # and inverting with expm1 biases values toward the low end
                    # of the bin.
                    spec.transform = "log"

                declared_min = declared.get("min")
                declared_max = declared.get("max")
                if declared_min is not None and declared_max is not None:
                    low = _coerce_bound(declared_min, kind)
                    high = _coerce_bound(declared_max, kind)
                    spec.domain_source = "declared"
                else:
                    numeric = _to_numeric(series, kind)
                    valid = numeric[~np.isnan(numeric)]
                    if valid.size:
                        low, high = float(np.min(valid)), float(np.max(valid))
                    else:
                        low, high = 0.0, 1.0
                    spec.domain_source = "data"

                if kind == DATETIME:
                    spec.datetime_granularity_seconds = _infer_datetime_granularity(series)
                elif kind == NUMERICAL:
                    spec.decimal_places = _infer_decimal_places(series)

                # Bounds are stored in the space the model works in.
                spec.min_value = float(_apply_transform(np.array([low]), spec.transform)[0])
                spec.max_value = float(_apply_transform(np.array([high]), spec.transform)[0])
                if spec.max_value <= spec.min_value:
                    spec.max_value = spec.min_value + 1e-9
            elif kind == NON_STD:
                spec.placeholder_prefix = f"{name}_"
                # Order of authority: an explicit setting, then a curated
                # catalog tag, then the column's own values, and only then the
                # spelling of its name.
                #
                # The values outrank the name deliberately. A column called
                # "mobile" holding "+65 9xxxxxxx" was being reissued by the
                # built-in phone generator as "+60xxxxxxxxx" -- a different
                # country, in a different format, from a column whose real
                # format was sitting right there. A learned shape cannot be
                # wrong about the data in that way.
                declared = placeholder_styles.get(name)
                if declared:
                    spec.placeholder_style = declared
                else:
                    masks = _learn_masks(series)
                    if masks:
                        spec.placeholder_masks = masks
                        spec.placeholder_style = "mask"
                    else:
                        # Nothing readable to learn from (an all-null column).
                        spec.placeholder_style = _placeholder_style_for(name)

            specs.append(spec)
        return cls(specs)

    # -- transform --

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        cols: list[np.ndarray] = []
        for spec in self.specs:
            series = frame[spec.name] if spec.name in frame.columns else pd.Series(
                [np.nan] * len(frame)
            )
            if spec.kind == CATEGORICAL:
                lookup = {c: i for i, c in enumerate(spec.categories)}
                other = lookup.get("__other__", NULL_CODE)
                encoded = series.map(
                    lambda v: NULL_CODE if pd.isna(v) else lookup.get(v, other)
                ).astype(float)
                cols.append(encoded.to_numpy(dtype=float))
            elif spec.kind == NON_STD:
                # Not modelled; emit a constant so the column keeps its slot.
                cols.append(np.zeros(len(frame), dtype=float))
            else:
                cols.append(_apply_transform(_to_numeric(series, spec.kind), spec.transform))
        if not cols:
            return np.zeros((len(frame), 0), dtype=float)
        return np.column_stack(cols)

    # -- inverse --

    def inverse_transform(self, matrix: np.ndarray) -> pd.DataFrame:
        data: dict[str, Any] = {}
        for index, spec in enumerate(self.specs):
            column = matrix[:, index]
            if spec.kind == CATEGORICAL:
                rng = np.random.default_rng(stable_seed(spec.name, "other"))
                values: list[Any] = []
                for code in column:
                    if np.isnan(code) or int(round(code)) == NULL_CODE:
                        values.append(None)
                        continue
                    position = int(round(code))
                    if 0 <= position < len(spec.categories):
                        category = spec.categories[position]
                        if category == OTHER_CATEGORY:
                            # Resample from the retained tail rather than
                            # emitting the internal marker.
                            if spec.other_values:
                                category = spec.other_values[
                                    int(rng.integers(0, len(spec.other_values)))
                                ]
                            else:
                                category = None
                        values.append(category)
                    else:
                        values.append(None)
                data[spec.name] = values
            elif spec.kind == NON_STD:
                values = _make_placeholders(
                    spec, matrix.shape[0], seed=stable_seed(spec.name, "placeholder")
                )
                data[spec.name] = _restore_placeholder_dtype(spec, values)
            elif spec.kind == DATETIME:
                seconds = np.where(np.isnan(column), np.nan, column)
                # Snap back onto the source's grid: day-aligned data must stay
                # day-aligned, or every synthetic row is identifiable by its
                # stray time component alone.
                grid = spec.datetime_granularity_seconds
                if grid and grid > 0:
                    seconds = np.where(
                        np.isnan(seconds), np.nan, np.round(seconds / grid) * grid
                    )
                data[spec.name] = pd.to_datetime(seconds, unit="s", errors="coerce")
            elif spec.kind == TIMEDELTA:
                data[spec.name] = pd.to_timedelta(
                    np.where(np.isnan(column), np.nan, column), unit="s"
                )
            else:
                values = column.astype(float)
                # Clip in model space, then invert the declared transform.
                values = np.clip(values, spec.min_value, spec.max_value)
                values = _invert_transform(values, spec.transform)
                if _looks_integral(spec.dtype_str):
                    values = np.round(values)
                elif spec.decimal_places is not None:
                    values = np.round(values, spec.decimal_places)

                if _looks_integral(spec.dtype_str):
                    # Restore the integer dtype. Emitting 462.0 where the source
                    # had 462 is a schema change, and anything downstream that
                    # reads the column as an int will either fail or silently
                    # coerce.
                    if np.isnan(values).any():
                        data[spec.name] = pd.array(values, dtype="Float64").astype("Int64")
                    else:
                        data[spec.name] = values.astype("int64")
                else:
                    data[spec.name] = values
        return pd.DataFrame(data)

    # -- domain provenance --

    def domain_summary(self) -> dict[str, Any]:
        """Which columns have a declared domain, and which were measured."""
        relevant = [s for s in self.specs if s.kind != NON_STD]
        measured = [s.name for s in relevant if s.domain_source != "declared"]
        return {
            "all_declared": not measured,
            "declared": [s.name for s in relevant if s.domain_source == "declared"],
            "measured_from_data": measured,
        }

    # -- persistence --

    def to_json(self) -> list[dict[str, Any]]:
        return [s.to_json() for s in self.specs]

    @classmethod
    def from_json(cls, blob: list[dict[str, Any]]) -> TableEncoder:
        return cls([ColumnSpec.from_json(b) for b in blob])


# Candidate grids, coarsest first. Real extracts almost always land on one.
_GRANULARITIES = (86400.0, 3600.0, 60.0, 1.0)


def _infer_datetime_granularity(series: pd.Series) -> float:
    """Coarsest grid the source timestamps all sit on (0 = continuous)."""
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return 0.0
    seconds = valid.astype("int64") / 1e9
    for grid in _GRANULARITIES:
        if np.allclose(np.mod(seconds, grid), 0.0, atol=1e-6):
            return grid
    return 0.0


def _infer_decimal_places(series: pd.Series, max_places: int = 6) -> int | None:
    """Fewest decimal places that reproduce the source values exactly."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    sample = numeric.to_numpy()[:5000]
    for places in range(0, max_places + 1):
        if np.allclose(sample, np.round(sample, places), rtol=0, atol=1e-9):
            return places
    return None


def _apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform != "log":
        return values
    # log1p keeps zero at zero and is defined on the whole non-negative range.
    return np.log1p(np.clip(values, 0.0, None))


def _invert_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform != "log":
        return values
    return np.expm1(values)


def _coerce_bound(value: Any, kind: str) -> float:
    """Declared bounds may be written as dates or numbers; normalise to float."""
    if kind == DATETIME:
        return float(pd.Timestamp(value).timestamp())
    if kind == TIMEDELTA:
        return float(pd.Timedelta(value).total_seconds())
    return float(value)


def _looks_integral(dtype_str: str) -> bool:
    return dtype_str.startswith("int") or dtype_str.startswith("uint")


def _to_numeric(series: pd.Series, kind: str) -> np.ndarray:
    if kind == DATETIME:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        return (parsed.astype("int64") / 1e9).where(parsed.notna(), np.nan).to_numpy(
            dtype=float
        )
    if kind == TIMEDELTA:
        parsed = pd.to_timedelta(series, errors="coerce")
        return (parsed.dt.total_seconds()).to_numpy(dtype=float)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


__all__ = [
    "CATEGORICAL",
    "DATETIME",
    "NON_STD",
    "NULL_CODE",
    "NUMERICAL",
    "TIMEDELTA",
    "ColumnSpec",
    "TableEncoder",
    "infer_column_kind",
]
