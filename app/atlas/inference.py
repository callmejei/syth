"""Fallback column classification for when the catalog is silent.

The policy engine is driven by curated Atlas classifications. That is the point
of the product, and it stays the authority. But a catalog that has never been
asked about a table returns nothing, and the failure mode is severe: an
untagged `full_name` column falls through to MODEL, gets learned as a
categorical, and the generator emits real customers' names verbatim.

So this module is a floor, not a replacement. It guesses a column's nature from
its name and its values, and the policy engine uses the guess **only** where the
catalog has no opinion. Every decision that came from here is labelled as
inferred, so a reviewer can always tell a curated tag from a guess -- a policy
built on guesses must never be presented as one built on your catalog.

**Scope: direct identifiers only.** OCBC's catalog uses a single tag, `PII`, so
that is the only distinction the platform infers. Columns that merely *look*
sensitive or quasi-identifying -- income, credit score, date of birth, postcode
-- are modelled normally and left alone.

That is a deliberate narrowing. Inferring a quasi-identifier changes the shape
of otherwise-safe data on the strength of a column name: `date_of_birth` was
being coarsened to a year, and nobody had asked for it. Inferring a direct
identifier only ever replaces a value that must not be published anyway, so the
cost of a false positive is far lower. The catalog remains free to ask for
GENERALISE or MODEL_DP explicitly -- policy.py still honours those tags -- but
inference will not invent them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Inferred tags carry this prefix so they can never be confused with a curated
# Atlas classification in a report or an audit record.
INFERRED = "INFERRED"

# Strategy groups. The policy engine maps these directly rather than re-deriving
# them from tag names: ColumnClassification.is_direct_identifier matches a fixed
# set of curated tag strings, which an INFERRED:-prefixed tag would never hit.
GROUP_DIRECT = "DIRECT_IDENTIFIER"


@dataclass
class Inference:
    """A guess about one column, with the justification that produced it."""

    group: str
    tags: list[str] = field(default_factory=list)
    reason: str = ""


def _tag(*names: str) -> list[str]:
    return [f"{INFERRED}:{n}" for n in names]


# --------------------------------------------------------------------------
# name patterns
#
# Matched against a normalised column name (lowercase, non-alphanumerics to
# underscore), so `Date_Of_Birth`, `dateOfBirth` and `date-of-birth` all match.
# --------------------------------------------------------------------------

_DIRECT_IDENTIFIER = [
    (r"(^|_)(full|first|last|given|middle|sur)?_?names?($|_)", "NAME"),
    (r"(^|_)(email|e_mail|mail_address)($|_)", "EMAIL"),
    (r"(^|_)(phone|mobile|telephone|tel|msisdn|contact_no)", "PHONE"),
    (r"(national_id|nric|fin_no|ic_no|passport|ssn|social_security|tax_id|uen)", "NATIONAL_ID"),
    (r"(^|_)(street|address_line|home_address|mailing_address|addr)", "ADDRESS"),
    (r"(^|_)(username|user_name|login|userid_name)($|_)", "USERNAME"),
    (r"(^|_)(iban|swift|bank_account_no|account_number|acct_no)($|_)", "BANK_ACCOUNT"),
    # A payment card number identifies a person as squarely as a passport does,
    # so it is protected here rather than through a separate PCI concept. It is
    # pseudonymised, not dropped: a catalog tag is what earns SUPPRESS.
    (r"(card_number|card_no|cardno|primary_account_number)", "CARD_NUMBER"),
    (r"(^|_)(cvv|cvc|csc|security_code)($|_)", "CARD_SECURITY_CODE"),
]

_EMAIL_VALUE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_PHONE_VALUE = re.compile(r"^\+?[\d][\d\s\-()]{6,}$")
_CARD_VALUE = re.compile(r"^(?:\d[ -]?){13,19}$")

# A date renders as "2020-10-26" (or with a time), which satisfies any
# digits-and-separators phone pattern. Dates are common and phone columns are
# not usually anonymous, so exclude them explicitly rather than trying to write
# a phone pattern narrow enough to miss them.
_DATE_LIKE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{1,2}[-/]\d{1,2}[-/]\d{4}")


def _normalise(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", column_name.lower())


def _match(patterns, name: str) -> str | None:
    for pattern, tag in patterns:
        if re.search(pattern, name):
            return tag
    return None


def _value_signal(series: Any) -> tuple[str, str] | None:
    """Classify from the values, for columns whose name gives nothing away."""
    try:
        import pandas as pd

        # A typed date column is never an identifier, and its string form
        # otherwise trips the phone pattern.
        if pd.api.types.is_datetime64_any_dtype(series):
            return None
        if pd.api.types.is_bool_dtype(series):
            return None
        sample = series.dropna().astype(str).head(200)
    except Exception:  # noqa: BLE001
        return None
    if sample.empty:
        return None

    def rate(pattern) -> float:
        return float(sample.str.match(pattern).mean())

    if rate(_DATE_LIKE) > 0.5:
        return None

    if rate(_EMAIL_VALUE) > 0.8:
        return "EMAIL", "values are email addresses"
    if rate(_CARD_VALUE) > 0.8:
        return "CARD_NUMBER", "values look like payment card numbers"

    # Phone numbers carry 7-15 digits; a longer or shorter run of digits with
    # separators is some other identifier or a measurement.
    if rate(_PHONE_VALUE) > 0.8:
        digits = sample.str.replace(r"\D", "", regex=True).str.len()
        if digits.between(7, 15).mean() > 0.8:
            return "PHONE", "values look like phone numbers"
    return None


def infer_column(column_name: str, series: Any = None) -> Inference | None:
    """Guess a column's nature, or None if nothing about it looks risky.

    Name patterns are checked before values because a name is cheap and, when it
    matches, unambiguous. Values are the fallback for columns whose name gives
    nothing away (`col_7`, `attr_3`).
    """
    name = _normalise(column_name)

    tag = _match(_DIRECT_IDENTIFIER, name)
    if tag:
        return Inference(
            group=GROUP_DIRECT,
            tags=_tag("PII", tag),
            reason=f"column name matches direct-identifier pattern ({tag})",
        )

    if series is not None:
        signal = _value_signal(series)
        if signal:
            tag, why = signal
            return Inference(group=GROUP_DIRECT, tags=_tag("PII", tag), reason=why)

    return None


__all__ = [
    "GROUP_DIRECT",
    "INFERRED",
    "Inference",
    "infer_column",
]
