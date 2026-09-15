"""Report columns that hold the same values, and why a pair narrowly misses.

Duplicate placeholder columns are mirrored during training so the copy survives
into the synthetic output -- but only when the columns match exactly, row for
row. This says whether a pair matches, and if not, what differs.

    python -m scripts.check_duplicate_columns <file.csv> [col_a col_b]
"""
from __future__ import annotations

import sys

import pandas as pd

from app.synth.preprocess import _column_fingerprint, infer_column_kind


def describe(frame: pd.DataFrame, a: str, b: str) -> None:
    left, right = frame[a], frame[b]
    print(f"\n{a}  vs  {b}")
    print(f"  dtype      : {left.dtype}  |  {right.dtype}")
    print(f"  kind       : {infer_column_kind(left)}  |  {infer_column_kind(right)}")
    print(f"  nulls      : {int(left.isna().sum())}  |  {int(right.isna().sum())}")
    print(f"  fingerprint: {'MATCH' if _column_fingerprint(left) == _column_fingerprint(right) else 'differ'}")

    exact = (left.astype(str) == right.astype(str)).mean()
    loose = (
        left.astype(str).str.strip().str.upper()
        == right.astype(str).str.strip().str.upper()
    ).mean()
    print(f"  rows equal : {exact:.1%} exact, {loose:.1%} ignoring case/whitespace")
    if loose > exact:
        print("  -> values agree only after normalising: mirroring will NOT fire")
    differing = frame.loc[left.astype(str) != right.astype(str), [a, b]]
    if not differing.empty:
        print("  first rows that differ:")
        print(differing.head(3).to_string(index=False))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    frame = pd.read_csv(argv[0])
    if len(argv) >= 3:
        describe(frame, argv[1], argv[2])
        return 0

    seen: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for name in frame.columns:
        fingerprint = _column_fingerprint(frame[name])
        if fingerprint is None:
            continue
        if fingerprint in seen:
            groups.setdefault(seen[fingerprint], []).append(name)
        else:
            seen[fingerprint] = name

    if groups:
        print("Exact duplicate columns (these WILL be mirrored):")
        for first, copies in groups.items():
            print(f"  {first} == {', '.join(copies)}   kind={infer_column_kind(frame[first])}")
    else:
        print("No exactly-duplicated columns found.")

    # Near misses are the interesting case: they look identical to a human.
    names = list(frame.columns)
    print("\nNear misses (equal for >90% of rows but not exactly):")
    found = False
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if _column_fingerprint(frame[a]) == _column_fingerprint(frame[b]):
                continue
            # Screen on the NORMALISED comparison, not the exact one: a
            # column that differs only in case agrees exactly nowhere, and is
            # exactly the near miss worth reporting.
            agree = (
                frame[a].astype(str).str.strip().str.upper()
                == frame[b].astype(str).str.strip().str.upper()
            ).mean()
            if agree > 0.9:
                found = True
                describe(frame, a, b)
    if not found:
        print("  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
