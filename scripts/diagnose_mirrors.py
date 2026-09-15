"""Say WHY two identical source columns came out different.

Duplicate columns are mirrored so the copy survives into the synthetic output.
That happens in two places -- the encoder, for modelled columns, and the
relational layer, for keys -- and both record their decision at TRAINING time.
So a model trained before the mirroring fix carries no mirrors at all, and
re-generating from it cannot fix the output. This walks the four things that
have to be true, in order, and stops at the first one that is not.

    python -m scripts.diagnose_mirrors --source <file.csv | dir> \
        [--model <artifact dir>] [--output <generated dir>] [--table NAME] \
        [--columns col_a col_b]

Run it on the machine that produced the bad output, against that run's own
model artifact -- the answer is usually which of these four is false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from app.synth.preprocess import TableEncoder, _column_fingerprint, infer_column_kind

OK = "  [ok  ]"
BAD = "  [FAIL]"
INFO = "        "


def _load(path: Path) -> dict[str, pd.DataFrame]:
    """Read one file, or every table in a directory."""
    if path.is_file():
        reader = pd.read_parquet if path.suffix.lower() == ".parquet" else pd.read_csv
        return {path.stem: reader(path)}
    frames: dict[str, pd.DataFrame] = {}
    for child in sorted(path.iterdir()):
        if child.suffix.lower() == ".parquet":
            frames[child.stem] = pd.read_parquet(child)
        elif child.suffix.lower() == ".csv":
            frames.setdefault(child.stem, pd.read_csv(child))
    return frames


def duplicate_groups(frame: pd.DataFrame) -> dict[str, str]:
    """{duplicate column: the first column holding those exact values}."""
    seen: dict[str, str] = {}
    mirrors: dict[str, str] = {}
    for name in frame.columns:
        fingerprint = _column_fingerprint(frame[name])
        if fingerprint is None:
            continue
        if fingerprint in seen:
            mirrors[name] = seen[fingerprint]
        else:
            seen[fingerprint] = name
    return mirrors


def check_code() -> bool:
    """1. Is the deployed code new enough to mirror at all?"""
    print("1. Deployed code")
    from app.synth import relational
    from app.synth.preprocess import ColumnSpec

    has_relational = hasattr(relational.RelationalSPN, "_apply_column_mirrors")
    has_encoder = "mirror_of" in ColumnSpec.__dataclass_fields__
    print(f"{OK if has_encoder else BAD} encoder mirroring (ColumnSpec.mirror_of)")
    print(f"{OK if has_relational else BAD} key mirroring (_apply_column_mirrors)")
    if not (has_relational and has_encoder):
        print(f"{INFO} -> this machine is running code from before the fix.")
        print(f"{INFO}    Deploy the current build and RETRAIN.")
    return has_relational and has_encoder


def check_source(frames: dict[str, pd.DataFrame], table: str | None,
                 columns: list[str] | None) -> dict[str, dict[str, str]]:
    """2. Do the columns actually hold identical values?"""
    print("\n2. Source data")
    found: dict[str, dict[str, str]] = {}
    for name, frame in frames.items():
        if table and name != table:
            continue
        mirrors = duplicate_groups(frame)
        if mirrors:
            found[name] = mirrors
            for duplicate, origin in mirrors.items():
                print(f"{OK} {name}: {origin} == {duplicate}  "
                      f"(kind={infer_column_kind(frame[origin])})")

        if not columns:
            continue
        a, b = columns[0], columns[1]
        if a not in frame.columns or b not in frame.columns:
            continue
        if b in mirrors or a in mirrors:
            continue
        # Named a pair that is NOT an exact duplicate: say what differs.
        left, right = frame[a].astype(str), frame[b].astype(str)
        exact = float((left == right).mean())
        loose = float(
            (left.str.strip().str.upper() == right.str.strip().str.upper()).mean()
        )
        print(f"{BAD} {name}: {a} and {b} are NOT exact duplicates")
        print(f"{INFO} dtypes    : {frame[a].dtype} | {frame[b].dtype}")
        print(f"{INFO} nulls     : {int(frame[a].isna().sum())} | "
              f"{int(frame[b].isna().sum())}")
        print(f"{INFO} rows equal: {exact:.1%} exact, {loose:.1%} ignoring "
              f"case/whitespace")
        if loose > exact:
            print(f"{INFO} -> they agree only after normalising, so mirroring "
                  f"will never fire.")
            print(f"{INFO}    Mirroring is exact-match by design; decide which "
                  f"spelling wins and align them upstream.")
        differing = frame.loc[left != right, [a, b]]
        if not differing.empty:
            print(f"{INFO} first rows that differ:")
            for line in differing.head(3).to_string(index=False).splitlines():
                print(f"{INFO}   {line}")

    if not found:
        print(f"{INFO} no exactly-duplicated columns in the source.")
    return found


def check_model(model_dir: Path, expected: dict[str, dict[str, str]]) -> bool:
    """3. Did the model that produced the output record the mirrors?"""
    print("\n3. Model artifact")
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"{BAD} no manifest.json under {model_dir}")
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if "column_mirrors" not in manifest:
        print(f"{BAD} manifest has no 'column_mirrors' key at all")
        print(f"{INFO} -> this model was TRAINED BEFORE the fix. Mirrors are")
        print(f"{INFO}    recorded during training, so re-generating from this")
        print(f"{INFO}    artifact can never mirror. Retrain, then generate.")
        return False

    recorded = manifest.get("column_mirrors") or {}
    if not recorded:
        print(f"{BAD} 'column_mirrors' is present but empty")
        print(f"{INFO} -> trained with the fix, but no duplicates were found in")
        print(f"{INFO}    the training data. Check section 2 above: the columns")
        print(f"{INFO}    the model saw may differ from the file you just read")
        print(f"{INFO}    (the policy drops suppressed and coarsens generalised")
        print(f"{INFO}    columns before the model sees them).")
        return False

    for table, mirrors in recorded.items():
        for duplicate, origin in mirrors.items():
            print(f"{OK} {table}: {duplicate} mirrors {origin}")
    for table, mirrors in expected.items():
        for duplicate, origin in mirrors.items():
            if recorded.get(table, {}).get(duplicate) != origin:
                print(f"{BAD} {table}: {origin} == {duplicate} in the source, "
                      f"but the model did not record it")
    return True


def check_output(frames: dict[str, pd.DataFrame],
                 expected: dict[str, dict[str, str]]) -> bool:
    """4. Did the duplication survive into the generated rows?"""
    print("\n4. Generated output")
    clean = True
    for table, mirrors in expected.items():
        frame = frames.get(table)
        if frame is None:
            print(f"{INFO} {table}: not in the output directory, skipped")
            continue
        for duplicate, origin in mirrors.items():
            if duplicate not in frame.columns or origin not in frame.columns:
                continue
            rate = float(
                (frame[origin].astype(str) == frame[duplicate].astype(str)).mean()
            )
            mark = OK if rate == 1.0 else BAD
            print(f"{mark} {table}: {origin} == {duplicate} on {rate:.1%} of rows")
            if rate != 1.0:
                clean = False
    return clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source CSV/parquet file or directory")
    parser.add_argument("--model", help="model artifact directory (holds manifest.json)")
    parser.add_argument("--output", help="generated output directory")
    parser.add_argument("--table", help="limit to one table")
    parser.add_argument("--columns", nargs=2, metavar=("A", "B"),
                        help="the pair you expected to match")
    args = parser.parse_args(argv)

    print("Duplicate-column mirroring: why the copy did not survive\n")

    code_ok = check_code()
    source = _load(Path(args.source))
    if not source:
        print(f"\n{BAD} no tables read from {args.source}")
        return 2
    expected = check_source(source, args.table, args.columns)

    model_ok = True
    if args.model:
        model_ok = check_model(Path(args.model), expected)
    else:
        print("\n3. Model artifact")
        print(f"{INFO} not checked -- pass --model <artifact dir> to include it.")
        print(f"{INFO} This is the most common cause: a model trained before")
        print(f"{INFO} the fix records no mirrors, and regenerating cannot help.")

    if args.output:
        check_output(_load(Path(args.output)), expected)
    else:
        print("\n4. Generated output")
        print(f"{INFO} not checked -- pass --output <generated dir> to include it.")

    print("\nVerdict")
    if not code_ok:
        print(f"{INFO} deployed code predates the fix -> deploy, then RETRAIN.")
    elif not expected:
        print(f"{INFO} the columns are not exact duplicates -> see section 2.")
    elif args.model and not model_ok:
        print(f"{INFO} the model carries no mirrors -> RETRAIN, then generate.")
    else:
        print(f"{INFO} source and model agree; compare against section 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
