"""Compare a raw dataset against a generated one, using the platform's own checks.

    python -m scripts.compare_datasets <real_dir> <synthetic_dir>
"""
import sys
from pathlib import Path

import pandas as pd

from app.synth.integrity import check_generation
from app.synth.schema_infer import infer_schema

REAL = Path(sys.argv[1])
SYN = Path(sys.argv[2])


def load(root):
    return {p.stem: pd.read_csv(p) for p in sorted(root.glob("*.csv"))}


real, syn = load(REAL), load(SYN)
schema = infer_schema(real)

print("=" * 78)
print("SHAPE")
print("=" * 78)
print(f"  {'table':<16}{'real rows':>12}{'synth rows':>12}{'ratio':>8}   columns")
for t in sorted(real):
    r, s = real[t], syn.get(t)
    if s is None:
        print(f"  {t:<16}{len(r):>12,}{'MISSING':>12}")
        continue
    same = list(r.columns) == list(s.columns)
    print(
        f"  {t:<16}{len(r):>12,}{len(s):>12,}{len(s)/max(len(r),1):>8.2f}   "
        + ("identical" if same else f"DIFFER real={len(r.columns)} synth={len(s.columns)}")
    )

print()
print("=" * 78)
print("SCHEMA DETECTED FROM THE RAW DATA")
print("=" * 78)
for t, pk in sorted(schema.primary_keys.items()):
    fks = ", ".join(f"{f.child_column}->{f.parent_table}" for f in schema.foreign_keys_of(t))
    print(f"  {t:<16} pk={pk:<16} {fks or '(root)'}")

print()
print("=" * 78)
print("INTEGRITY (the platform's own checks)")
print("=" * 78)
# Treat every key column as protected, plus the obvious identity columns, so the
# disclosure checks run even though no policy object came with these files.
policy = {
    "tables": {
        t: {
            "pseudonymised": sorted(
                schema.key_columns_of(t)
                | {c for c in real[t].columns if c in {"full_name", "email", "phone"}}
            )
        }
        for t in real
    }
}
result = check_generation(real, syn, schema=schema, policy=policy)
print(f"  STATUS: {result.status}")
for c in result.checks:
    mark = "ok  " if c.status == "PASS" else c.status
    print(f"  [{mark}] {c.name}")
    if c.status != "PASS":
        print(f"         {c.detail}")

print()
print("=" * 78)
print("DISTRIBUTIONS")
print("=" * 78)
for t in sorted(real):
    s = syn.get(t)
    if s is None:
        continue
    shared = [c for c in real[t].columns if c in s.columns]
    num = [c for c in shared if pd.api.types.is_numeric_dtype(real[t][c]) and real[t][c].nunique() > 10]
    cat = [c for c in shared if real[t][c].dtype == object and real[t][c].nunique() <= 25]
    if not num and not cat:
        continue
    print(f"\n  --- {t} ---")
    for c in num:
        r, sv = real[t][c].dropna(), s[c].dropna()
        if r.empty or sv.empty:
            continue
        drift = abs(sv.median() - r.median()) / max(abs(r.median()), 1e-9)
        flag = "  <-- drift" if drift > 0.15 else ""
        print(
            f"    {c:<24} median {r.median():>12,.2f} vs {sv.median():>12,.2f}   "
            f"range [{r.min():,.0f}, {r.max():,.0f}] vs [{sv.min():,.0f}, {sv.max():,.0f}]{flag}"
        )
    for c in cat:
        rp = real[t][c].astype(str).value_counts(normalize=True)
        sp = s[c].astype(str).value_counts(normalize=True)
        idx = rp.index.union(sp.index)
        tvd = 0.5 * (rp.reindex(idx, fill_value=0) - sp.reindex(idx, fill_value=0)).abs().sum()
        unseen = sorted(set(sp.index) - set(rp.index))[:3]
        flag = "  <-- drift" if tvd > 0.15 else ""
        extra = f"  UNSEEN: {unseen}" if unseen else ""
        print(f"    {c:<24} total-variation distance {tvd:.3f}{flag}{extra}")

print()
print("=" * 78)
print("SAMPLE ROWS (customers)")
print("=" * 78)
if "customers" in real and "customers" in syn:
    cols = [c for c in ("customer_id", "full_name", "date_of_birth", "customer_segment") if c in real["customers"].columns]
    print("  REAL :")
    print(real["customers"][cols].head(3).to_string(index=False))
    print("  SYNTH:")
    print(syn["customers"][cols].head(3).to_string(index=False))
