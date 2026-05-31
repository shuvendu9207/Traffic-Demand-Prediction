"""
Traffic Demand Prediction - Leakage Detection
=============================================
Comprehensive leakage analysis for competition datasets.
"""
import numpy as np
import pandas as pd


def detect_leakage(train, test, schema):
    """
    Run all leakage detection checks:
    1. Train-test overlap
    2. Duplicate groups
    3. Timestamp overlap
    4. Geohash overlap
    5. Deterministic patterns
    6. Target variance analysis
    7. Hidden memorization patterns
    """
    print("\n" + "=" * 70)
    print("  LEAKAGE DETECTION REPORT")
    print("=" * 70)
    target = schema['target']
    geo_cols = schema.get('geohash', [])
    dt_cols = schema.get('datetime', [])
    id_col = schema.get('id_column')
    findings = []

    # 1. Train-test overlap
    print("\n1. Train-Test Row Overlap")
    print("-" * 40)
    common_cols = [c for c in train.columns if c in test.columns and c != id_col]
    if common_cols:
        train_keys = train[common_cols].astype(str).fillna('nan').apply('|'.join, axis=1)
        test_keys = test[common_cols].astype(str).fillna('nan').apply('|'.join, axis=1)
        overlap = set(train_keys) & set(test_keys)
        pct = len(overlap) / len(test_keys) * 100
        msg = f"  Overlap: {len(overlap)} unique rows ({pct:.2f}% of test)"
        print(msg)
        if pct > 50:
            findings.append(f"HIGH LEAKAGE: {pct:.1f}% train-test row overlap")
    else:
        print("  No common columns to check.")

    # 2. Duplicate groups
    print("\n2. Duplicate Groups in Train")
    print("-" * 40)
    if target and target in train.columns:
        excl = [target, id_col] if id_col else [target]
        feat_cols = [c for c in train.columns if c not in excl]
        if feat_cols:
            dup_groups = train.groupby(feat_cols).size().reset_index(name='count')
            multi = dup_groups[dup_groups['count'] > 1]
            print(f"  Total feature groups: {len(dup_groups)}")
            print(f"  Groups with duplicates: {len(multi)}")
            if len(multi) > 0:
                avg = multi['count'].mean()
                mx = multi['count'].max()
                print(f"  Avg group size: {avg:.2f}, Max: {mx}")
                if avg > 2:
                    findings.append(f"Duplicate groups detected (avg size {avg:.1f})")

    # 3. Timestamp overlap
    print("\n3. Timestamp Overlap")
    print("-" * 40)
    for col in dt_cols:
        if col in train.columns and col in test.columns:
            tr_ts = set(train[col].unique())
            te_ts = set(test[col].unique())
            overlap = tr_ts & te_ts
            print(f"  {col}: {len(overlap)} shared timestamps "
                  f"({len(overlap)/len(te_ts)*100:.1f}% of test timestamps)")
            if len(overlap) / max(len(te_ts), 1) > 0.8:
                findings.append(f"High timestamp overlap: {len(overlap)}/{len(te_ts)}")

    # 4. Geohash overlap
    print("\n4. Geohash Overlap")
    print("-" * 40)
    for col in geo_cols:
        if col in train.columns and col in test.columns:
            tr_g = set(train[col].unique())
            te_g = set(test[col].unique())
            overlap = tr_g & te_g
            unseen = te_g - tr_g
            print(f"  {col}: {len(overlap)} shared ({len(overlap)/len(te_g)*100:.1f}% of test)")
            print(f"  Unseen in test: {len(unseen)}")
            if len(overlap) / max(len(te_g), 1) > 0.9:
                findings.append(f"High geohash overlap: {len(overlap)}/{len(te_g)}")

    # 5. Deterministic patterns
    print("\n5. Deterministic Patterns")
    print("-" * 40)
    if target and target in train.columns and geo_cols and dt_cols:
        geo = geo_cols[0]; dt = dt_cols[0]
        if geo in train.columns and dt in train.columns:
            grp = train.groupby([geo, dt])[target]
            var = grp.var().dropna()
            zero_var = (var == 0).sum()
            low_var = (var < 0.01).sum()
            print(f"  Zero-variance groups: {zero_var}/{len(var)}")
            print(f"  Low-variance groups (<0.01): {low_var}/{len(var)}")
            if zero_var / max(len(var), 1) > 0.5:
                findings.append(f"Many deterministic patterns: {zero_var} zero-var groups")

    # 6. Target variance by group
    print("\n6. Target Variance by Group")
    print("-" * 40)
    if target and target in train.columns:
        for col in geo_cols + dt_cols:
            if col in train.columns:
                var = train.groupby(col)[target].var()
                print(f"  {col}: mean_var={var.mean():.4f}, median_var={var.median():.4f}")

    # 7. Hidden memorization patterns
    print("\n7. Hidden Memorization Patterns")
    print("-" * 40)
    if target and geo_cols and dt_cols:
        geo = geo_cols[0]; dt = dt_cols[0]
        if geo in train.columns and dt in train.columns:
            lookup = train.groupby([geo, dt])[target].mean().reset_index()
            lookup.columns = [geo, dt, '_lookup']
            merged = train.merge(lookup, on=[geo, dt], how='left')
            residual = (merged[target] - merged['_lookup']).abs()
            print(f"  Lookup residual: mean={residual.mean():.6f}, max={residual.max():.6f}")
            perfect = (residual < 1e-6).sum()
            print(f"  Perfect matches: {perfect}/{len(train)} ({perfect/len(train)*100:.1f}%)")
            if perfect / len(train) > 0.5:
                findings.append("Strong memorization pattern detected")

    # Summary
    print("\n" + "=" * 70)
    print("  LEAKAGE SUMMARY")
    print("=" * 70)
    if findings:
        for i, f in enumerate(findings, 1):
            print(f"  [{i}] {f}")
        print("\n  RECOMMENDATION: Exploit these patterns in feature engineering.")
    else:
        print("  No significant leakage detected.")
    print("=" * 70)

    return findings
