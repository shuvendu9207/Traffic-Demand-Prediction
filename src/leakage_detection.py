"""
Traffic Demand Prediction - Leakage Detection
=============================================
Comprehensive leakage analysis for competition datasets.
"""
import numpy as np
import pandas as pd


def detect_leakage(train, test, schema):
    """Run leakage checks silently and print a concise summary."""
    geo_cols = schema.get('geohash', [])
    dt_cols = schema.get('datetime', [])
    id_col = schema.get('id_column')
    findings = []

    # 1. Train-test overlap
    common_cols = [c for c in train.columns if c in test.columns and c != id_col]
    if common_cols:
        train_keys = train[common_cols].astype(str).fillna('nan').apply('|'.join, axis=1)
        test_keys = test[common_cols].astype(str).fillna('nan').apply('|'.join, axis=1)
        overlap = set(train_keys) & set(test_keys)
        pct = len(overlap) / len(test_keys) * 100
        findings.append(f"Row Overlap: {len(overlap)} rows ({pct:.2f}%)")

    # 2. Timestamp overlap
    for col in dt_cols:
        if col in train.columns and col in test.columns:
            tr_ts = set(train[col].unique())
            te_ts = set(test[col].unique())
            overlap = tr_ts & te_ts
            findings.append(f"Shared {col}s: {len(overlap)} ({len(overlap)/max(len(te_ts),1)*100:.1f}%)")

    # 3. Geohash overlap
    for col in geo_cols:
        if col in train.columns and col in test.columns:
            tr_g = set(train[col].unique())
            te_g = set(test[col].unique())
            overlap = tr_g & te_g
            findings.append(f"Shared {col}s: {len(overlap)} ({len(overlap)/max(len(te_g),1)*100:.1f}%)")

    print("Leakage Analysis -> " + " | ".join(findings))
    return findings
