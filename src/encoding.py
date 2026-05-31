"""
Traffic Demand Prediction - Encoding
====================================
Frequency, count, and K-Fold target encoding (leakage-safe).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def frequency_encoding(train, test, cols):
    """Frequency encoding: replace category with its frequency in train."""
    for col in cols:
        if col not in train.columns:
            continue
        freq = train[col].value_counts(normalize=True)
        train[f'{col}_freq'] = train[col].map(freq).astype(np.float32)
        test[f'{col}_freq'] = test[col].map(freq).fillna(0).astype(np.float32)
    return train, test


def count_encoding(train, test, cols):
    """Count encoding: replace category with its count in train."""
    for col in cols:
        if col not in train.columns:
            continue
        cnt = train[col].value_counts()
        train[f'{col}_count'] = train[col].map(cnt).astype(np.int32)
        test[f'{col}_count'] = test[col].map(cnt).fillna(0).astype(np.int32)
    return train, test


def kfold_target_encoding(train, test, cols, target_col, n_folds=5, random_state=42, alpha=5):
    """
    K-Fold target encoding (leakage-safe).
    Uses OOF strategy: for each fold, compute target mean on other folds.
    """
    global_mean = train[target_col].mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for col in cols:
        if col not in train.columns:
            continue
        feat_name = f'{col}_target_enc'
        train[feat_name] = np.nan

        for tr_idx, val_idx in kf.split(train):
            tr_data = train.iloc[tr_idx]
            agg = tr_data.groupby(col)[target_col].agg(['mean', 'count'])
            # Smoothed target encoding
            smoothed = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
            train.loc[train.index[val_idx], feat_name] = train.iloc[val_idx][col].map(smoothed)

        train[feat_name] = train[feat_name].fillna(global_mean).astype(np.float32)

        # For test: use full train data
        agg = train.groupby(col)[target_col].agg(['mean', 'count'])
        smoothed = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
        test[feat_name] = test[col].map(smoothed).fillna(global_mean).astype(np.float32)

    return train, test


def apply_encodings(train, test, schema):
    """Apply all encoding methods."""
    print("\n--- Encoding ---")
    target = schema['target']
    cat_cols = schema.get('categorical', [])
    geo_cols = schema.get('geohash', [])
    encode_cols = cat_cols + geo_cols

    # Also encode string interaction features
    str_interact = [c for c in train.columns if train[c].dtype == 'object'
                    and c != target and c not in encode_cols]
    all_encode = encode_cols + str_interact

    if not all_encode:
        print("  No categorical columns to encode.")
        return train, test

    print(f"  Encoding {len(all_encode)} columns: {all_encode[:10]}")

    train, test = frequency_encoding(train, test, all_encode)
    print("    Frequency encoding done")
    train, test = count_encoding(train, test, all_encode)
    print("    Count encoding done")

    if target and target in train.columns:
        # Target encoding only on categorical + geohash (most impactful)
        te_cols = [c for c in encode_cols if c in train.columns]
        if te_cols:
            train, test = kfold_target_encoding(train, test, te_cols, target)
            print(f"    K-Fold target encoding done ({len(te_cols)} cols)")

    print(f"  Train: {train.shape}, Test: {test.shape}")
    return train, test
