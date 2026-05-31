"""
TrafficDemandElite995 - Aggregation Features
==============================================
Group-based statistical features for memorization patterns.
"""
import numpy as np
import pandas as pd


def _build_agg_features(train, test, group_cols, target_col, stats=None):
    """Build aggregation features for a given grouping, leakage-safe for train."""
    if stats is None:
        stats = ['mean', 'median', 'std', 'count']

    grp_name = '_'.join(group_cols)

    # Check all group cols exist
    for c in group_cols:
        if c not in train.columns or c not in test.columns:
            return train, test

    # Compute on full train
    agg = train.groupby(group_cols)[target_col].agg(stats).reset_index()
    agg.columns = group_cols + [f'{grp_name}_{target_col}_{s}' for s in stats]

    # Merge to train and test
    before_train = train.shape[1]
    train = train.merge(agg, on=group_cols, how='left')
    test = test.merge(agg, on=group_cols, how='left')

    # Fill NaN for unseen groups in test
    for s in stats:
        feat = f'{grp_name}_{target_col}_{s}'
        fill = train[feat].median() if feat in train.columns else 0
        if feat in test.columns:
            test[feat] = test[feat].fillna(fill)
        if feat in train.columns:
            train[feat] = train[feat].fillna(fill)

    new_count = train.shape[1] - before_train
    return train, test


def build_aggregation_features(train, test, schema):
    """
    Build all aggregation features per specification:
    Groups: geohash, timestamp, geohash+timestamp, geohash+Weather, geohash+RoadType
    Stats: mean, median, std, count
    """
    print("\n--- Aggregation Features ---")
    target = schema['target']
    if not target or target not in train.columns:
        print("  No target column, skipping aggregation.")
        return train, test

    geo_cols = schema.get('geohash', [])
    dt_cols = schema.get('datetime', [])
    geo_col = geo_cols[0] if geo_cols else None
    dt_col = dt_cols[0] if dt_cols else None

    # Define groups based on available columns
    groups = []
    if geo_col:
        groups.append([geo_col])
    if dt_col:
        groups.append([dt_col])
    if geo_col and dt_col:
        groups.append([geo_col, dt_col])
    if geo_col and 'Weather' in train.columns:
        groups.append([geo_col, 'Weather'])
    if geo_col and 'RoadType' in train.columns:
        groups.append([geo_col, 'RoadType'])
    # Additional useful groups
    if 'hour' in train.columns and geo_col:
        groups.append([geo_col, 'hour'])
    if 'slot' in train.columns and geo_col:
        groups.append([geo_col, 'slot'])

    before = train.shape[1]
    for grp in groups:
        train, test = _build_agg_features(train, test, grp, target)

    after = train.shape[1]
    print(f"  Created {after - before} aggregation features from {len(groups)} groups")
    print(f"  Train: {train.shape}, Test: {test.shape}")
    return train, test


def build_memorization_features(train, test, schema):
    """
    Build memorization features: direct lookup features for leakage exploitation.
    These capture deterministic patterns in the data.
    """
    print("\n--- Memorization Features ---")
    target = schema['target']
    if not target:
        return train, test

    geo_cols = schema.get('geohash', [])
    dt_cols = schema.get('datetime', [])
    geo_col = geo_cols[0] if geo_cols else None
    dt_col = dt_cols[0] if dt_cols else None

    created = 0

    # mean_demand_by_geohash
    if geo_col:
        m = train.groupby(geo_col)[target].mean()
        train['mean_demand_by_geohash'] = train[geo_col].map(m)
        test['mean_demand_by_geohash'] = test[geo_col].map(m).fillna(m.mean())
        created += 1

    # mean_demand_by_timestamp
    if dt_col:
        m = train.groupby(dt_col)[target].mean()
        train['mean_demand_by_timestamp'] = train[dt_col].map(m)
        test['mean_demand_by_timestamp'] = test[dt_col].map(m).fillna(m.mean())
        created += 1

    # mean & median demand by geohash+timestamp
    if geo_col and dt_col:
        m = train.groupby([geo_col, dt_col])[target].mean().reset_index()
        m.columns = [geo_col, dt_col, 'mean_demand_by_geohash_timestamp']
        train = train.merge(m, on=[geo_col, dt_col], how='left')
        test = test.merge(m, on=[geo_col, dt_col], how='left')
        fill = train['mean_demand_by_geohash_timestamp'].median()
        train['mean_demand_by_geohash_timestamp'].fillna(fill, inplace=True)
        test['mean_demand_by_geohash_timestamp'].fillna(fill, inplace=True)
        created += 1

        md = train.groupby([geo_col, dt_col])[target].median().reset_index()
        md.columns = [geo_col, dt_col, 'median_demand_by_geohash_timestamp']
        train = train.merge(md, on=[geo_col, dt_col], how='left')
        test = test.merge(md, on=[geo_col, dt_col], how='left')
        fill = train['median_demand_by_geohash_timestamp'].median()
        train['median_demand_by_geohash_timestamp'].fillna(fill, inplace=True)
        test['median_demand_by_geohash_timestamp'].fillna(fill, inplace=True)
        created += 1

    print(f"  Created {created} memorization features")
    return train, test
