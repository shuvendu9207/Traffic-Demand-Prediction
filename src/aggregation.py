"""
Traffic Demand Prediction - Aggregation Features
================================================
Group-based statistical features for memorization patterns.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _build_agg_features(train, test, group_cols, target_col, stats=None,
                        n_folds=5, random_state=42):
    """Build target aggregation features with OOF values for train."""
    if stats is None:
        stats = ['mean', 'median', 'std', 'count', 'min', 'max']

    grp_name = '_'.join(group_cols)

    # Check all group cols exist
    for c in group_cols:
        if c not in train.columns or c not in test.columns:
            return train, test

    feat_cols = [f'{grp_name}_{target_col}_{s}' for s in stats]
    before_train = train.shape[1]

    for feat in feat_cols:
        train[feat] = np.nan

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr_idx, val_idx in kf.split(train):
        tr_data = train.iloc[tr_idx]
        val_keys = train.iloc[val_idx][group_cols].copy()
        agg = tr_data.groupby(group_cols)[target_col].agg(stats).reset_index()
        agg.columns = group_cols + feat_cols
        val_features = val_keys.merge(agg, on=group_cols, how='left')
        train.loc[train.index[val_idx], feat_cols] = val_features[feat_cols].values

    # Test can use all training targets because those targets are available at inference.
    full_agg = train.groupby(group_cols)[target_col].agg(stats).reset_index()
    full_agg.columns = group_cols + feat_cols
    test = test.merge(full_agg, on=group_cols, how='left')

    for feat in feat_cols:
        fill = full_agg[feat].median() if feat in full_agg.columns else train[target_col].median()
        train[feat] = train[feat].fillna(fill)
        test[feat] = test[feat].fillna(fill)

    new_count = train.shape[1] - before_train
    return train, test


def build_aggregation_features(train, test, schema):
    target = schema['target']
    if not target or target not in train.columns:
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
    if 'hour' in train.columns and geo_col:
        groups.append([geo_col, 'hour'])
    if 'slot' in train.columns and geo_col:
        groups.append([geo_col, 'slot'])
    for prefix_col in [c for c in train.columns if c.endswith('_prefix4') or c.endswith('_prefix5')]:
        groups.append([prefix_col])
        if 'hour' in train.columns:
            groups.append([prefix_col, 'hour'])
        if 'slot' in train.columns:
            groups.append([prefix_col, 'slot'])
        if 'RoadType' in train.columns:
            groups.append([prefix_col, 'RoadType'])
    if 'RoadType' in train.columns and 'hour' in train.columns:
        groups.append(['RoadType', 'hour'])
    if 'RoadType' in train.columns and 'is_rush_hour' in train.columns:
        groups.append(['RoadType', 'is_rush_hour'])
    if 'Weather' in train.columns and 'hour' in train.columns:
        groups.append(['Weather', 'hour'])
    if 'Weather' in train.columns and 'is_rush_hour' in train.columns:
        groups.append(['Weather', 'is_rush_hour'])

    before = train.shape[1]
    for grp in groups:
        train, test = _build_agg_features(train, test, grp, target)

    after = train.shape[1]
    print(f"Aggregation Features -> Created {after - before} features from {len(groups)} groups | Final train shape: {train.shape}")
    return train, test


def _oof_group_stat(train, test, group_cols, target_col, stat='mean',
                    feature_name=None, n_folds=5, random_state=42):
    """Create one leakage-safe grouped target statistic."""
    if feature_name is None:
        feature_name = f"{'_'.join(group_cols)}_{target_col}_{stat}"

    for c in group_cols:
        if c not in train.columns or c not in test.columns:
            return train, test, 0

    train[feature_name] = np.nan
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for tr_idx, val_idx in kf.split(train):
        tr_data = train.iloc[tr_idx]
        val_keys = train.iloc[val_idx][group_cols].copy()
        agg = tr_data.groupby(group_cols)[target_col].agg(stat).reset_index(name=feature_name)
        val_features = val_keys.merge(agg, on=group_cols, how='left')
        train.loc[train.index[val_idx], feature_name] = val_features[feature_name].values

    full_agg = train.groupby(group_cols)[target_col].agg(stat).reset_index(name=feature_name)
    test = test.merge(full_agg, on=group_cols, how='left')
    fill = full_agg[feature_name].median()
    train[feature_name] = train[feature_name].fillna(fill)
    test[feature_name] = test[feature_name].fillna(fill)
    return train, test, 1


def build_memorization_features(train, test, schema):
    target = schema['target']
    if not target:
        return train, test

    geo_cols = schema.get('geohash', [])
    dt_cols = schema.get('datetime', [])
    geo_col = geo_cols[0] if geo_cols else None
    dt_col = dt_cols[0] if dt_cols else None

    created = 0

    if geo_col:
        train, test, c = _oof_group_stat(
            train, test, [geo_col], target, 'mean', 'mean_demand_by_geohash'
        )
        created += c

    if dt_col:
        train, test, c = _oof_group_stat(
            train, test, [dt_col], target, 'mean', 'mean_demand_by_timestamp'
        )
        created += c

    if geo_col and dt_col:
        train, test, c = _oof_group_stat(
            train, test, [geo_col, dt_col], target, 'mean',
            'mean_demand_by_geohash_timestamp'
        )
        created += c
        train, test, c = _oof_group_stat(
            train, test, [geo_col, dt_col], target, 'median',
            'median_demand_by_geohash_timestamp'
        )
        created += c

    for prefix_col in [c for c in train.columns if c.endswith('_prefix4') or c.endswith('_prefix5')]:
        train, test, c = _oof_group_stat(
            train, test, [prefix_col], target, 'mean',
            f'mean_demand_by_{prefix_col}'
        )
        created += c

    print(f"Memorization Features -> Created {created} features | Final train shape: {train.shape}")
    return train, test
