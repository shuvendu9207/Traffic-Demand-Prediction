"""
TrafficDemandElite995 - Feature Engineering
============================================
Timestamp, cyclic, geohash, and interaction features.
"""
import numpy as np
import pandas as pd

def _safe_geohash_decode(gh):
    """Decode geohash to (lat, lon). Returns (NaN, NaN) on failure."""
    try:
        import geohash as gh_lib
        lat, lon = gh_lib.decode(gh)
        return float(lat), float(lon)
    except Exception:
        return np.nan, np.nan

def add_timestamp_features(df, dt_cols):
    """Extract hour, minute, slot, is_rush_hour from datetime columns."""
    for col in dt_cols:
        if col not in df.columns:
            continue
        ts = pd.to_datetime(df[col], errors='coerce')
        df['hour'] = ts.dt.hour
        df['minute'] = ts.dt.minute
        df['day_of_week'] = ts.dt.dayofweek
        df['slot'] = df['hour'] * 4 + df['minute'] // 15
        df['is_rush_hour'] = df['hour'].apply(
            lambda h: 1 if (7 <= h <= 10) or (16 <= h <= 20) else 0
        )
        # Cyclic encoding
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_minute'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['cos_minute'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        break  # only first datetime col
    return df

def add_geohash_features(df, geo_cols):
    """Decode geohash columns to lat/lon coordinates."""
    for col in geo_cols:
        if col not in df.columns:
            continue
        decoded = df[col].apply(_safe_geohash_decode)
        df['latitude'] = decoded.apply(lambda x: x[0])
        df['longitude'] = decoded.apply(lambda x: x[1])
        # Geohash precision features
        df['geohash_len'] = df[col].astype(str).str.len()
    return df

def add_interaction_features(df):
    """Create interaction features between existing columns."""
    # Weather x hour
    if 'Weather' in df.columns and 'hour' in df.columns:
        df['Weather_x_hour'] = df['Weather'].astype(str) + '_' + df['hour'].astype(str)
    # Temperature x hour
    if 'Temperature' in df.columns and 'hour' in df.columns:
        df['Temperature_x_hour'] = df['Temperature'] * df['hour']
    # RoadType x NumberofLanes
    if 'RoadType' in df.columns and 'NumberofLanes' in df.columns:
        df['RoadType_x_NumberofLanes'] = df['RoadType'].astype(str) + '_' + df['NumberofLanes'].astype(str)
    # Additional useful interactions
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['lat_x_lon'] = df['latitude'] * df['longitude']
    if 'Temperature' in df.columns and 'is_rush_hour' in df.columns:
        df['Temperature_x_rush'] = df['Temperature'] * df['is_rush_hour']
    return df

def engineer_features(train, test, schema):
    """Main feature engineering pipeline."""
    print("\n--- Feature Engineering ---")
    dt_cols = schema.get('datetime', [])
    geo_cols = schema.get('geohash', [])

    for label, df in [("Train", train), ("Test", test)]:
        add_timestamp_features(df, dt_cols)
        add_geohash_features(df, geo_cols)
        add_interaction_features(df)

    new_cols = [c for c in train.columns if c not in schema.get('all_features', [])
                and c != schema.get('target') and c != schema.get('id_column')]
    print(f"  Created {len(new_cols)} new features: {new_cols[:15]}{'...' if len(new_cols)>15 else ''}")
    print(f"  Train: {train.shape}, Test: {test.shape}")
    return train, test
