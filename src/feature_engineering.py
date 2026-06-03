"""
Traffic Demand Prediction - Feature Engineering
===============================================
Timestamp, cyclic, geohash, and interaction features.
"""
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def _safe_geohash_decode(gh):
    """Decode geohash to (lat, lon). Returns (NaN, NaN) on failure."""
    try:
        import pygeohash as gh_lib
        lat, lon = gh_lib.decode(gh)
        return float(lat), float(lon)
    except Exception:
        return np.nan, np.nan

def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def add_timestamp_features(df, dt_cols):
    """Extract calendar, slot, cyclic, and traffic-period features."""
    for col in dt_cols:
        if col not in df.columns:
            continue
        ts = pd.to_datetime(df[col], errors='coerce')
        df['year'] = ts.dt.year
        df['month'] = ts.dt.month
        df['day'] = ts.dt.day
        df['hour'] = ts.dt.hour
        df['minute'] = ts.dt.minute
        df['day_of_week'] = ts.dt.dayofweek
        df['day_of_year'] = ts.dt.dayofyear
        df['week_of_year'] = ts.dt.isocalendar().week.astype(float)
        df['slot'] = df['hour'] * 4 + df['minute'] // 15
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].apply(
            lambda h: 1 if (7 <= h <= 10) or (16 <= h <= 20) else 0
        )
        df['is_morning_peak'] = df['hour'].between(7, 10).astype(int)
        df['is_evening_peak'] = df['hour'].between(16, 20).astype(int)
        df['is_night'] = ((df['hour'] <= 5) | (df['hour'] >= 22)).astype(int)
        df['is_business_hour'] = df['hour'].between(9, 18).astype(int)
        df['minutes_since_midnight'] = df['hour'] * 60 + df['minute']
        # Cyclic encoding
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_minute'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['cos_minute'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['sin_slot'] = np.sin(2 * np.pi * df['slot'] / 96)
        df['cos_slot'] = np.cos(2 * np.pi * df['slot'] / 96)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        break  # only first datetime col
    return df

def add_geohash_features(df, geo_cols):
    """Decode geohash columns and create location granularity features."""
    for col in geo_cols:
        if col not in df.columns:
            continue
        decoded = df[col].apply(_safe_geohash_decode)
        df['latitude'] = decoded.apply(lambda x: x[0])
        df['longitude'] = decoded.apply(lambda x: x[1])
        # Geohash precision features
        df['geohash_len'] = df[col].astype(str).str.len()
        gh = df[col].astype(str)
        for precision in [4, 5, 6]:
            df[f'{col}_prefix{precision}'] = gh.str[:precision]
    return df

def add_interaction_features(df, geo_center=None):
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
        df['lat_round_2'] = df['latitude'].round(2)
        df['lon_round_2'] = df['longitude'].round(2)
        df['lat_round_3'] = df['latitude'].round(3)
        df['lon_round_3'] = df['longitude'].round(3)
        if geo_center is None:
            center_lat = df['latitude'].median()
            center_lon = df['longitude'].median()
        else:
            center_lat, center_lon = geo_center
        df['distance_to_geo_center_km'] = _haversine_km(
            df['latitude'], df['longitude'], center_lat, center_lon
        )
    if 'Temperature' in df.columns and 'is_rush_hour' in df.columns:
        df['Temperature_x_rush'] = df['Temperature'] * df['is_rush_hour']
    if 'Temperature' in df.columns:
        df['Temperature_sq'] = df['Temperature'] ** 2
        df['Temperature_bin'] = pd.cut(
            df['Temperature'], bins=8, labels=False, duplicates='drop'
        ).astype(float)
    if 'NumberofLanes' in df.columns:
        df['lanes_sq'] = df['NumberofLanes'] ** 2
        if 'is_rush_hour' in df.columns:
            df['lanes_x_rush'] = df['NumberofLanes'] * df['is_rush_hour']
        if 'hour' in df.columns:
            df['lanes_x_hour'] = df['NumberofLanes'] * df['hour']
    if 'RoadType' in df.columns and 'is_rush_hour' in df.columns:
        df['RoadType_x_rush'] = df['RoadType'].astype(str) + '_' + df['is_rush_hour'].astype(str)
    if 'Weather' in df.columns and 'is_rush_hour' in df.columns:
        df['Weather_x_rush'] = df['Weather'].astype(str) + '_' + df['is_rush_hour'].astype(str)
    return df

def engineer_features(train, test, schema):
    dt_cols = schema.get('datetime', [])
    geo_cols = schema.get('geohash', [])

    for label, df in [("Train", train), ("Test", test)]:
        add_timestamp_features(df, dt_cols)
        add_geohash_features(df, geo_cols)

    geo_center = None
    if 'latitude' in train.columns and 'longitude' in train.columns:
        geo_center = (train['latitude'].median(), train['longitude'].median())

    for label, df in [("Train", train), ("Test", test)]:
        add_interaction_features(df, geo_center=geo_center)

    new_cols = [c for c in train.columns if c not in schema.get('all_features', [])
                and c != schema.get('target') and c != schema.get('id_column')]
    print(f"Feature Engineering -> Created {len(new_cols)} features | Final train shape: {train.shape}")
    return train, test
