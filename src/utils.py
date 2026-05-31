"""
Traffic Demand Prediction - Utility Functions
=============================================
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

class Config:
    PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = PROJECT_ROOT / "data"
    SRC_DIR = PROJECT_ROOT / "src"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    MODEL_DIR = OUTPUT_DIR / "models"
    PRED_DIR = OUTPUT_DIR / "predictions"
    SUB_DIR = OUTPUT_DIR / "submissions"
    TRAIN_FILE = DATA_DIR / "train.csv"
    TEST_FILE = DATA_DIR / "test.csv"
    N_FOLDS = 5
    RANDOM_STATE = 42
    SHUFFLE = True
    PRIMARY_METRIC = "r2"
    TARGET_SCORE = 0.995
    OPTUNA_TRIALS = 50
    SUBMISSION_COLUMNS = ["Index", "demand"]
    SUBMISSION_FILENAME = "submission.csv"
    USE_GPU = True

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.DATA_DIR, cls.MODEL_DIR, cls.PRED_DIR, cls.SUB_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def display(cls):
        print("=" * 70)
        print("  Traffic Demand Prediction - Configuration")
        print("=" * 70)
        for k in ['PROJECT_ROOT','TRAIN_FILE','TEST_FILE','N_FOLDS','RANDOM_STATE','PRIMARY_METRIC','TARGET_SCORE','OPTUNA_TRIALS','USE_GPU']:
            print(f"  {k:20s}: {getattr(cls, k)}")
        print("=" * 70)

def validate_datasets():
    print("\n--- Dataset Validation ---")
    errors = []
    for name, path in [("Train", Config.TRAIN_FILE), ("Test", Config.TEST_FILE)]:
        if not path.exists():
            errors.append(f"  {name} file not found: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"  {name} file is empty: {path}")
        else:
            print(f"  OK {name}: {path} ({path.stat().st_size / 1024**2:.2f} MB)")
    if errors:
        print("\n  DATASET VALIDATION FAILED")
        for e in errors: print(e)
        print("\n  Expected structure:\n  data/train.csv\n  data/test.csv")
        raise FileNotFoundError("Required datasets missing.")
    print("  All datasets validated.\n")

def load_data():
    print("\n--- Loading Datasets ---")
    t0 = time.time()
    train = pd.read_csv(Config.TRAIN_FILE)
    test = pd.read_csv(Config.TEST_FILE)
    print(f"  Train: {train.shape}, Test: {test.shape}, Time: {time.time()-t0:.2f}s")
    return train, test

def detect_schema(train, test):
    print("\n--- Schema Detection ---")
    schema = {'target': None, 'id_column': None, 'categorical': [], 'numerical': [],
              'datetime': [], 'geohash': [], 'all_features': []}
    train_only = set(train.columns) - set(test.columns)
    for c in train_only:
        if train[c].dtype in ['float64','float32','int64','int32']:
            schema['target'] = c; break
    if not schema['target']:
        for c in ['demand','target','y']: 
            if c in train.columns: schema['target'] = c; break
    for c in test.columns:
        if c.lower() in ['index','id','row_id']:
            schema['id_column'] = c; break
    feat_cols = [c for c in test.columns if c != schema.get('id_column')]
    for col in feat_cols:
        if col.lower() in ['geohash','geohash6','geo_hash']:
            schema['geohash'].append(col)
        elif col.lower() in ['timestamp','datetime','date','time']:
            schema['datetime'].append(col)
        elif pd.api.types.is_string_dtype(train[col].dtype):
            try:
                pd.to_datetime(train[col].head(50)); schema['datetime'].append(col)
            except:
                s = train[col].dropna().head(100)
                if len(s) > 0 and s.str.match(r'^[0-9a-z]+$').all() and s.str.len().std() < 1:
                    schema['geohash'].append(col)
                else:
                    schema['categorical'].append(col)
        else:
            schema['numerical'].append(col)
    schema['all_features'] = feat_cols
    print(f"  Target: {schema['target']}, ID: {schema['id_column']}")
    print(f"  Cat: {len(schema['categorical'])}, Num: {len(schema['numerical'])}, "
          f"DT: {len(schema['datetime'])}, Geo: {len(schema['geohash'])}")
    return schema

def run_eda(train, test, schema):
    import matplotlib.pyplot as plt
    print("\n--- EDA ---")
    target = schema['target']
    if target: print(train[target].describe())
    miss = train.isnull().sum(); miss = miss[miss > 0]
    if len(miss): print("Missing (train):"); print(miss)
    else: print("  No missing values in train")
    print(f"  Train duplicates: {train.duplicated().sum()}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if target:
        axes[0].hist(train[target], bins=50, color='#4CAF50', edgecolor='white')
        axes[0].set_title(f'{target} Distribution')
        axes[1].boxplot(train[target].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#2196F3', alpha=0.7))
        axes[1].set_title(f'{target} Boxplot')
    plt.tight_layout(); plt.show()

def clean_data(train, test, schema):
    print("\n--- Data Cleaning ---")
    target = schema['target']
    b = len(train); train = train.drop_duplicates().reset_index(drop=True)
    print(f"  Removed {b - len(train)} duplicates")
    geo_cols = schema.get('geohash', [])
    geo_col = geo_cols[0] if geo_cols else None
    if 'Temperature' in train.columns:
        if geo_col:
            med = train.groupby(geo_col)['Temperature'].transform('median')
            train['Temperature'] = train['Temperature'].fillna(med).fillna(train['Temperature'].median())
            gm = train.groupby(geo_col)['Temperature'].median()
            gmd = gm.to_dict(); gfill = train['Temperature'].median()
            test['Temperature'] = test.apply(lambda r: gmd.get(r[geo_col], gfill) if pd.isna(r['Temperature']) else r['Temperature'], axis=1)
        else:
            v = train['Temperature'].median()
            train['Temperature'] = train['Temperature'].fillna(v)
            test['Temperature'] = test['Temperature'].fillna(v)
    if 'Weather' in train.columns:
        if geo_col:
            mm = train.groupby(geo_col)['Weather'].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else 'Unknown')
            mmd = mm.to_dict()
            train['Weather'] = train.apply(lambda r: mmd.get(r[geo_col],'Unknown') if pd.isna(r['Weather']) else r['Weather'], axis=1)
            test['Weather'] = test.apply(lambda r: mmd.get(r[geo_col],'Unknown') if pd.isna(r['Weather']) else r['Weather'], axis=1)
        else:
            v = train['Weather'].mode()[0] if len(train['Weather'].mode())>0 else 'Unknown'
            train['Weather'] = train['Weather'].fillna(v)
            test['Weather'] = test['Weather'].fillna(v)
    if 'RoadType' in train.columns:
        train['RoadType'] = train['RoadType'].fillna('Unknown')
        test['RoadType'] = test['RoadType'].fillna('Unknown')
    for col in train.columns:
        if train[col].isnull().sum() > 0:
            if pd.api.types.is_string_dtype(train[col].dtype):
                v = train[col].mode()[0] if len(train[col].mode())>0 else 'Unknown'
            else: v = train[col].median()
            train[col] = train[col].fillna(v)
            if col in test.columns: test[col] = test[col].fillna(v)
    for col in test.columns:
        if test[col].isnull().sum() > 0:
            test[col] = test[col].fillna(test[col].median() if not pd.api.types.is_string_dtype(test[col].dtype) else 'Unknown')
    for col in schema['categorical']:
        if col in train.columns: train[col] = train[col].astype(str).str.strip().str.lower()
        if col in test.columns: test[col] = test[col].astype(str).str.strip().str.lower()
    const = [c for c in train.columns if c != target and train[c].nunique() <= 1]
    if const:
        print(f"  Removing constant: {const}")
        train.drop(columns=const, inplace=True)
        test.drop(columns=[c for c in const if c in test.columns], inplace=True)
    print(f"  Done. Train: {train.shape}, Test: {test.shape}")
    return train, test, schema

class Timer:
    def __init__(self, name="Block"): self.name = name
    def __enter__(self): self.start = time.time(); return self
    def __exit__(self, *a): print(f"  {self.name}: {time.time()-self.start:.2f}s")

def reduce_mem_usage(df):
    for col in df.columns:
        t = df[col].dtype
        if t != object and t.name != 'category':
            mi, ma = df[col].min(), df[col].max()
            if str(t)[:3] == 'int':
                for dt in [np.int8, np.int16, np.int32]:
                    if mi >= np.iinfo(dt).min and ma <= np.iinfo(dt).max:
                        df[col] = df[col].astype(dt); break
            else:
                if mi >= np.finfo(np.float32).min and ma <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df
