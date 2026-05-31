"""
TrafficDemandElite995 - Training
=================================
CatBoost, LightGBM, XGBoost training with Optuna HPO and OOF predictions.
"""
import numpy as np
import pandas as pd
import json, joblib, time, warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings('ignore')


def get_feature_columns(train, schema):
    """Get list of numeric feature columns for modeling."""
    target = schema['target']
    id_col = schema.get('id_column')
    exclude = {target, id_col}
    # Exclude raw string columns
    cols = [c for c in train.columns if c not in exclude
            and train[c].dtype != 'object']
    return cols


def _catboost_params(trial=None, use_gpu=True):
    """CatBoost parameters, optionally from Optuna trial."""
    base = {
        'loss_function': 'RMSE',
        'eval_metric': 'R2',
        'random_seed': 42,
        'verbose': 0,
        'allow_writing_files': False,
    }
    if use_gpu:
        base['task_type'] = 'GPU'
        base['devices'] = '0'

    if trial:
        base['learning_rate'] = trial.suggest_float('cb_lr', 0.01, 0.3, log=True)
        base['depth'] = trial.suggest_int('cb_depth', 4, 10)
        base['iterations'] = trial.suggest_int('cb_iters', 500, 5000, step=100)
        base['l2_leaf_reg'] = trial.suggest_float('cb_l2', 1.0, 10.0)
        base['subsample'] = trial.suggest_float('cb_subsample', 0.6, 1.0)
        base['min_data_in_leaf'] = trial.suggest_int('cb_min_leaf', 1, 50)
        base['bagging_temperature'] = trial.suggest_float('cb_bag_temp', 0.0, 1.0)
    else:
        base['learning_rate'] = 0.05
        base['depth'] = 8
        base['iterations'] = 3000
        base['l2_leaf_reg'] = 3.0
        base['subsample'] = 0.8
        base['min_data_in_leaf'] = 5
    return base


def _lgbm_params(trial=None, use_gpu=True):
    """LightGBM parameters, optionally from Optuna trial."""
    base = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
    }
    if use_gpu:
        base['device'] = 'gpu'
        base['gpu_platform_id'] = 0
        base['gpu_device_id'] = 0

    if trial:
        base['learning_rate'] = trial.suggest_float('lgb_lr', 0.01, 0.3, log=True)
        base['num_leaves'] = trial.suggest_int('lgb_leaves', 31, 512)
        base['n_estimators'] = trial.suggest_int('lgb_iters', 500, 5000, step=100)
        base['reg_alpha'] = trial.suggest_float('lgb_alpha', 0.0, 10.0)
        base['reg_lambda'] = trial.suggest_float('lgb_lambda', 0.0, 10.0)
        base['subsample'] = trial.suggest_float('lgb_subsample', 0.6, 1.0)
        base['colsample_bytree'] = trial.suggest_float('lgb_colsample', 0.5, 1.0)
        base['min_child_samples'] = trial.suggest_int('lgb_min_child', 5, 100)
        base['max_depth'] = trial.suggest_int('lgb_depth', 4, 12)
    else:
        base['learning_rate'] = 0.05
        base['num_leaves'] = 255
        base['n_estimators'] = 3000
        base['reg_alpha'] = 1.0
        base['reg_lambda'] = 1.0
        base['subsample'] = 0.8
        base['colsample_bytree'] = 0.8
        base['min_child_samples'] = 10
        base['max_depth'] = 8
    return base


def train_catboost_cv(X, y, feature_cols, n_folds=5, params=None, use_gpu=True):
    """Train CatBoost with KFold CV, return OOF predictions and models."""
    from catboost import CatBoostRegressor, Pool

    if params is None:
        params = _catboost_params(use_gpu=use_gpu)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []
    scores = []

    print(f"\n  CatBoost {n_folds}-Fold CV")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx][feature_cols], X.iloc[val_idx][feature_cols]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=0)
        pred = model.predict(X_val)
        oof[val_idx] = pred
        r2 = r2_score(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        scores.append(r2)
        models.append(model)
        print(f"    Fold {fold+1}: R2={r2:.6f}, RMSE={rmse:.6f}")

    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)
    print(f"  Mean R2: {mean_r2:.6f} +/- {std_r2:.6f}")
    return oof, models, scores


def train_lgbm_cv(X, y, feature_cols, n_folds=5, params=None, use_gpu=True):
    """Train LightGBM with KFold CV, return OOF predictions and models."""
    import lightgbm as lgb

    if params is None:
        params = _lgbm_params(use_gpu=use_gpu)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []
    scores = []

    n_est = params.pop('n_estimators', 3000)

    print(f"\n  LightGBM {n_folds}-Fold CV")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx][feature_cols], X.iloc[val_idx][feature_cols]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)

        callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
        model = lgb.train(params, dtrain, num_boost_round=n_est,
                          valid_sets=[dval], callbacks=callbacks)
        pred = model.predict(X_val)
        oof[val_idx] = pred
        r2 = r2_score(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        scores.append(r2)
        models.append(model)
        print(f"    Fold {fold+1}: R2={r2:.6f}, RMSE={rmse:.6f}")

    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)
    print(f"  Mean R2: {mean_r2:.6f} +/- {std_r2:.6f}")
    return oof, models, scores


def optuna_catboost(X, y, feature_cols, n_trials=50, n_folds=5, use_gpu=True):
    """Optimize CatBoost hyperparameters with Optuna."""
    import optuna
    from catboost import CatBoostRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = _catboost_params(trial, use_gpu)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X):
            X_tr = X.iloc[tr_idx][feature_cols]
            X_val = X.iloc[val_idx][feature_cols]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=50, verbose=0)
            pred = model.predict(X_val)
            scores.append(r2_score(y_val, pred))
        return np.mean(scores)

    print(f"\n  Optuna CatBoost ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best R2: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def optuna_lgbm(X, y, feature_cols, n_trials=50, n_folds=5, use_gpu=True):
    """Optimize LightGBM hyperparameters with Optuna."""
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = _lgbm_params(trial, use_gpu)
        n_est = params.pop('n_estimators', 3000)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X):
            X_tr = X.iloc[tr_idx][feature_cols]
            X_val = X.iloc[val_idx][feature_cols]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val)
            callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
            model = lgb.train(params, dtrain, num_boost_round=n_est,
                              valid_sets=[dval], callbacks=callbacks)
            pred = model.predict(X_val)
            scores.append(r2_score(y_val, pred))
        return np.mean(scores)

    print(f"\n  Optuna LightGBM ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best R2: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def build_best_catboost_params(best_trial_params, use_gpu=True):
    """Convert Optuna best params to CatBoost param dict."""
    params = {
        'loss_function': 'RMSE',
        'eval_metric': 'R2',
        'random_seed': 42,
        'verbose': 0,
        'allow_writing_files': False,
        'learning_rate': best_trial_params.get('cb_lr', 0.05),
        'depth': best_trial_params.get('cb_depth', 8),
        'iterations': best_trial_params.get('cb_iters', 3000),
        'l2_leaf_reg': best_trial_params.get('cb_l2', 3.0),
        'subsample': best_trial_params.get('cb_subsample', 0.8),
        'min_data_in_leaf': best_trial_params.get('cb_min_leaf', 5),
        'bagging_temperature': best_trial_params.get('cb_bag_temp', 0.5),
    }
    if use_gpu:
        params['task_type'] = 'GPU'
        params['devices'] = '0'
    return params


def build_best_lgbm_params(best_trial_params, use_gpu=True):
    """Convert Optuna best params to LightGBM param dict."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
        'learning_rate': best_trial_params.get('lgb_lr', 0.05),
        'num_leaves': best_trial_params.get('lgb_leaves', 255),
        'n_estimators': best_trial_params.get('lgb_iters', 3000),
        'reg_alpha': best_trial_params.get('lgb_alpha', 1.0),
        'reg_lambda': best_trial_params.get('lgb_lambda', 1.0),
        'subsample': best_trial_params.get('lgb_subsample', 0.8),
        'colsample_bytree': best_trial_params.get('lgb_colsample', 0.8),
        'min_child_samples': best_trial_params.get('lgb_min_child', 10),
        'max_depth': best_trial_params.get('lgb_depth', 8),
    }
    if use_gpu:
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
    return params


def predict_catboost(models, X_test, feature_cols):
    """Average predictions from multiple CatBoost models."""
    preds = np.zeros(len(X_test))
    for m in models:
        preds += m.predict(X_test[feature_cols])
    return preds / len(models)


def predict_lgbm(models, X_test, feature_cols):
    """Average predictions from multiple LightGBM models."""
    preds = np.zeros(len(X_test))
    for m in models:
        preds += m.predict(X_test[feature_cols])
    return preds / len(models)


def save_models(cb_models, lgb_models, output_dir):
    """Save trained models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(cb_models):
        m.save_model(str(output_dir / f'catboost_fold{i}.cbm'))
    for i, m in enumerate(lgb_models):
        m.save_model(str(output_dir / f'lgbm_fold{i}.txt'))
    print(f"  Models saved to {output_dir}")


def get_feature_importance(cb_models, lgb_models, feature_cols):
    """Get combined feature importance from all models."""
    import matplotlib.pyplot as plt

    imp = pd.DataFrame({'feature': feature_cols})

    # CatBoost importance
    cb_imp = np.zeros(len(feature_cols))
    for m in cb_models:
        cb_imp += m.get_feature_importance()
    cb_imp /= len(cb_models)
    imp['catboost_importance'] = cb_imp

    # LightGBM importance
    lgb_imp = np.zeros(len(feature_cols))
    for m in lgb_models:
        lgb_imp += m.feature_importance(importance_type='gain')
    lgb_imp /= len(lgb_models)
    imp['lgbm_importance'] = lgb_imp

    # Combined (normalized)
    imp['cb_norm'] = imp['catboost_importance'] / (imp['catboost_importance'].sum() + 1e-8)
    imp['lgb_norm'] = imp['lgbm_importance'] / (imp['lgbm_importance'].sum() + 1e-8)
    imp['combined'] = (imp['cb_norm'] + imp['lgb_norm']) / 2
    imp = imp.sort_values('combined', ascending=False).reset_index(drop=True)

    # Plot top 30
    top = imp.head(30)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].barh(top['feature'][::-1], top['catboost_importance'][::-1], color='#4CAF50')
    axes[0].set_title('CatBoost Feature Importance (Top 30)')
    axes[1].barh(top['feature'][::-1], top['lgbm_importance'][::-1], color='#2196F3')
    axes[1].set_title('LightGBM Feature Importance (Top 30)')
    plt.tight_layout()
    plt.show()

    return imp
