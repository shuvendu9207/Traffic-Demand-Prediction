"""
TrafficDemandElite995 - Ensemble
=================================
OOF-based ensemble weight optimization.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score


def optimize_ensemble_weights(oof_preds_list, y_true, model_names=None):
    """
    Find optimal ensemble weights by maximizing R2 on OOF predictions.

    Args:
        oof_preds_list: list of arrays, OOF predictions from each model
        y_true: array, true target values
        model_names: list of str, names of models

    Returns:
        best_weights: list of optimal weights
        best_r2: R2 score with optimal weights
    """
    n_models = len(oof_preds_list)
    if model_names is None:
        model_names = [f'Model_{i}' for i in range(n_models)]

    print(f"\n--- Ensemble Weight Optimization ({n_models} models) ---")

    # Individual model scores
    for i, (name, oof) in enumerate(zip(model_names, oof_preds_list)):
        r2 = r2_score(y_true, oof)
        print(f"  {name} OOF R2: {r2:.6f}")

    # Grid search for 2 models (fast)
    if n_models == 2:
        best_r2, best_w = -np.inf, 0.5
        for w in np.arange(0.0, 1.01, 0.01):
            blend = w * oof_preds_list[0] + (1 - w) * oof_preds_list[1]
            r2 = r2_score(y_true, blend)
            if r2 > best_r2:
                best_r2, best_w = r2, w
        best_weights = [best_w, 1 - best_w]
    else:
        # Scipy minimize for N models
        def neg_r2(weights):
            w = np.abs(weights) / np.sum(np.abs(weights))
            blend = sum(w[i] * oof_preds_list[i] for i in range(n_models))
            return -r2_score(y_true, blend)

        init = np.ones(n_models) / n_models
        result = minimize(neg_r2, init, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-8})
        raw = np.abs(result.x)
        best_weights = (raw / raw.sum()).tolist()
        best_r2 = -result.fun

    # Display results
    print(f"\n  Optimal Weights:")
    for name, w in zip(model_names, best_weights):
        print(f"    {name}: {w:.4f}")
    print(f"  Ensemble R2: {best_r2:.6f}")

    return best_weights, best_r2


def blend_predictions(preds_list, weights):
    """Blend predictions using given weights."""
    result = np.zeros_like(preds_list[0], dtype=np.float64)
    for pred, w in zip(preds_list, weights):
        result += w * pred
    return result


def generate_submission(test_preds, test_df, id_col, target_col, output_path):
    """Generate submission CSV."""
    sub = test_df[[id_col]].copy()
    sub[target_col] = test_preds
    sub.to_csv(output_path, index=False)
    print(f"\n  Submission saved: {output_path}")
    print(f"  Shape: {sub.shape}")
    print(f"  Preview:\n{sub.head()}")
    return sub
