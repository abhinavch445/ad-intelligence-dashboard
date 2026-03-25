"""
train_models.py  —  Train Order + Revenue models, bid response model, and save artifacts.

Usage:
    python training/train_models.py

Outputs:
    models/order_model.pkl
    models/revenue_model.pkl
    models/bid_model.pkl
    models/encoders.pkl
    models/feature_importance.pkl
    models/data_stats.pkl
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    USE_XGB = True
    print("Using XGBoost")
except ImportError:
    USE_XGB = False
    print("XGBoost not found — using GradientBoostingRegressor")

from utils.feature_engineering import (
    load_and_merge, encode_features,
    ORDER_FEATURES, REVENUE_FEATURES, BID_FEATURES,
    get_data_stats, get_category_stats, get_city_stats
)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMP_PATH   = os.path.join(ROOT, "data", "campaign_data.csv")
SELL_PATH   = os.path.join(ROOT, "data", "final_dataset.xlsx")
MODELS_DIR  = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def make_model(n_estimators=300, max_depth=5, lr=0.05):
    if USE_XGB:
        return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbosity=0)
    return GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                     learning_rate=lr, subsample=0.8,
                                     random_state=42)


def evaluate(model, X_te, y_te, name):
    yp  = model.predict(X_te)
    r2  = r2_score(y_te, yp)
    rmse= mean_squared_error(y_te, yp) ** 0.5
    mae = mean_absolute_error(y_te, yp)
    print(f"\n── {name} ────────────────────────")
    print(f"   R²   = {r2:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   MAE  = {mae:.4f}")
    return {"r2": round(r2,4), "rmse": round(rmse,4), "mae": round(mae,4)}


def train():
    # 1. Load data
    print("Loading data...")
    if not os.path.exists(CAMP_PATH):
        raise FileNotFoundError(
            f"campaign_data.csv not found at {CAMP_PATH}\n"
            "Copy your data files into the data/ folder."
        )
    df = load_and_merge(CAMP_PATH, SELL_PATH)
    df, cat_enc, city_enc = encode_features(df)
    print(f"Dataset: {len(df):,} rows after cleaning")

    stats       = get_data_stats(df)
    cat_stats   = get_category_stats(df)
    city_stats  = get_city_stats(df)

    # 2. MODEL 1 — Order Prediction
    print("\nTraining Model 1: Order Prediction...")
    X1 = df[ORDER_FEATURES]
    y1 = df["orders"]
    X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=42)

    order_model = make_model()
    order_model.fit(X1_tr, y1_tr)
    m1_metrics = evaluate(order_model, X1_te, y1_te, "Order Model")

    # Feature importance for orders
    imp1 = dict(zip(ORDER_FEATURES, order_model.feature_importances_))

    # 3. MODEL 2 — Revenue Prediction
    # Add predicted orders as a feature
    print("\nTraining Model 2: Revenue Prediction...")
    df["orders_pred"] = order_model.predict(df[ORDER_FEATURES])
    X2 = df[REVENUE_FEATURES]
    y2 = df["revenue"]
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)

    rev_model = make_model()
    rev_model.fit(X2_tr, y2_tr)
    m2_metrics = evaluate(rev_model, X2_te, y2_te, "Revenue Model")
    imp2 = dict(zip(REVENUE_FEATURES, rev_model.feature_importances_))

    # 4. BID RESPONSE MODEL — CPC → Clicks
    print("\nTraining Bid Response Model: CPC → Clicks...")
    X3 = df[BID_FEATURES]
    y3 = df["clicks"]
    X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)

    bid_model = make_model(n_estimators=200, max_depth=4)
    bid_model.fit(X3_tr, y3_tr)
    m3_metrics = evaluate(bid_model, X3_te, y3_te, "Bid Response Model")

    # 5. Confidence interval estimation (using residuals)
    # Compute residual std on held-out set for order model
    y1_pred = order_model.predict(X1_te)
    residuals = y1_te.values - y1_pred
    order_residual_std = float(np.std(residuals))

    y2_pred = rev_model.predict(X2_te)
    rev_residuals = y2_te.values - y2_pred
    rev_residual_std = float(np.std(rev_residuals))

    # 6. Save everything
    print("\nSaving models...")
    with open(os.path.join(MODELS_DIR, "order_model.pkl"),  "wb") as f: pickle.dump(order_model, f)
    with open(os.path.join(MODELS_DIR, "revenue_model.pkl"),"wb") as f: pickle.dump(rev_model,   f)
    with open(os.path.join(MODELS_DIR, "bid_model.pkl"),    "wb") as f: pickle.dump(bid_model,   f)
    with open(os.path.join(MODELS_DIR, "encoders.pkl"),     "wb") as f: pickle.dump((cat_enc, city_enc), f)
    with open(os.path.join(MODELS_DIR, "feature_importance.pkl"), "wb") as f:
        pickle.dump({"order": imp1, "revenue": imp2}, f)
    with open(os.path.join(MODELS_DIR, "data_stats.pkl"), "wb") as f:
        pickle.dump({
            "stats": stats, "cat_stats": cat_stats, "city_stats": city_stats,
            "metrics": {"order": m1_metrics, "revenue": m2_metrics, "bid": m3_metrics},
            "order_residual_std": order_residual_std,
            "rev_residual_std":   rev_residual_std,
        }, f)

    print("\n✅ All models saved to models/")
    print(f"   Order Model   R² = {m1_metrics['r2']}")
    print(f"   Revenue Model R² = {m2_metrics['r2']}")
    print(f"   Bid Model     R² = {m3_metrics['r2']}")


if __name__ == "__main__":
    train()
