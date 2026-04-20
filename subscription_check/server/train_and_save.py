"""
로컬에서 한 번만 실행하는 스크립트.
모델을 학습하고 model.cbm + stats.json 으로 저장합니다.

실행:
  cd subscription_check/server
  source .venv/bin/activate
  python train_and_save.py
"""

import os, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

USE_FREQ_MAP = {"rare": 1, "monthly": 2, "weekly": 3, "frequent": 4}
RECENCY_MAP  = {">30d": 1, "7-30d": 2, "1-7d": 3, "<1d": 4}

cat_cols = ["subscription_type", "use_frequency", "last_use_recency"]
num_cols = [
    "effective_monthly_cost", "perceived_necessity", "cost_burden", "would_rebuy",
    "replacement_available", "billing_cycle", "remaining_months", "discount_amount",
    "log_monthly_cost", "monthly_cost_z", "value_gap", "rebuy_satisfaction_gap",
    "cost_to_necessity_ratio", "cost_burden_x_replacement", "necessity_x_recency",
    "frequency_x_rebuy", "is_high_cost", "has_churn_signal", "is_zero_cost",
]

SERVER_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SERVER_DIR, "model.cbm")
STATS_PATH = os.path.join(SERVER_DIR, "stats.json")


def feature_engineer(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    df["use_frequency_score"]    = df["use_frequency"].map(USE_FREQ_MAP)
    df["last_use_recency_score"] = df["last_use_recency"].map(RECENCY_MAP)
    df["effective_monthly_cost"]  = (df["monthly_cost"] - df["discount_amount"]).clip(lower=0)
    df["is_zero_cost"]            = (df["effective_monthly_cost"] == 0).astype(int)
    df["log_monthly_cost"]        = np.log1p(df["effective_monthly_cost"])
    df["monthly_cost_z"]          = (
        (df["effective_monthly_cost"] - stats["emc_mean"]) / stats["emc_std"]
    )
    df["rebuy_satisfaction_gap"]  = df["effective_monthly_cost"] - df["perceived_necessity"]
    df["cost_to_necessity_ratio"] = df["effective_monthly_cost"] * (df["perceived_necessity"] + 1e-6)
    df["value_gap"] = (
        df["use_frequency_score"] + df["last_use_recency_score"]
        + df["perceived_necessity"] - df["cost_burden"]
    )
    df["cost_burden_x_replacement"] = df["cost_burden"] * df["replacement_available"]
    df["necessity_x_recency"]       = df["perceived_necessity"] * df["last_use_recency_score"]
    df["frequency_x_rebuy"]         = df["use_frequency_score"] * df["would_rebuy"]
    df["is_high_cost"]   = (df["effective_monthly_cost"] > stats["high_cost_threshold"]).astype(int)
    df["has_churn_signal"] = (
        (df["use_frequency_score"] <= 2)
        & (df["last_use_recency_score"] <= 2)
        & (df["cost_burden"] >= 4)
    ).astype(int)
    return df


def main():
    csv_path = os.path.abspath(os.path.join(SERVER_DIR, "..", "..", "mock_data_3.csv"))
    print(f"[INFO] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    df["use_frequency_score"]    = df["use_frequency"].map(USE_FREQ_MAP)
    df["last_use_recency_score"] = df["last_use_recency"].map(RECENCY_MAP)
    df["effective_monthly_cost"] = (df["monthly_cost"] - df["discount_amount"]).clip(lower=0)

    stats = {
        "emc_mean":             float(df["effective_monthly_cost"].mean()),
        "emc_std":              float(df["effective_monthly_cost"].std(ddof=0)),
        "high_cost_threshold":  float(df["effective_monthly_cost"].quantile(0.75)),
    }
    print(f"[INFO] Stats — mean: {stats['emc_mean']:.1f}, "
          f"std: {stats['emc_std']:.1f}, "
          f"high_cost_75p: {stats['high_cost_threshold']:.1f}")

    df = feature_engineer(df, stats)

    feature_cols = cat_cols + num_cols
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y = df["target"].astype(int)
    for c in cat_cols:
        X[c] = X[c].astype(str)

    cat_feature_idx = [X.columns.get_loc(c) for c in cat_cols]
    neg, pos = np.bincount(y)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.018051353424731104,
        depth=7,
        l2_leaf_reg=3.937323054801693,
        bagging_temperature=0.4970853936108525,
        random_strength=2.23973725904685,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1.0, neg / pos],
        random_seed=42,
        verbose=200,
    )
    print(f"[INFO] Training CatBoost on {len(X)} samples...")
    model.fit(X, y, cat_features=cat_feature_idx)

    model.save_model(MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Stats saved  → {STATS_PATH}")


if __name__ == "__main__":
    main()
