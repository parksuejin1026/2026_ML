"""
재학습 파이프라인.

데이터 소스:
  1) 원본 CSV (mock_data_3.csv) - 학습의 기본 토대
  2) DB predictions 테이블 중 actual_target 이 채워진 행 - 사용자 피드백

두 데이터를 병합해 CatBoost 모델을 학습하고 model.cbm + stats.json 으로 저장.

직접 실행:
  python retrain.py

서버에서 호출:
  from retrain import run_retrain
  run_retrain()
"""

import os
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from db import session_scope, Prediction, init_db

# ─── 경로 / 상수 ───
SERVER_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SERVER_DIR, "model.cbm")
STATS_PATH = os.path.join(SERVER_DIR, "stats.json")
CHURN_THRESHOLD = 0.4
MODEL_SOURCE = "ML_Submission3.ipynb"

SUBMISSION3_MODEL_PARAMS = {
    "iterations": 432,
    "learning_rate": 0.018,
    "depth": 5,
    "loss_function": "Logloss",
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": 200,
}


def _resolve_csv_path() -> str:
    """CSV 위치는 로컬 / Docker 환경에 따라 달라짐.
    우선순위: 환경변수 → server 디렉토리 → 레포 루트"""
    env = os.environ.get("CSV_PATH")
    if env and os.path.exists(env):
        return env
    candidates = [
        os.path.join(SERVER_DIR, "mock_data_3.csv"),
        os.path.abspath(os.path.join(SERVER_DIR, "..", "..", "mock_data_3.csv")),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # 에러 메시지용 기본값


CSV_PATH = _resolve_csv_path()

USE_FREQ_MAP = {"rare": 1, "monthly": 2, "weekly": 3, "frequent": 4}
RECENCY_MAP  = {">30d": 1, "7-30d": 2, "1-7d": 3, "<1d": 4}

CAT_COLS = ["subscription_type", "use_frequency", "last_use_recency"]
NUM_COLS = [
    "effective_monthly_cost", "perceived_necessity", "cost_burden", "would_rebuy",
    "replacement_available", "billing_cycle", "remaining_months", "discount_amount",
    "log_monthly_cost", "monthly_cost_z", "value_gap", "rebuy_satisfaction_gap",
    "cost_to_necessity_ratio", "cost_burden_x_replacement", "necessity_x_recency",
    "frequency_x_rebuy", "is_high_cost", "has_churn_signal", "is_zero_cost",
]


# ─── 피처 엔지니어링 ───
def feature_engineer(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    df["use_frequency_score"]    = df["use_frequency"].map(USE_FREQ_MAP)
    df["last_use_recency_score"] = df["last_use_recency"].map(RECENCY_MAP)
    df["effective_monthly_cost"] = (df["monthly_cost"] - df["discount_amount"]).clip(lower=0)
    df["is_zero_cost"]           = (df["effective_monthly_cost"] == 0).astype(int)
    df["log_monthly_cost"]       = np.log1p(df["effective_monthly_cost"])
    df["monthly_cost_z"]         = (
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


# ─── 데이터 로딩 ───
def load_csv_data() -> pd.DataFrame:
    print(f"[RETRAIN] CSV: {CSV_PATH}")
    return pd.read_csv(CSV_PATH)


def load_feedback_data() -> pd.DataFrame:
    """DB 에서 actual_target 이 채워진 피드백 행만 학습 포맷으로 변환."""
    with session_scope() as session:
        rows = (
            session.query(Prediction)
            .filter(Prediction.actual_target.isnot(None))
            .all()
        )
        if not rows:
            return pd.DataFrame()

        records = [{
            "subscription_type":     r.subscription_type,
            "monthly_cost":          r.monthly_cost,
            "use_frequency":         r.use_frequency,
            "last_use_recency":      r.last_use_recency,
            "perceived_necessity":   r.perceived_necessity,
            "cost_burden":           r.cost_burden,
            "would_rebuy":           r.would_rebuy,
            "replacement_available": r.replacement_available,
            "billing_cycle":         r.billing_cycle,
            "remaining_months":      r.remaining_months,
            "discount_amount":       r.discount_amount,
            "target":                r.actual_target,
        } for r in rows]
        return pd.DataFrame(records)


# ─── 핵심 학습 루틴 ───
def run_retrain() -> dict:
    """CSV + DB 피드백 데이터 병합 → 학습 → 파일 저장. 요약 정보 반환."""
    csv_df = load_csv_data()
    fb_df  = load_feedback_data()

    print(f"[RETRAIN] CSV rows: {len(csv_df)}, feedback rows: {len(fb_df)}")

    df = pd.concat([csv_df, fb_df], ignore_index=True) if len(fb_df) else csv_df

    # 통계 계산 (z-score, is_high_cost 기준)
    df["use_frequency_score"]    = df["use_frequency"].map(USE_FREQ_MAP)
    df["last_use_recency_score"] = df["last_use_recency"].map(RECENCY_MAP)
    df["effective_monthly_cost"] = (df["monthly_cost"] - df["discount_amount"]).clip(lower=0)

    stats = {
        "emc_mean":            float(df["effective_monthly_cost"].mean()),
        "emc_std":             float(df["effective_monthly_cost"].std(ddof=0)),
        "high_cost_threshold": float(df["effective_monthly_cost"].quantile(0.75)),
        "churn_threshold":     CHURN_THRESHOLD,
        "model_source":        MODEL_SOURCE,
        "model_params":        SUBMISSION3_MODEL_PARAMS,
    }

    df = feature_engineer(df, stats)

    feature_cols = [c for c in CAT_COLS + NUM_COLS if c in df.columns]
    X = df[feature_cols].copy()
    y = df["target"].astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)

    cat_feature_idx = [X.columns.get_loc(c) for c in CAT_COLS]
    model = CatBoostClassifier(**SUBMISSION3_MODEL_PARAMS)
    print(f"[RETRAIN] Training CatBoost on {len(X)} samples...")
    model.fit(X, y, cat_features=cat_feature_idx)

    model.save_model(MODEL_PATH)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    version = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    print(f"[RETRAIN] Saved model={MODEL_PATH} stats={STATS_PATH} version={version}")

    return {
        "version": version,
        "total_samples": int(len(df)),
        "csv_samples": int(len(csv_df)),
        "feedback_samples": int(len(fb_df)),
        "stats": stats,
    }


if __name__ == "__main__":
    # 로컬에서 직접 실행 시 DB 테이블 보장
    init_db()
    result = run_retrain()
    print(json.dumps(result, indent=2))
