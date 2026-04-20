"""
CatBoost 추론 서버
- train_and_save.py 로 생성한 model.cbm + stats.json 을 로드
- POST /predict 로 단건 예측
- POST /predict_batch 로 다건 예측
"""

import os, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── 전역 변수 ────────────────────────────────────────────────────────────────
model: CatBoostClassifier = None
dataset_stats: dict = {}
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

USE_FREQ_MAP = {"rare": 1, "monthly": 2, "weekly": 3, "frequent": 4}
RECENCY_MAP  = {">30d": 1, "7-30d": 2, "1-7d": 3, "<1d": 4}


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """노트북과 동일한 피처 엔지니어링"""
    df = df.copy()
    df["use_frequency_score"]    = df["use_frequency"].map(USE_FREQ_MAP)
    df["last_use_recency_score"] = df["last_use_recency"].map(RECENCY_MAP)
    df["effective_monthly_cost"]  = (df["monthly_cost"] - df["discount_amount"]).clip(lower=0)
    df["is_zero_cost"]            = (df["effective_monthly_cost"] == 0).astype(int)
    df["log_monthly_cost"]        = np.log1p(df["effective_monthly_cost"])
    df["monthly_cost_z"]          = (
        (df["effective_monthly_cost"] - dataset_stats["emc_mean"])
        / dataset_stats["emc_std"]
    )
    df["rebuy_satisfaction_gap"]  = df["effective_monthly_cost"] - df["perceived_necessity"]
    df["cost_to_necessity_ratio"] = df["effective_monthly_cost"] * (df["perceived_necessity"] + 1e-6)
    df["value_gap"] = (
        df["use_frequency_score"]
        + df["last_use_recency_score"]
        + df["perceived_necessity"]
        - df["cost_burden"]
    )
    df["cost_burden_x_replacement"] = df["cost_burden"] * df["replacement_available"]
    df["necessity_x_recency"]       = df["perceived_necessity"] * df["last_use_recency_score"]
    df["frequency_x_rebuy"]         = df["use_frequency_score"] * df["would_rebuy"]
    df["is_high_cost"]   = (df["effective_monthly_cost"] > dataset_stats["high_cost_threshold"]).astype(int)
    df["has_churn_signal"] = (
        (df["use_frequency_score"] <= 2)
        & (df["last_use_recency_score"] <= 2)
        & (df["cost_burden"] >= 4)
    ).astype(int)
    return df


def load_model():
    """저장된 model.cbm + stats.json 로드"""
    global model, dataset_stats

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"model.cbm 파일이 없습니다: {MODEL_PATH}\n"
            "로컬에서 `python train_and_save.py` 를 먼저 실행하세요."
        )
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"stats.json 파일이 없습니다: {STATS_PATH}\n"
            "로컬에서 `python train_and_save.py` 를 먼저 실행하세요."
        )

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print(f"[INFO] Model loaded ← {MODEL_PATH}")

    with open(STATS_PATH) as f:
        dataset_stats.update(json.load(f))
    print(f"[INFO] Stats loaded ← {STATS_PATH} : {dataset_stats}")


def build_reason(row: dict, is_churn: bool, proba: float) -> str:
    """피처 값을 기반으로 판정 이유 생성"""
    if not is_churn:
        reasons = []
        freq = USE_FREQ_MAP.get(row.get("use_frequency", ""), 0)
        recency = RECENCY_MAP.get(row.get("last_use_recency", ""), 0)
        if freq >= 3:
            reasons.append("이용 빈도 높음")
        if recency >= 3:
            reasons.append("최근 사용 이력 있음")
        if row.get("perceived_necessity", 0) >= 4:
            reasons.append("체감 필요도 높음")
        if row.get("would_rebuy", 0) >= 4:
            reasons.append("재구독 의향 높음")
        if row.get("cost_burden", 0) <= 2:
            reasons.append("비용 부담 낮음")
        return reasons[0:3] and " · ".join(reasons[:3]) or "전반적으로 유지 가치가 있습니다."

    reasons = []
    freq = USE_FREQ_MAP.get(row.get("use_frequency", ""), 0)
    recency = RECENCY_MAP.get(row.get("last_use_recency", ""), 0)
    if freq <= 2:
        reasons.append("이용 빈도 낮음")
    if recency <= 2:
        reasons.append("오래 미사용")
    emc = max(0, row.get("monthly_cost", 0) - row.get("discount_amount", 0))
    if emc >= dataset_stats.get("high_cost_threshold", 18500):
        reasons.append("실질 비용 높음")
    if row.get("would_rebuy", 0) <= 2:
        reasons.append("재구독 의향 낮음")
    if row.get("cost_burden", 0) >= 4:
        reasons.append("비용 부담 큼")
    if row.get("replacement_available", 0) == 1:
        reasons.append("대체 서비스 존재")
    return " · ".join(reasons[:4]) if reasons else "모델 판단: 해지 고려 대상"


def _estimate_cost_burden(monthly_cost: int) -> int:
    """월 구독료 기반 비용 부담 자동 추정"""
    if monthly_cost <= 5000:
        return 1
    if monthly_cost <= 10000:
        return 2
    if monthly_cost <= 20000:
        return 3
    if monthly_cost <= 30000:
        return 4
    return 5


def _estimate_would_rebuy(freq_score: int, necessity: int) -> int:
    """이용 빈도 + 체감 필요도 기반 재구독 의향 추정"""
    return max(1, min(5, round((freq_score + necessity) / 2)))


def predict_single(raw: dict) -> dict:
    """단건 예측: raw 입력 → 미입력 필드 자동 추정 → 피처 엔지니어링 → 모델 추론"""
    monthly_cost = int(raw.get("monthly_cost", 0))
    use_frequency = raw.get("use_frequency", "monthly")
    perceived_necessity = int(raw.get("perceived_necessity", 3))
    freq_score = USE_FREQ_MAP.get(use_frequency, 2)

    row = {
        "subscription_type":    raw.get("subscription_type", "Video"),
        "monthly_cost":         monthly_cost,
        "use_frequency":        use_frequency,
        "last_use_recency":     raw.get("last_use_recency", "7-30d"),
        "perceived_necessity":  perceived_necessity,
        "cost_burden":          int(raw["cost_burden"]) if "cost_burden" in raw else _estimate_cost_burden(monthly_cost),
        "would_rebuy":          int(raw["would_rebuy"]) if "would_rebuy" in raw else _estimate_would_rebuy(freq_score, perceived_necessity),
        "replacement_available": int(raw.get("replacement_available", 0)),
        "billing_cycle":        int(raw.get("billing_cycle", 0)),
        "remaining_months":     float(raw.get("remaining_months", 0)),
        "discount_amount":      int(raw.get("discount_amount", 0)),
    }
    df = pd.DataFrame([row])
    df = feature_engineer(df)

    feature_cols = cat_cols + num_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)

    pred = int(model.predict(X).ravel()[0])
    proba_arr = model.predict_proba(X)[0]  # [prob_class0, prob_class1]
    prob_keep = float(proba_arr[1])

    # target: 0=해지 후보, 1=유지 후보
    is_churn = (pred == 0)
    confidence = (1 - prob_keep) if is_churn else prob_keep

    return {
        "is_churn_candidate": is_churn,
        "confidence": round(confidence, 4),
        "reason": build_reason(row, is_churn, confidence),
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    result = predict_single(data)
    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    items = request.get_json()
    if not isinstance(items, list):
        return jsonify({"error": "Expected JSON array"}), 400
    results = {}
    for item in items:
        sid = item.get("id", "")
        results[sid] = predict_single(item)
    return jsonify(results)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    load_model()
    print("[INFO] Server starting on port 5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
