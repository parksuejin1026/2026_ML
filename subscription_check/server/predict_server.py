"""
CatBoost 추론 서버 + 피드백 수집 + 재학습 엔드포인트

엔드포인트:
  POST /predict          단건 예측 (DB 저장)
  POST /predict_batch    다건 예측 (DB 저장)
  POST /feedback         실제 결과 피드백 (target 업데이트)
  POST /retrain          즉시 재학습 트리거 (스케줄러 / 관리자용)
  GET  /stats            누적 데이터 통계
  GET  /health           헬스체크
"""

import os
import json
import csv
import io
import calendar
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

from db import init_db, session_scope, Prediction, AppSetting, AppSubscription
from retrain import (
    feature_engineer, run_retrain,
    CAT_COLS, NUM_COLS, USE_FREQ_MAP, RECENCY_MAP,
)

app = Flask(__name__)
CORS(app)

# ── 전역 변수 ────────────────────────────────────────────────────────────────
model: CatBoostClassifier = None
dataset_stats: dict = {}
current_model_version: str = "initial"

SERVER_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SERVER_DIR, "model.cbm")
STATS_PATH = os.path.join(SERVER_DIR, "stats.json")

RETRAIN_TOKEN = os.environ.get("RETRAIN_TOKEN")  # 설정 시 /retrain 호출에 필요
MODEL_CHURN_THRESHOLD = float(os.environ.get("MODEL_CHURN_THRESHOLD", 0.4))

# 스케줄 재학습: 매일 RETRAIN_HOUR 시(UTC)에 실행 (기본 18 UTC = KST 새벽 3시)
RETRAIN_HOUR = int(os.environ.get("RETRAIN_HOUR", 18))
RETRAIN_MIN_FEEDBACK = int(os.environ.get("RETRAIN_MIN_FEEDBACK", 10))


# ── 모델 로드 ────────────────────────────────────────────────────────────────
def load_model():
    global model, dataset_stats, current_model_version

    if not (os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH)):
        raise FileNotFoundError(
            "model.cbm / stats.json 이 없습니다. 먼저 `python retrain.py` 를 실행하세요."
        )

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    with open(STATS_PATH) as f:
        dataset_stats.clear()
        dataset_stats.update(json.load(f))

    # 모델 버전 = 파일 mtime 을 유닉스 초로 단순 표기
    current_model_version = datetime.utcfromtimestamp(
        os.path.getmtime(MODEL_PATH)
    ).strftime("%Y%m%d-%H%M%S")

    print(f"[SERVER] Model loaded version={current_model_version} stats={dataset_stats}")


# ── 이유 문구 ────────────────────────────────────────────────────────────────
def build_reason(row: dict, is_churn: bool) -> str:
    freq    = USE_FREQ_MAP.get(row.get("use_frequency", ""), 0)
    recency = RECENCY_MAP.get(row.get("last_use_recency", ""), 0)

    if not is_churn:
        reasons = []
        if freq >= 3:    reasons.append("이용 빈도 높음")
        if recency >= 3: reasons.append("최근 사용 이력 있음")
        if row.get("perceived_necessity", 0) >= 4: reasons.append("체감 필요도 높음")
        if row.get("would_rebuy", 0) >= 4:          reasons.append("재구독 의향 높음")
        if row.get("cost_burden", 0) <= 2:          reasons.append("비용 부담 낮음")
        return " · ".join(reasons[:3]) if reasons else "전반적으로 유지 가치가 있습니다."

    reasons = []
    if freq <= 2:    reasons.append("이용 빈도 낮음")
    if recency <= 2: reasons.append("오래 미사용")
    emc = max(0, row.get("monthly_cost", 0) - row.get("discount_amount", 0))
    if emc >= dataset_stats.get("high_cost_threshold", 18500):
        reasons.append("실질 비용 높음")
    if row.get("would_rebuy", 0) <= 2:          reasons.append("재구독 의향 낮음")
    if row.get("cost_burden", 0) >= 4:          reasons.append("비용 부담 큼")
    if row.get("replacement_available", 0) == 1: reasons.append("대체 서비스 존재")
    return " · ".join(reasons[:4]) if reasons else "모델 판단: 해지 고려 대상"


# ── 자동 추정 보조 ─────────────────────────────────────────────────────────
def _estimate_cost_burden(monthly_cost: int) -> int:
    if monthly_cost <= 5000:  return 1
    if monthly_cost <= 10000: return 2
    if monthly_cost <= 20000: return 3
    if monthly_cost <= 30000: return 4
    return 5


def _estimate_would_rebuy(freq_score: int, necessity: int) -> int:
    return max(1, min(5, round((freq_score + necessity) / 2)))


# ── 단건 예측 (DB 저장 포함) ───────────────────────────────────────────────
def predict_and_log(raw: dict, session, device_id: str | None = None) -> dict:
    monthly_cost        = int(raw.get("monthly_cost", 0))
    use_frequency       = raw.get("use_frequency", "monthly")
    perceived_necessity = int(raw.get("perceived_necessity", 3))
    freq_score          = USE_FREQ_MAP.get(use_frequency, 2)

    row = {
        "subscription_type":     raw.get("subscription_type", "Video"),
        "monthly_cost":          monthly_cost,
        "use_frequency":         use_frequency,
        "last_use_recency":      raw.get("last_use_recency", "7-30d"),
        "perceived_necessity":   perceived_necessity,
        "cost_burden":           int(raw["cost_burden"]) if raw.get("cost_burden") is not None else _estimate_cost_burden(monthly_cost),
        "would_rebuy":           int(raw["would_rebuy"]) if raw.get("would_rebuy") is not None else _estimate_would_rebuy(freq_score, perceived_necessity),
        "replacement_available": int(raw.get("replacement_available", 0)),
        "billing_cycle":         int(raw.get("billing_cycle", 0)),
        "remaining_months":      float(raw.get("remaining_months", 0)),
        "discount_amount":       int(raw.get("discount_amount", 0)),
    }

    df = feature_engineer(pd.DataFrame([row]), dataset_stats)
    feature_cols = [c for c in CAT_COLS + NUM_COLS if c in df.columns]
    X = df[feature_cols].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str)

    proba = model.predict_proba(X)[0]
    prob_churn = float(proba[0])
    prob_keep = float(proba[1])
    threshold = float(dataset_stats.get("churn_threshold", MODEL_CHURN_THRESHOLD))

    is_churn = prob_churn >= threshold
    confidence = prob_churn if is_churn else prob_keep

    # DB 저장
    record = Prediction(
        **row,
        device_id=device_id,
        predicted_churn=is_churn,
        predicted_confidence=round(confidence, 4),
        model_version=current_model_version,
    )
    session.add(record)
    session.flush()  # id 채번

    return {
        "prediction_id":      record.id,
        "is_churn_candidate": is_churn,
        "confidence":         round(confidence, 4),
        "reason":             build_reason(row, is_churn),
    }


# ── 기기 식별 헬퍼 ────────────────────────────────────────────────────────
def _require_device_id() -> str:
    """X-Device-Id 헤더 필수. 없으면 400."""
    device_id = request.headers.get("X-Device-Id", "").strip()
    if not device_id:
        return ""
    return device_id


def _parse_optional_datetime(value):
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed
    raise ValueError("expected ISO datetime string")


def _coerce_subscription_payload(data: dict) -> dict:
    payload = {k: v for k, v in data.items() if k in SUBSCRIPTION_WRITABLE_FIELDS}
    for key in ("next_billing_at", "trial_ends_at", "discount_ends_at"):
        if key in payload:
            payload[key] = _parse_optional_datetime(payload[key])

    if "billing_day" in payload and payload["billing_day"] is not None:
        billing_day = int(payload["billing_day"])
        if billing_day < 1 or billing_day > 31:
            raise ValueError("billing_day must be between 1 and 31")
        payload["billing_day"] = billing_day

    if "renewal_notice_days" in payload and payload["renewal_notice_days"] is not None:
        payload["renewal_notice_days"] = max(0, int(payload["renewal_notice_days"]))

    return payload


def _effective_monthly_cost(sub: AppSubscription) -> int:
    return max(0, int(sub.monthly_cost or 0) - int(sub.discount_amount or 0))


def _subscription_prediction_payload(sub: AppSubscription) -> dict:
    return {
        "id": str(sub.id),
        "subscription_type": sub.subscription_type,
        "monthly_cost": sub.monthly_cost,
        "use_frequency": sub.use_frequency,
        "last_use_recency": sub.last_use_recency,
        "perceived_necessity": sub.perceived_necessity,
        "cost_burden": sub.cost_burden,
        "would_rebuy": sub.would_rebuy,
        "replacement_available": 1 if sub.replacement_available else 0,
        "billing_cycle": 1 if sub.is_annual else 0,
        "remaining_months": sub.remaining_months or 0,
        "discount_amount": sub.discount_amount or 0,
    }


def _is_in_month(value: datetime | None, year: int, month: int) -> bool:
    return value is not None and value.year == year and value.month == month


def _billing_datetime_for_month(sub: AppSubscription, year: int, month: int):
    if _is_in_month(sub.next_billing_at, year, month):
        return sub.next_billing_at
    if not sub.billing_day:
        return None
    day = min(int(sub.billing_day), calendar.monthrange(year, month)[1])
    return datetime(year, month, day)


def _subscription_event(sub: AppSubscription, scheduled_at: datetime, event_type: str):
    return {
        "id": f"{event_type}-{sub.id}-{scheduled_at.date().isoformat()}",
        "subscription_id": str(sub.id),
        "name": sub.name,
        "emoji": sub.emoji,
        "amount": _effective_monthly_cost(sub),
        "event_type": event_type,
        "scheduled_at": scheduled_at.isoformat(),
        "renewal_notice_days": sub.renewal_notice_days,
    }


# ── 구독 CRUD 엔드포인트 ─────────────────────────────────────────────────
SUBSCRIPTION_WRITABLE_FIELDS = {
    "name", "emoji", "subscription_type", "monthly_cost", "use_frequency",
    "last_use_recency", "perceived_necessity", "cost_burden", "would_rebuy",
    "replacement_available", "is_annual", "remaining_months", "discount_amount",
    "billing_day", "next_billing_at", "trial_ends_at", "discount_ends_at",
    "renewal_notice_days",
}


@app.route("/subscriptions", methods=["GET"])
def list_subscriptions():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        rows = (
            session.query(AppSubscription)
            .filter(
                AppSubscription.device_id == device_id,
                AppSubscription.archived_at.is_(None),
            )
            .order_by(AppSubscription.created_at.asc())
            .all()
        )
        return jsonify([r.to_dict() for r in rows])


@app.route("/subscriptions", methods=["POST"])
def create_subscription():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    data = request.get_json(silent=True) or {}
    try:
        payload = _coerce_subscription_payload(data)
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    required = {"name", "subscription_type", "monthly_cost",
                "use_frequency", "last_use_recency", "perceived_necessity"}
    missing = required - payload.keys()
    if missing:
        return jsonify({"error": f"missing fields: {sorted(missing)}"}), 400

    with session_scope() as session:
        row = AppSubscription(device_id=device_id, **payload)
        session.add(row)
        session.flush()
        result = row.to_dict()
    return jsonify(result), 201


@app.route("/subscriptions/<int:sub_id>", methods=["PATCH"])
def update_subscription(sub_id: int):
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    data = request.get_json(silent=True) or {}
    try:
        updates = _coerce_subscription_payload(data)
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    if not updates:
        return jsonify({"error": "no updatable fields"}), 400

    with session_scope() as session:
        row = (
            session.query(AppSubscription)
            .filter(AppSubscription.id == sub_id,
                    AppSubscription.device_id == device_id,
                    AppSubscription.archived_at.is_(None))
            .one_or_none()
        )
        if row is None:
            return jsonify({"error": "subscription not found"}), 404
        for k, v in updates.items():
            setattr(row, k, v)
        session.flush()
        result = row.to_dict()
    return jsonify(result)


@app.route("/subscriptions/<int:sub_id>", methods=["DELETE"])
def delete_subscription(sub_id: int):
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        row = (
            session.query(AppSubscription)
            .filter(AppSubscription.id == sub_id,
                    AppSubscription.device_id == device_id,
                    AppSubscription.archived_at.is_(None))
            .one_or_none()
        )
        if row is None:
            return jsonify({"error": "subscription not found"}), 404
        session.delete(row)
    return jsonify({"ok": True})


@app.route("/archived_subscriptions", methods=["GET"])
def archived_subscriptions():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        rows = (
            session.query(AppSubscription)
            .filter(
                AppSubscription.device_id == device_id,
                AppSubscription.archived_at.isnot(None),
            )
            .order_by(AppSubscription.archived_at.desc())
            .all()
        )
        return jsonify([r.to_dict() for r in rows])


@app.route("/subscriptions/<int:sub_id>/archive", methods=["POST"])
def archive_subscription(sub_id: int):
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        row = (
            session.query(AppSubscription)
            .filter(AppSubscription.id == sub_id,
                    AppSubscription.device_id == device_id)
            .one_or_none()
        )
        if row is None:
            return jsonify({"error": "subscription not found"}), 404
        row.archived_at = datetime.utcnow()
        session.flush()
        result = row.to_dict()
    return jsonify(result)


@app.route("/subscriptions/<int:sub_id>/restore", methods=["POST"])
def restore_subscription(sub_id: int):
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        row = (
            session.query(AppSubscription)
            .filter(AppSubscription.id == sub_id,
                    AppSubscription.device_id == device_id)
            .one_or_none()
        )
        if row is None:
            return jsonify({"error": "subscription not found"}), 404
        row.archived_at = None
        session.flush()
        result = row.to_dict()
    return jsonify(result)


@app.route("/billing_events", methods=["GET"])
def billing_events():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    try:
        year = int(request.args.get("year", datetime.utcnow().year))
        month = int(request.args.get("month", datetime.utcnow().month))
        if month < 1 or month > 12:
            raise ValueError
    except ValueError:
        return jsonify({"error": "year and month must be valid integers"}), 400

    with session_scope() as session:
        rows = (
            session.query(AppSubscription)
            .filter(
                AppSubscription.device_id == device_id,
                AppSubscription.archived_at.is_(None),
            )
            .order_by(AppSubscription.created_at.asc())
            .all()
        )

        events = []
        for sub in rows:
            scheduled_at = _billing_datetime_for_month(sub, year, month)
            if scheduled_at is not None:
                event_type = "annual_renewal" if sub.is_annual else "monthly_payment"
                events.append(_subscription_event(sub, scheduled_at, event_type))
            if _is_in_month(sub.trial_ends_at, year, month):
                events.append(_subscription_event(sub, sub.trial_ends_at, "trial_ends"))
            if _is_in_month(sub.discount_ends_at, year, month):
                events.append(_subscription_event(sub, sub.discount_ends_at, "discount_ends"))

    events.sort(key=lambda item: item["scheduled_at"])
    return jsonify(events)


SETTING_WRITABLE_FIELDS = {
    "billing_alert_enabled", "weekly_report_enabled", "app_lock_enabled",
    "biometric_lock_enabled", "terms_accepted_at", "default_notice_days",
}


def _settings_payload(data: dict) -> dict:
    payload = {k: v for k, v in data.items() if k in SETTING_WRITABLE_FIELDS}
    if "terms_accepted_at" in payload:
        payload["terms_accepted_at"] = _parse_optional_datetime(
            payload["terms_accepted_at"])
    if "default_notice_days" in payload and payload["default_notice_days"] is not None:
        payload["default_notice_days"] = max(0, int(payload["default_notice_days"]))
    return payload


def _get_or_create_settings(session, device_id: str) -> AppSetting:
    row = (
        session.query(AppSetting)
        .filter(AppSetting.device_id == device_id)
        .one_or_none()
    )
    if row is None:
        row = AppSetting(device_id=device_id)
        session.add(row)
        session.flush()
    return row


@app.route("/settings", methods=["GET"])
def get_settings():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        row = _get_or_create_settings(session, device_id)
        result = row.to_dict()
    return jsonify(result)


@app.route("/settings", methods=["PATCH"])
def patch_settings():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    data = request.get_json(silent=True) or {}
    try:
        updates = _settings_payload(data)
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    if not updates:
        return jsonify({"error": "no updatable fields"}), 400

    with session_scope() as session:
        row = _get_or_create_settings(session, device_id)
        for key, value in updates.items():
            setattr(row, key, value)
        session.flush()
        result = row.to_dict()
    return jsonify(result)


@app.route("/export/subscriptions.json", methods=["GET"])
def export_subscriptions_json():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        rows = (
            session.query(AppSubscription)
            .filter(AppSubscription.device_id == device_id)
            .order_by(AppSubscription.created_at.asc())
            .all()
        )
        return jsonify([r.to_dict() for r in rows])


@app.route("/export/subscriptions.csv", methods=["GET"])
def export_subscriptions_csv():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        rows = (
            session.query(AppSubscription)
            .filter(AppSubscription.device_id == device_id)
            .order_by(AppSubscription.created_at.asc())
            .all()
        )
        output = io.StringIO()
        fieldnames = [
            "id", "name", "emoji", "subscription_type", "monthly_cost",
            "effective_monthly_cost", "use_frequency", "last_use_recency",
            "perceived_necessity", "replacement_available", "is_annual",
            "billing_day", "next_billing_at", "trial_ends_at",
            "discount_ends_at", "renewal_notice_days", "archived_at",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = row.to_dict()
            data["effective_monthly_cost"] = _effective_monthly_cost(row)
            writer.writerow({key: data.get(key) for key in fieldnames})

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=subscriptions.csv"},
    )


@app.route("/recommendations", methods=["GET"])
def recommendations():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        rows = (
            session.query(AppSubscription)
            .filter(
                AppSubscription.device_id == device_id,
                AppSubscription.archived_at.is_(None),
            )
            .order_by(AppSubscription.created_at.asc())
            .all()
        )

        items = []
        for sub in rows:
            result = predict_and_log(
                _subscription_prediction_payload(sub),
                session,
                device_id=device_id,
            )
            items.append({
                "subscription": sub.to_dict(),
                "prediction_id": result["prediction_id"],
                "is_churn_candidate": result["is_churn_candidate"],
                "confidence": result["confidence"],
                "reason": result["reason"],
                "estimated_monthly_savings": (
                    _effective_monthly_cost(sub)
                    if result["is_churn_candidate"] else 0
                ),
                "recommended_action": (
                    "cancel" if result["is_churn_candidate"] else "keep"
                ),
            })

    items.sort(
        key=lambda item: (
            0 if item["is_churn_candidate"] else 1,
            -item["confidence"],
            -item["estimated_monthly_savings"],
        )
    )
    for index, item in enumerate(items, start=1):
        item["rank"] = index
    return jsonify(items)


# ── 엔드포인트 ────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    device_id = _require_device_id() or None
    with session_scope() as session:
        result = predict_and_log(data, session, device_id=device_id)
    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    items = request.get_json(silent=True)
    if not isinstance(items, list):
        return jsonify({"error": "Expected JSON array"}), 400

    device_id = _require_device_id() or None
    results = {}
    with session_scope() as session:
        for item in items:
            sid = item.get("id", "")
            results[sid] = predict_and_log(item, session, device_id=device_id)
    return jsonify(results)


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Body: {
      "prediction_id": int,
      "actual_kept":   bool,
      "subscription_id": int (optional) — 구독 카드의 피드백 UI 상태 영구 저장용
    }
    actual_kept=true  → target=1 (유지)
    actual_kept=false → target=0 (해지)
    """
    data = request.get_json(silent=True) or {}
    pid = data.get("prediction_id")
    if pid is None or "actual_kept" not in data:
        return jsonify({"error": "prediction_id, actual_kept required"}), 400

    kept   = bool(data["actual_kept"])
    target = 1 if kept else 0
    sub_id = data.get("subscription_id")
    now    = datetime.utcnow()

    # subscription_id 와 함께 보낸 경우, device_id 소유권도 검증
    device_id = _require_device_id() or None

    with session_scope() as session:
        row = session.get(Prediction, int(pid))
        if row is None:
            return jsonify({"error": "prediction not found"}), 404
        row.actual_target = target
        row.feedback_at   = now

        if sub_id is not None:
            sub = session.get(AppSubscription, int(sub_id))
            if sub is not None and (device_id is None or sub.device_id == device_id):
                sub.last_feedback_kept = kept
                sub.last_feedback_at   = now

    return jsonify({
        "ok": True,
        "prediction_id":  int(pid),
        "actual_target":  target,
        "last_feedback_kept": kept,
        "last_feedback_at":   now.isoformat(),
    })


@app.route("/retrain", methods=["POST"])
def retrain_endpoint():
    """스케줄러 또는 관리자가 호출. RETRAIN_TOKEN 이 설정되어 있으면 헤더로 검증."""
    if RETRAIN_TOKEN:
        token = request.headers.get("X-Retrain-Token", "")
        if token != RETRAIN_TOKEN:
            return jsonify({"error": "unauthorized"}), 401

    result = run_retrain()
    load_model()  # 새 모델 즉시 반영
    return jsonify({"ok": True, **result, "loaded_version": current_model_version})


@app.route("/stats", methods=["GET"])
def stats():
    with session_scope() as session:
        total     = session.query(Prediction).count()
        labeled   = session.query(Prediction).filter(Prediction.actual_target.isnot(None)).count()
        kept      = session.query(Prediction).filter(Prediction.actual_target == 1).count()
        churned   = session.query(Prediction).filter(Prediction.actual_target == 0).count()

    return jsonify({
        "total_predictions": total,
        "labeled_samples":   labeled,
        "kept_count":        kept,
        "churned_count":     churned,
        "model_version":     current_model_version,
    })


@app.route("/savings", methods=["GET"])
def savings():
    """
    기기별 누적 절감액 (해지 피드백 기반).

    응답:
      - cancelled_count     : 해지한 예측 건수 (actual_target=0)
      - kept_count          : 유지한 예측 건수
      - monthly_savings     : 해지 구독의 실질 월 구독료 합계 (= 이 달 절약되는 금액)
      - cumulative_savings  : 해지 시점부터 지금까지 누적 절약 추정치
      - history             : 최근 해지 내역 (최신순 20건)
    """
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        rows = (
            session.query(Prediction)
            .filter(
                Prediction.device_id == device_id,
                Prediction.actual_target.isnot(None),
                Prediction.feedback_at.isnot(None),
            )
            .order_by(Prediction.feedback_at.desc())
            .all()
        )

        cancelled_count = 0
        kept_count = 0
        monthly_savings = 0
        cumulative_savings = 0
        history: list[dict] = []

        now = datetime.utcnow()
        DAYS_PER_MONTH = 30

        for r in rows:
            if r.actual_target == 1:
                kept_count += 1
                continue

            cancelled_count += 1
            monthly_cost = int(r.monthly_cost or 0)
            discount = int(r.discount_amount or 0)
            effective = max(0, monthly_cost - discount)
            monthly_savings += effective

            if r.feedback_at is not None:
                days_since = max(0, (now - r.feedback_at).days)
                months_since = max(1, round(days_since / DAYS_PER_MONTH) or 1)
            else:
                months_since = 1
            cumulative_savings += effective * months_since

            if len(history) < 20:
                history.append({
                    "prediction_id":     r.id,
                    "subscription_type": r.subscription_type,
                    "monthly_cost":      monthly_cost,
                    "discount_amount":   discount,
                    "effective_monthly": effective,
                    "feedback_at":       r.feedback_at.isoformat() if r.feedback_at else None,
                })

    return jsonify({
        "cancelled_count":    cancelled_count,
        "kept_count":         kept_count,
        "monthly_savings":    monthly_savings,
        "cumulative_savings": cumulative_savings,
        "history":            history,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "model_loaded":  model is not None,
        "model_version": current_model_version,
    })


# ── 스케줄러 ──────────────────────────────────────────────────────────────
def scheduled_retrain_job():
    """백그라운드 스레드에서 실행. 새 피드백이 충분할 때만 재학습."""
    try:
        with session_scope() as session:
            labeled = (
                session.query(Prediction)
                .filter(Prediction.actual_target.isnot(None))
                .count()
            )
        if labeled < RETRAIN_MIN_FEEDBACK:
            print(f"[SCHEDULER] Skip retrain (labeled={labeled} < min={RETRAIN_MIN_FEEDBACK})")
            return
        print(f"[SCHEDULER] Retraining with {labeled} labeled samples...")
        run_retrain()
        load_model()
        print(f"[SCHEDULER] Retrain done. version={current_model_version}")
    except Exception as e:
        print(f"[SCHEDULER] Retrain failed: {e}")


# ── 부트스트랩 ────────────────────────────────────────────────────────────
def bootstrap():
    """서버 시작 전 필수 초기화."""
    init_db()
    load_model()

    scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
    scheduler.add_job(
        scheduled_retrain_job,
        trigger="cron",
        hour=RETRAIN_HOUR,
        minute=0,
        id="daily_retrain",
    )
    scheduler.start()
    print(f"[SCHEDULER] Daily retrain scheduled at {RETRAIN_HOUR}:00 UTC")


bootstrap()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"[SERVER] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
