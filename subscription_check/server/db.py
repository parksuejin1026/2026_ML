"""
SQLAlchemy 기반 DB 설정.
- Railway 의 DATABASE_URL 환경변수를 자동 사용
- 로컬 환경에서는 SQLite (subs.db) 로 fallback
"""

import os
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean,
    Float, DateTime, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

SERVER_DIR = os.path.dirname(__file__)
DEFAULT_SQLITE_PATH = os.path.join(SERVER_DIR, "subs.db")

# Railway 는 DATABASE_URL 환경변수로 PostgreSQL 연결정보를 주입
# 예: postgresql://user:pass@host:5432/dbname
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{DEFAULT_SQLITE_PATH}",
)

# Railway 가 주는 postgres:// URL 은 SQLAlchemy 2.x 에서 postgresql:// 로 교정 필요
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class AppSubscription(Base):
    """
    사용자 기기별 구독 목록. 기기 식별자(device_id)로 격리.
    """
    __tablename__ = "app_subscriptions"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    device_id  = Column(String(128), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow, nullable=False)

    name                  = Column(String(100), nullable=False)
    emoji                 = Column(String(20))
    subscription_type     = Column(String(50), nullable=False)
    monthly_cost          = Column(Integer, nullable=False)
    use_frequency         = Column(String(20), nullable=False)
    last_use_recency      = Column(String(20), nullable=False)
    perceived_necessity   = Column(Integer, nullable=False)
    cost_burden           = Column(Integer, nullable=True)
    would_rebuy           = Column(Integer, nullable=True)
    replacement_available = Column(Boolean, default=False)
    is_annual             = Column(Boolean, default=False)
    remaining_months      = Column(Float, default=0.0)
    discount_amount       = Column(Integer, default=0)
    billing_day           = Column(Integer, nullable=True)
    next_billing_at       = Column(DateTime, nullable=True)
    trial_ends_at         = Column(DateTime, nullable=True)
    discount_ends_at      = Column(DateTime, nullable=True)
    renewal_notice_days   = Column(Integer, default=3)
    archived_at           = Column(DateTime, nullable=True, index=True)

    # ── 사용자 피드백 상태 (영구 저장) ──
    # True = 유지, False = 해지, None = 피드백 없음
    last_feedback_kept    = Column(Boolean, nullable=True)
    last_feedback_at      = Column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id":                   str(self.id),
            "name":                 self.name,
            "emoji":                self.emoji,
            "subscription_type":    self.subscription_type,
            "monthly_cost":         self.monthly_cost,
            "use_frequency":        self.use_frequency,
            "last_use_recency":     self.last_use_recency,
            "perceived_necessity":  self.perceived_necessity,
            "cost_burden":          self.cost_burden,
            "would_rebuy":          self.would_rebuy,
            "replacement_available": bool(self.replacement_available),
            "is_annual":            bool(self.is_annual),
            "remaining_months":     self.remaining_months,
            "discount_amount":      self.discount_amount,
            "billing_day":          self.billing_day,
            "next_billing_at":      self.next_billing_at.isoformat()
                                      if self.next_billing_at else None,
            "trial_ends_at":        self.trial_ends_at.isoformat()
                                      if self.trial_ends_at else None,
            "discount_ends_at":     self.discount_ends_at.isoformat()
                                      if self.discount_ends_at else None,
            "renewal_notice_days":  self.renewal_notice_days,
            "archived_at":          self.archived_at.isoformat()
                                      if self.archived_at else None,
            "last_feedback_kept":   self.last_feedback_kept,
            "last_feedback_at":     self.last_feedback_at.isoformat()
                                      if self.last_feedback_at else None,
        }


class AppSetting(Base):
    """
    기기별 앱 설정. 앱 잠금 자체는 클라이언트 로컬 보안 기능이지만,
    서버 동기화가 필요한 토글/동의 상태를 저장할 수 있다.
    """
    __tablename__ = "app_settings"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    device_id  = Column(String(128), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow, nullable=False)

    billing_alert_enabled  = Column(Boolean, default=True, nullable=False)
    weekly_report_enabled  = Column(Boolean, default=False, nullable=False)
    app_lock_enabled       = Column(Boolean, default=False, nullable=False)
    biometric_lock_enabled = Column(Boolean, default=False, nullable=False)
    terms_accepted_at      = Column(DateTime, nullable=True)
    default_notice_days    = Column(Integer, default=3, nullable=False)

    def to_dict(self) -> dict:
        return {
            "billing_alert_enabled":  bool(self.billing_alert_enabled),
            "weekly_report_enabled":  bool(self.weekly_report_enabled),
            "app_lock_enabled":       bool(self.app_lock_enabled),
            "biometric_lock_enabled": bool(self.biometric_lock_enabled),
            "terms_accepted_at":      self.terms_accepted_at.isoformat()
                                      if self.terms_accepted_at else None,
            "default_notice_days":    self.default_notice_days,
        }


class Prediction(Base):
    """
    사용자가 입력한 구독 정보 + 예측 결과 + (선택적) 실제 결과 피드백.
    나중에 actual_target 이 채워진 행만 재학습 데이터로 사용한다.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    device_id  = Column(String(128), nullable=True, index=True)

    # ── 입력 피처 (모델 학습에 사용되는 원본 컬럼) ──
    subscription_id       = Column(Integer, nullable=True)
    subscription_name     = Column(String(100), nullable=True)
    emoji                 = Column(String(20), nullable=True)
    subscription_type     = Column(String(50))
    monthly_cost          = Column(Integer)
    use_frequency         = Column(String(20))
    last_use_recency      = Column(String(20))
    perceived_necessity   = Column(Integer)
    cost_burden           = Column(Integer)
    would_rebuy           = Column(Integer)
    replacement_available = Column(Integer)
    billing_cycle         = Column(Integer)
    remaining_months      = Column(Float)
    discount_amount       = Column(Integer)

    # ── 예측 결과 ──
    predicted_churn      = Column(Boolean)
    predicted_confidence = Column(Float)
    model_version        = Column(String(50))

    # ── 사용자 피드백 (나중에 채워짐) ──
    # target 규약: 1 = 유지, 0 = 해지 (mock_data_3.csv 와 동일)
    actual_target = Column(Integer, nullable=True, index=True)
    feedback_at   = Column(DateTime, nullable=True)


def _apply_lightweight_migrations() -> None:
    """SQLAlchemy 의 create_all 은 신규 컬럼을 추가하지 않는다.
    IF NOT EXISTS 가 가능한 PostgreSQL / SQLite 에서 안전하게 ALTER 한다.
    """
    with engine.begin() as conn:
        if IS_SQLITE:
            _sqlite_add_column_if_missing(
                conn, "predictions", "device_id", "VARCHAR(128)")
            _sqlite_add_column_if_missing(
                conn, "predictions", "subscription_id", "INTEGER")
            _sqlite_add_column_if_missing(
                conn, "predictions", "subscription_name", "VARCHAR(100)")
            _sqlite_add_column_if_missing(
                conn, "predictions", "emoji", "VARCHAR(20)")
            _sqlite_add_column_if_missing(
                conn, "app_subscriptions", "last_feedback_kept", "BOOLEAN")
            _sqlite_add_column_if_missing(
                conn, "app_subscriptions", "last_feedback_at", "DATETIME")
            _sqlite_add_column_if_missing(
                conn, "app_subscriptions", "billing_day", "INTEGER")
            _sqlite_add_column_if_missing(
                conn, "app_subscriptions", "next_billing_at", "DATETIME")
            _sqlite_add_column_if_missing(
                conn, "app_subscriptions", "trial_ends_at", "DATETIME")
            _sqlite_add_column_if_missing(
                conn, "app_subscriptions", "discount_ends_at", "DATETIME")
            _sqlite_add_column_if_missing(
                conn, "app_subscriptions", "renewal_notice_days", "INTEGER")
            _sqlite_add_column_if_missing(
                conn, "app_subscriptions", "archived_at", "DATETIME")
        else:
            conn.execute(text(
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS device_id VARCHAR(128)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_predictions_device_id ON predictions(device_id)"
            ))
            conn.execute(text(
                "ALTER TABLE predictions "
                "ADD COLUMN IF NOT EXISTS subscription_id INTEGER"
            ))
            conn.execute(text(
                "ALTER TABLE predictions "
                "ADD COLUMN IF NOT EXISTS subscription_name VARCHAR(100)"
            ))
            conn.execute(text(
                "ALTER TABLE predictions "
                "ADD COLUMN IF NOT EXISTS emoji VARCHAR(20)"
            ))
            conn.execute(text(
                "ALTER TABLE app_subscriptions "
                "ADD COLUMN IF NOT EXISTS last_feedback_kept BOOLEAN"
            ))
            conn.execute(text(
                "ALTER TABLE app_subscriptions "
                "ADD COLUMN IF NOT EXISTS last_feedback_at TIMESTAMP"
            ))
            conn.execute(text(
                "ALTER TABLE app_subscriptions "
                "ADD COLUMN IF NOT EXISTS billing_day INTEGER"
            ))
            conn.execute(text(
                "ALTER TABLE app_subscriptions "
                "ADD COLUMN IF NOT EXISTS next_billing_at TIMESTAMP"
            ))
            conn.execute(text(
                "ALTER TABLE app_subscriptions "
                "ADD COLUMN IF NOT EXISTS trial_ends_at TIMESTAMP"
            ))
            conn.execute(text(
                "ALTER TABLE app_subscriptions "
                "ADD COLUMN IF NOT EXISTS discount_ends_at TIMESTAMP"
            ))
            conn.execute(text(
                "ALTER TABLE app_subscriptions "
                "ADD COLUMN IF NOT EXISTS renewal_notice_days INTEGER"
            ))
            conn.execute(text(
                "ALTER TABLE app_subscriptions "
                "ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP"
            ))


def _sqlite_add_column_if_missing(conn, table: str, column: str, sqltype: str) -> None:
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    if not rows:
        return
    cols = {row[1] for row in rows}
    if column not in cols:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sqltype}"))


def init_db() -> None:
    """서버 시작 시 호출. 테이블이 없으면 생성 + 경량 마이그레이션."""
    Base.metadata.create_all(bind=engine)
    _apply_lightweight_migrations()
    print(f"[DB] Connected: {_mask_url(DATABASE_URL)}")


@contextmanager
def session_scope():
    """with 블록으로 세션을 안전하게 열고 닫는 헬퍼."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _mask_url(url: str) -> str:
    """로그용: password 부분을 가림."""
    if "@" not in url:
        return url
    scheme_and_auth, rest = url.split("@", 1)
    if "://" in scheme_and_auth and ":" in scheme_and_auth.split("://", 1)[1]:
        scheme, auth = scheme_and_auth.split("://", 1)
        user = auth.split(":", 1)[0]
        return f"{scheme}://{user}:***@{rest}"
    return url
