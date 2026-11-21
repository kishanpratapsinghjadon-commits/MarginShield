import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Database Configuration (Vercel Serverless & Local Compatible)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Detect Vercel / AWS Lambda serverless read-only filesystem
is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

DATABASE_URL = os.getenv("DATABASE_URL")

if is_serverless:
    # Vercel /var/task is strictly read-only. Only /tmp is writable.
    tmp_data_dir = os.path.join(tempfile.gettempdir(), "data")
    try:
        os.makedirs(tmp_data_dir, exist_ok=True)
    except Exception:
        pass

    tmp_db_path = os.path.join(tmp_data_dir, "secureprice.db").replace("\\", "/")

    # If seed database exists in read-only /var/task/data, copy it to writable /tmp/data
    source_seed_db = os.path.join(BASE_DIR, "data", "secureprice.db")
    if not os.path.exists(tmp_db_path) and os.path.exists(source_seed_db):
        try:
            shutil.copy2(source_seed_db, tmp_db_path)
        except Exception as copy_err:
            print(f"Notice: Could not copy seed db to /tmp/data: {copy_err}")

    # If no remote DB is provided (or if default sqlite ./data is specified), use /tmp/data
    if not DATABASE_URL or ("sqlite:///" in DATABASE_URL and ("./data" in DATABASE_URL or "/var/task" in DATABASE_URL)):
        DATABASE_URL = f"sqlite:///{tmp_db_path}"
else:
    # Local development or Docker environment
    if not DATABASE_URL:
        local_data_dir = os.path.join(BASE_DIR, "data")
        try:
            os.makedirs(local_data_dir, exist_ok=True)
            local_db_path = os.path.join(local_data_dir, "secureprice.db").replace("\\", "/")
            DATABASE_URL = f"sqlite:///{local_db_path}"
        except (OSError, PermissionError):
            tmp_data_dir = os.path.join(tempfile.gettempdir(), "data")
            os.makedirs(tmp_data_dir, exist_ok=True)
            tmp_db_path = os.path.join(tmp_data_dir, "secureprice.db").replace("\\", "/")
            DATABASE_URL = f"sqlite:///{tmp_db_path}"

try:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    )
except Exception:
    # Failsafe in-memory database to prevent import-time crashes
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# SQLAlchemy ORM Models
# ---------------------------------------------------------------------------

class ProductPricing(Base):
    """Stores product pricing parameters and the latest NumPy-optimized price."""
    __tablename__ = "product_pricing"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(64), unique=True, index=True, nullable=False)
    product_name = Column(String(128), nullable=False)
    base_price = Column(Float, nullable=False)
    minimum_allowed_price = Column(Float, nullable=False)  # Cost + minimum margin floor
    competitor_prices = Column(JSON, nullable=False)
    lowest_competitor_price = Column(Float, nullable=False)
    optimized_price = Column(Float, nullable=False)
    floor_applied = Column(Boolean, default=False, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class TransactionLog(Base):
    """Stores analyzed checkout transactions and AI explanations for anomalies."""
    __tablename__ = "transaction_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_ip = Column(String(64), nullable=False, index=True)
    cart_value = Column(Float, nullable=False)
    transaction_time = Column(String(64), nullable=False)
    z_score = Column(Float, nullable=False)
    is_fraud = Column(Boolean, default=False, nullable=False, index=True)
    ai_explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


# ---------------------------------------------------------------------------
# Pydantic Validation Schemas
# ---------------------------------------------------------------------------

class PricingRequest(BaseModel):
    sku: str = Field(..., min_length=1, max_length=64, examples=["SKU-ELEC-101"])
    product_name: str = Field(..., min_length=1, max_length=128, examples=["Wireless ANC Headphones"])
    base_price: float = Field(..., gt=0, examples=[4999.0])
    minimum_allowed_price: float = Field(..., gt=0, examples=[4200.0])
    competitor_prices: List[float] = Field(..., min_length=1, examples=[[4500.0, 4350.0, 4800.0]])


class PricingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sku: str
    product_name: str
    base_price: float
    minimum_allowed_price: float
    competitor_prices: List[float]
    lowest_competitor_price: float
    optimized_price: float
    floor_applied: bool
    updated_at: datetime


class TransactionRequest(BaseModel):
    user_ip: str = Field(..., min_length=7, max_length=64, examples=["192.168.1.45"])
    cart_value: float = Field(..., gt=0, examples=[80000.0])
    transaction_time: Optional[str] = Field(default=None, examples=["2026-11-13T03:14:00Z"])


class TransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_ip: str
    cart_value: float
    transaction_time: str
    z_score: float
    is_fraud: bool
    ai_explanation: Optional[str] = None
    created_at: datetime