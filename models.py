import os
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Ensure SQLite data directory exists for local and Docker volume persistence
os.makedirs("data", exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/secureprice.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
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