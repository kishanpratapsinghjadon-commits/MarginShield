import os
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Generator, List

from fastapi import Depends, FastAPI  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai  # Official Google GenAI SDK replacing legacy google-generativeai
from sqlalchemy.orm import Session

from engine import (
    HISTORICAL_CART_MEAN,
    HISTORICAL_CART_STD,
    calculate_dynamic_price,
    detect_fraud_zscore,
)
from models import (
    Base,
    PricingRequest,
    PricingResponse,
    ProductPricing,
    SessionLocal,
    TransactionLog,
    TransactionRequest,
    TransactionResponse,
    engine,
)


def get_db() -> Generator[Session, None, None]:
    """Provide a transactional SQLAlchemy database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_fraud_explanation(
    user_ip: str,
    cart_value: float,
    transaction_time: str,
    z_score: float,
) -> str:
    """
    Query Gemini to generate a 1-sentence plain-English explanation for the seller
    describing why the flagged transaction appears suspicious.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    fallback_msg = (
        f"Order of ₹{cart_value:,.2f} from IP {user_ip} at {transaction_time} is "
        f"{z_score:.1f} standard deviations above your ₹{HISTORICAL_CART_MEAN:,.0f} average cart value."
    )

    if not api_key:
        return fallback_msg

    prompt = (
        "You are a B2B retail fraud analyst assistant. Write strictly ONE concise, professional sentence "
        "explaining to an online store owner why this flagged checkout transaction looks suspicious and poses risk. "
        f"Context: IP={user_ip}, Cart Value=₹{cart_value:,.2f}, Time={transaction_time}, "
        f"Statistical Z-Score={z_score:.2f} (historical average=₹{HISTORICAL_CART_MEAN:,.0f}, "
        f"std dev=₹{HISTORICAL_CART_STD:,.0f})."
    )

    models_to_try = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
    try:
        client = genai.Client(api_key=api_key)
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = (response.text or "").strip()
                if text:
                    return text
            except Exception:
                continue
        return fallback_msg
    except Exception:
        return fallback_msg


def seed_initial_data(db: Session) -> None:
    """Populate initial demo product pricing and sample transactions on first boot."""
    if db.query(ProductPricing).count() == 0:
        demo_products = [
            PricingRequest(
                sku="SKU-WH-1000",
                product_name="Noise-Canceling Wireless Headphones",
                base_price=5499.0,
                minimum_allowed_price=4200.0,
                competitor_prices=[4599.0, 4399.0, 4750.0],
            ),
            PricingRequest(
                sku="SKU-KB-402",
                product_name="Mechanical RGB Gaming Keyboard",
                base_price=3299.0,
                minimum_allowed_price=2850.0,
                competitor_prices=[2800.0, 2999.0, 3150.0],  # Triggers Price Floor
            ),
        ]
        for item in demo_products:
            calc = calculate_dynamic_price(
                item.base_price, item.minimum_allowed_price, item.competitor_prices
            )
            db.add(
                ProductPricing(
                    sku=item.sku,
                    product_name=item.product_name,
                    base_price=item.base_price,
                    minimum_allowed_price=item.minimum_allowed_price,
                    competitor_prices=item.competitor_prices,
                    lowest_competitor_price=calc["lowest_competitor_price"],
                    optimized_price=calc["optimized_price"],
                    floor_applied=calc["floor_applied"],
                )
            )

    if db.query(TransactionLog).count() == 0:
        demo_z = detect_fraud_zscore(80000.0)
        db.add(
            TransactionLog(
                user_ip="185.220.101.42",
                cart_value=80000.0,
                transaction_time="2026-11-13T02:41:18Z",
                z_score=demo_z["z_score"],
                is_fraud=demo_z["is_fraud"],
                ai_explanation=(
                    "This ₹80,000.00 late-night order from 185.220.101.42 is over 261 standard "
                    "deviations above your ₹1,500 average cart size, strongly indicating potential "
                    "card-testing or account takeover fraud."
                ),
            )
        )
        normal_z = detect_fraud_zscore(1650.0)
        db.add(
            TransactionLog(
                user_ip="49.36.128.91",
                cart_value=1650.0,
                transaction_time="2026-11-13T11:15:04Z",
                z_score=normal_z["z_score"],
                is_fraud=normal_z["is_fraud"],
                ai_explanation=None,
            )
        )

    db.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_initial_data(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="MarginShield AI",
    description="Dynamic Pricing & AI Fraud Protection Platform",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the single-page Tailwind/JS seller dashboard."""
    return FileResponse("index.html")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "MarginShield AI"}


@app.get("/api/pricing", response_model=List[PricingResponse])
def list_optimized_prices(db: Session = Depends(get_db)):
    """Fetch all products and their latest NumPy-optimized prices."""
    return db.query(ProductPricing).order_by(ProductPricing.updated_at.desc()).all()


@app.post("/api/pricing/optimize", response_model=PricingResponse)
def optimize_product_price(payload: PricingRequest, db: Session = Depends(get_db)):
    """Compute and persist a competitive price protected by the retailer's Price Floor."""
    result = calculate_dynamic_price(
        base_price=payload.base_price,
        minimum_allowed_price=payload.minimum_allowed_price,
        competitor_prices=payload.competitor_prices,
    )

    record = db.query(ProductPricing).filter(ProductPricing.sku == payload.sku).first()
    now = datetime.now(timezone.utc)

    if record:
        record.product_name = payload.product_name
        record.base_price = payload.base_price
        record.minimum_allowed_price = payload.minimum_allowed_price
        record.competitor_prices = payload.competitor_prices
        record.lowest_competitor_price = result["lowest_competitor_price"]
        record.optimized_price = result["optimized_price"]
        record.floor_applied = result["floor_applied"]
        record.updated_at = now
    else:
        record = ProductPricing(
            sku=payload.sku,
            product_name=payload.product_name,
            base_price=payload.base_price,
            minimum_allowed_price=payload.minimum_allowed_price,
            competitor_prices=payload.competitor_prices,
            lowest_competitor_price=result["lowest_competitor_price"],
            optimized_price=result["optimized_price"],
            floor_applied=result["floor_applied"],
            updated_at=now,
        )
        db.add(record)

    db.commit()
    db.refresh(record)
    return record


@app.get("/api/fraud/flagged", response_model=List[TransactionResponse])
def list_flagged_transactions(db: Session = Depends(get_db)):
    """Return all transactions flagged as statistical anomalies with AI explanations."""
    return (
        db.query(TransactionLog)
        .filter(TransactionLog.is_fraud.is_(True))
        .order_by(TransactionLog.created_at.desc())
        .all()
    )


@app.get("/api/transactions", response_model=List[TransactionResponse])
def list_all_transactions(db: Session = Depends(get_db)):
    """Return recent evaluated transactions (both legitimate and flagged)."""
    return db.query(TransactionLog).order_by(TransactionLog.created_at.desc()).limit(25).all()


@app.post("/api/transactions/analyze", response_model=TransactionResponse)
def analyze_transaction(payload: TransactionRequest, db: Session = Depends(get_db)):
    """Evaluate a checkout transaction via NumPy Z-score and enrich anomalies with Gemini AI."""
    tx_time = payload.transaction_time or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    analysis = detect_fraud_zscore(payload.cart_value)

    explanation = None
    if analysis["is_fraud"]:
        explanation = generate_fraud_explanation(
            user_ip=payload.user_ip,
            cart_value=payload.cart_value,
            transaction_time=tx_time,
            z_score=analysis["z_score"],
        )

    tx_record = TransactionLog(
        user_ip=payload.user_ip,
        cart_value=payload.cart_value,
        transaction_time=tx_time,
        z_score=analysis["z_score"],
        is_fraud=analysis["is_fraud"],
        ai_explanation=explanation,
    )
    db.add(tx_record)
    db.commit()
    db.refresh(tx_record)
    return tx_record


@app.get("/api/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Return aggregated SaaS KPIs for the MarginShield executive dashboard."""
    pricing_list = db.query(ProductPricing).all()
    transactions = db.query(TransactionLog).all()

    total_products = len(pricing_list)
    floors_enforced = sum(1 for p in pricing_list if p.floor_applied)
    
    # Calculate estimated margin protected (difference when floor saved from selling at undercutting price)
    margin_protected_rupees = 0.0
    for p in pricing_list:
        if p.floor_applied:
            # Under-cutting price would have been 99% of lowest competitor
            undercut_price = round(p.lowest_competitor_price * 0.99, 2)
            margin_protected_rupees += max(0.0, p.optimized_price - undercut_price)

    total_transactions = len(transactions)
    flagged_fraud_txs = [t for t in transactions if t.is_fraud]
    fraud_count = len(flagged_fraud_txs)
    total_fraud_blocked_amount = sum(t.cart_value for t in flagged_fraud_txs)

    return {
        "total_products": total_products,
        "floors_enforced": floors_enforced,
        "floor_enforcement_rate": round((floors_enforced / total_products * 100), 1) if total_products > 0 else 0.0,
        "margin_protected_rupees": round(margin_protected_rupees, 2),
        "total_transactions": total_transactions,
        "fraud_count": fraud_count,
        "total_fraud_blocked_amount": round(total_fraud_blocked_amount, 2),
        "fraud_rate": round((fraud_count / total_transactions * 100), 1) if total_transactions > 0 else 0.0,
        "baseline_mean": HISTORICAL_CART_MEAN,
        "baseline_std": HISTORICAL_CART_STD,
        "fraud_threshold": 3.0,
    }


@app.delete("/api/pricing/{sku}")
def delete_product_pricing(sku: str, db: Session = Depends(get_db)):
    """Delete a tracked SKU from dynamic pricing."""
    item = db.query(ProductPricing).filter(ProductPricing.sku == sku).first()
    if not item:
        return {"deleted": False, "message": f"SKU {sku} not found"}
    db.delete(item)
    db.commit()
    return {"deleted": True, "sku": sku}


@app.delete("/api/transactions")
def clear_transaction_logs(db: Session = Depends(get_db)):
    """Clear transaction activity logs."""
    count = db.query(TransactionLog).delete()
    db.commit()
    return {"cleared": True, "count": count}


@app.post("/api/demo/reset")
def reset_demo_data(db: Session = Depends(get_db)):
    """Reset the database to rich default demo state for showcase testing."""
    db.query(ProductPricing).delete()
    db.query(TransactionLog).delete()
    db.commit()

    demo_products = [
        PricingRequest(
            sku="HRS-ANC-PRO",
            product_name="AeroSound ANC Pro Headphones",
            base_price=5499.0,
            minimum_allowed_price=4200.0,
            competitor_prices=[4599.0, 4399.0, 4750.0],
        ),
        PricingRequest(
            sku="HRS-KB-RGB",
            product_name="Apex Mechanical RGB Keyboard",
            base_price=3299.0,
            minimum_allowed_price=2850.0,
            competitor_prices=[2800.0, 2999.0, 3150.0],  # Triggers Floor
        ),
        PricingRequest(
            sku="HRS-TSHIRT-01",
            product_name="Signature Organic Cotton Tee",
            base_price=899.0,
            minimum_allowed_price=650.0,
            competitor_prices=[850.0, 799.0, 620.0],  # Lowest is 620 -> 1% cut is 613.8 -> Triggers 650 floor
        ),
        PricingRequest(
            sku="HRS-WATCH-GT",
            product_name="Vanguard AMOLED Smartwatch",
            base_price=12999.0,
            minimum_allowed_price=9800.0,
            competitor_prices=[11499.0, 11800.0, 12200.0],
        ),
    ]

    for item in demo_products:
        calc = calculate_dynamic_price(
            item.base_price, item.minimum_allowed_price, item.competitor_prices
        )
        db.add(
            ProductPricing(
                sku=item.sku,
                product_name=item.product_name,
                base_price=item.base_price,
                minimum_allowed_price=item.minimum_allowed_price,
                competitor_prices=item.competitor_prices,
                lowest_competitor_price=calc["lowest_competitor_price"],
                optimized_price=calc["optimized_price"],
                floor_applied=calc["floor_applied"],
            )
        )

    # Seed transactions
    demo_z1 = detect_fraud_zscore(85000.0)
    db.add(
        TransactionLog(
            user_ip="185.220.101.42",
            cart_value=85000.0,
            transaction_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            z_score=demo_z1["z_score"],
            is_fraud=demo_z1["is_fraud"],
            ai_explanation="Order of ₹85,000.00 from known Tor Exit Node is 278.3σ above historical mean (₹1,500), indicating card testing velocity fraud.",
        )
    )

    demo_z2 = detect_fraud_zscore(1650.0)
    db.add(
        TransactionLog(
            user_ip="49.36.128.91",
            cart_value=1650.0,
            transaction_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            z_score=demo_z2["z_score"],
            is_fraud=demo_z2["is_fraud"],
            ai_explanation=None,
        )
    )

    demo_z3 = detect_fraud_zscore(18200.0)
    db.add(
        TransactionLog(
            user_ip="103.241.202.15",
            cart_value=18200.0,
            transaction_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            z_score=demo_z3["z_score"],
            is_fraud=demo_z3["is_fraud"],
            ai_explanation="High-value cart anomaly detected: cart value is 55.7 standard deviations above average retail checkout norm.",
        )
    )

    demo_z4 = detect_fraud_zscore(1280.0)
    db.add(
        TransactionLog(
            user_ip="152.58.14.88",
            cart_value=1280.0,
            transaction_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            z_score=demo_z4["z_score"],
            is_fraud=demo_z4["is_fraud"],
            ai_explanation=None,
        )
    )

    db.commit()
    return {"status": "ok", "message": "Demo data restored"}