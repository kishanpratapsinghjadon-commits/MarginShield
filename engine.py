from typing import Dict, List
import numpy as np

# Historical baseline cart distribution (INR) for statistical anomaly detection
HISTORICAL_CART_MEAN = 1500.0
HISTORICAL_CART_STD = 300.0
FRAUD_Z_THRESHOLD = 3.0


def calculate_dynamic_price(
    base_price: float,
    minimum_allowed_price: float,
    competitor_prices: List[float],
) -> Dict[str, float | bool]:
    """
    Determine the optimal selling price using NumPy while strictly enforcing
    the retailer's minimum margin Price Floor.
    """
    if not competitor_prices:
        return {
            "lowest_competitor_price": round(base_price, 2),
            "target_price": round(base_price, 2),
            "optimized_price": max(round(base_price, 2), round(minimum_allowed_price, 2)),
            "floor_applied": base_price < minimum_allowed_price,
        }

    # Vectorized minimum calculation across all competitor listings
    comp_array = np.asarray(competitor_prices, dtype=np.float64)
    lowest_competitor = float(np.min(comp_array))

    # Undercut the lowest competitor by 1%
    target_price = float(np.round(lowest_competitor * 0.99, 2))
    floor_price = round(float(minimum_allowed_price), 2)

    # Enforce the Price Floor so the retailer never sells below cost + margin
    optimized_price = max(target_price, floor_price)
    floor_applied = target_price < floor_price

    return {
        "lowest_competitor_price": round(lowest_competitor, 2),
        "target_price": target_price,
        "optimized_price": optimized_price,
        "floor_applied": floor_applied,
    }


def detect_fraud_zscore(
    cart_value: float,
    historical_mean: float = HISTORICAL_CART_MEAN,
    historical_std: float = HISTORICAL_CART_STD,
    threshold: float = FRAUD_Z_THRESHOLD,
) -> Dict[str, float | bool]:
    """
    Compute the statistical Z-score of an incoming order value against
    historical cart metrics using NumPy to flag high-value anomalies.
    """
    mean = np.float64(historical_mean)
    std = np.float64(historical_std)
    value = np.float64(cart_value)

    # Guard against zero standard deviation
    if std <= 0:
        z_score = 0.0
    else:
        z_score = float(np.round((value - mean) / std, 2))

    # Flag transactions exceeding the upper Z-score threshold (e.g., > 3 sigma)
    is_fraud = bool(z_score > threshold)

    return {
        "z_score": z_score,
        "is_fraud": is_fraud,
    }