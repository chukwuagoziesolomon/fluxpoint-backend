from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np


# =========================
# DATA TYPES
# =========================

@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    timestamp: Any


@dataclass
class Indicators:
    ma6: float
    ma18: float
    ma50: float
    ma200: float
    slope6: float
    slope18: float
    slope50: float
    slope200: float
    atr: float  # Average True Range for stop loss calculation


@dataclass
class Swing:
    type: str  # "high" or "low"
    price: float
    fib_level: Optional[float] = None  # 0.382, 0.5, 0.618
    fib_618_price: Optional[float] = None  # 61.8% Fibonacci price level


@dataclass
class HigherTFCandle:
    indicators: Indicators
    high: float
    low: float


@dataclass
class MarketStructure:
    highs: List[float]
    lows: List[float]