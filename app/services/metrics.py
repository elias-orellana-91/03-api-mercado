"""Métricas financieras (funciones puras). Lógica reutilizada del proyecto 1 (``src/data.py``)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    total_return: float
    volatility: float
    max_drawdown: float
    sharpe: float | None
    n_days: int


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change(fill_method=None).iloc[1:]


def total_return(prices: pd.Series) -> float:
    return float(prices.iloc[-1] / prices.iloc[0] - 1)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(prices: pd.Series) -> float:
    return float((prices / prices.cummax() - 1).min())


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float | None:
    """Sharpe anualizado; ``None`` si la volatilidad es nula o no definida."""
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    if not np.isfinite(vol) or vol <= 1e-12:
        return None
    return float((returns.mean() * TRADING_DAYS - risk_free) / vol)


def compute_metrics(prices: pd.Series) -> Metrics:
    """Métricas de una serie de precios con al menos 2 observaciones."""
    prices = prices.dropna()
    rets = daily_returns(prices)
    return Metrics(
        total_return=total_return(prices),
        volatility=annualized_volatility(rets),
        max_drawdown=max_drawdown(prices),
        sharpe=sharpe_ratio(rets),
        n_days=int(len(prices)),
    )
