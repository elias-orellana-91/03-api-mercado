"""Universo de tickers soportados (idéntico al del proyecto 2 / proyecto 1)."""
from __future__ import annotations

SECTORS: dict[str, list[str]] = {
    "Tecnología": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "Bancos": ["JPM", "BAC", "GS"],
    "Energía": ["XOM", "CVX"],
    "Salud": ["JNJ", "UNH", "PFE"],
    "Consumo": ["KO", "WMT"],
    "Índice": ["SPY"],
}
TICKERS: list[str] = [t for ts in SECTORS.values() for t in ts]
TICKER_TO_SECTOR: dict[str, str] = {t: s for s, ts in SECTORS.items() for t in ts}

TICKER_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc. (Clase A)",
    "AMZN": "Amazon.com, Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation",
    "GS": "The Goldman Sachs Group, Inc.",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Incorporated",
    "PFE": "Pfizer Inc.",
    "KO": "The Coca-Cola Company",
    "WMT": "Walmart Inc.",
    "SPY": "SPDR S&P 500 ETF Trust",
}

START_DATE = "2015-01-01"
