"""Adaptador de Yahoo Finance (vía ``yfinance``). Único módulo que importa ``yfinance``."""
from __future__ import annotations

from datetime import date

import pandas as pd

from app.sources.base import MarketData


class YahooSource:
    """Implementa ``PriceSource`` con yfinance."""

    def fetch(self, tickers: list[str], start: date, end: date) -> MarketData:
        import yfinance as yf  # import local: solo el ETL lo necesita

        raw = yf.download(
            tickers, start=start.isoformat(), end=end.isoformat(), auto_adjust=True, progress=False
        )
        if raw is None or raw.empty:
            empty = pd.DataFrame(columns=tickers, dtype=float)
            return MarketData(empty, empty.copy(), pd.Series(dtype=float, name="VIX"))
        prices = raw["Close"].copy()
        volume = raw["Volume"].copy()
        for df in (prices, volume):
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"

        vix_raw = yf.download("^VIX", start=start.isoformat(), end=end.isoformat(), progress=False)
        if vix_raw is None or vix_raw.empty:
            vix = pd.Series(dtype=float, name="VIX")
        else:
            vix = vix_raw["Close"].squeeze().rename("VIX")
            vix.index = pd.to_datetime(vix.index).tz_localize(None)
        vix.index.name = "Date"
        return MarketData(prices.sort_index(), volume.sort_index(), vix.dropna().sort_index())
