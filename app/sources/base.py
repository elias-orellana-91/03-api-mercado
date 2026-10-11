"""Interfaz de la fuente externa de precios.

Es la ÚNICA frontera con el exterior: solo el proceso de carga (ETL) usa una ``PriceSource``.
Los endpoints de la API leen siempre de la base de datos propia.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd


@dataclass
class MarketData:
    """Datos de mercado en formato ancho (índice = fecha)."""

    prices: pd.DataFrame  # cierre ajustado, columnas = tickers
    volume: pd.DataFrame  # volumen, columnas = tickers
    vix: pd.Series  # cierre del VIX


class PriceSource(Protocol):
    def fetch(self, tickers: list[str], start: date, end: date) -> MarketData:
        """Descarga precios, volumen y VIX para ``start <= fecha < end``."""
        ...
