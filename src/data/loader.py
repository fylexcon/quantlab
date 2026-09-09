"""Yahoo Finance veri indirme ve OHLCV standardizasyonu."""

from __future__ import annotations

from typing import Any

import pandas as pd


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class MarketDataError(RuntimeError):
    """Piyasa verisi indirilemediginde veya gecersiz oldugunda yukseltilir."""


def download_ohlcv(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Tek bir sembol icin Yahoo Finance'tan temiz OHLCV verisi indirir.

    Sonuc kucuk harfli ``open``, ``high``, ``low``, ``close`` ve ``volume``
    sutunlarina sahiptir. Ag baglantisi veya veri saglayici hatalari, acik bir
    ``MarketDataError`` olarak yuzeye cikar.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    if not interval:
        raise ValueError("interval must be a non-empty string")

    try:
        import yfinance as yf
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise MarketDataError("yfinance is required; install requirements.txt") from error

    try:
        raw = yf.download(
            ticker.strip().upper(),
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            multi_level_index=False,
        )
    except Exception as error:  # pragma: no cover - provider boundary
        raise MarketDataError(f"Unable to download data for {ticker!r}") from error

    if raw.empty:
        raise MarketDataError(f"No data returned for {ticker!r}")
    return normalize_ohlcv(raw)


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Saglayici kaynakli bir tabloyu tek sembollu OHLCV semasina donusturur."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise MarketDataError("Cannot normalize an empty market-data frame")

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized = _flatten_single_ticker_columns(normalized)

    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
    missing = set(OHLCV_COLUMNS).difference(normalized.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise MarketDataError(f"OHLCV data is missing required columns: {missing_columns}")
    if normalized.columns.duplicated().any():
        raise MarketDataError("Expected data for one ticker, but found duplicate price columns")

    normalized = normalized.loc[:, OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    normalized = normalized.dropna(subset=["close"]).sort_index()
    normalized = normalized.loc[~normalized.index.duplicated(keep="last")]
    if isinstance(normalized.index, pd.DatetimeIndex) and normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    if normalized.empty:
        raise MarketDataError("No valid close prices remain after normalization")
    return normalized


def _flatten_single_ticker_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """YFinance'in tek sembol icin dahi olusturabildigi MultiIndex'i acar."""
    for level in range(frame.columns.nlevels):
        labels = [str(value).strip().lower().replace(" ", "_") for value in frame.columns.get_level_values(level)]
        if "close" in labels:
            flattened = frame.copy()
            flattened.columns = labels
            return flattened
    raise MarketDataError("Unable to identify price columns in MultiIndex data")


def as_price_series(frame: pd.DataFrame, column: str = "close") -> pd.Series[Any]:
    """Normalizasyon sonrasi fiyat sutununa erisim icin kisa yardimci."""
    if column not in frame.columns:
        raise KeyError(f"Price column {column!r} is not present")
    return frame[column].astype(float).rename(column)
