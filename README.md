# Quant Quantum Lab

`quant-quantum-lab`, teknik analiz tabanli trend stratejilerini vektorize bicimde test etmek ve kucuk portfoy secim problemlerini QAOA ile incelemek icin tasarlanmis bir Python arastirma deposudur.

> Bu proje egitim ve arastirma amaclidir; yatirim tavsiyesi degildir.

## Kapsam

- Yahoo Finance uzerinden OHLCV veri indirme ve standartlastirma
- SMA, RSI, MACD ve ATR gostergeleri
- Gecikmeli (look-ahead bias olmadan) SMA-RSI trend sinyalleri
- Vektorize getiri, maliyet, Sharpe ve maksimum dusus hesaplama
- Qiskit ile sabit sayida varlik secen QAOA portfoy optimizasyonu

## Kurulum

Python 3.10 veya ustu kullanin.

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Hizli Baslangic

```python
from data.loader import download_ohlcv
from strategies.trend_following import sma_rsi_signals
from backtest.engine import run_long_only_backtest

prices = download_ohlcv("AAPL", start="2022-01-01", end="2024-01-01")
signals = sma_rsi_signals(prices["close"])
result = run_long_only_backtest(prices["close"], signals["position"])

print(result.metrics)
```

Not defterlerini baslatmak icin:

```bash
jupyter lab
```

## Dizin Yapisi

```text
src/
  data/          Veri indirme ve on isleme
  indicators/    Teknik analiz fonksiyonlari
  strategies/    Sinyal kurallari
  backtest/      Vektorize performans motoru
  quantum/       QAOA portfoy modeli ve cozucu
notebooks/       Tekrar calistirilabilir arastirma calismalari
tests/           Deterministik birim testleri
```

## Gelistirme Akisi

Her degisiklik konuya ozel bir dalda gelistirilir; commitler tek bir davranissal amaca odaklanir. Birim testleri calistirmak icin `python -m pytest` kullanin. Varsayilan dal `main`, baslangic uygulamasi ise `codex/bootstrap-architecture` dalinda gelistirilmistir.
