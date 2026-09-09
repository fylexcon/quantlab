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

ATR tabanli kaymayi gercekci maliyet modeline eklemek icin ``high`` ve ``low``
serilerini de verin. Kayma, her pozisyon degisiminde ``ATR / close`` ile
orantilidir:

```python
result = run_long_only_backtest(
    prices["close"],
    signals["position"],
    high=prices["high"],
    low=prices["low"],
    slippage_atr_multiplier=0.10,
)
```

## Walk-Forward Dogrulama

``run_sma_rsi_walk_forward`` aday parametreleri sadece egitim penceresinde
degerlendirir, ardindan sonraki out-of-sample pencerede test eder. Gunluk veri
icin 6 ay/2 ay yaklasimi yaklasik ``126`` ve ``42`` gozleme karsilik gelir.

```python
from backtest.walk_forward import WalkForwardConfig, run_sma_rsi_walk_forward
from strategies.trend_following import SmaRsiConfig

result = run_sma_rsi_walk_forward(
    prices["close"],
    [SmaRsiConfig(10, 30), SmaRsiConfig(20, 50)],
    WalkForwardConfig(train_periods=126, test_periods=42),
)
print(result.selections)
```

## QAOA Benchmark

QAOA katman sayisi ile COBYLA/SPSA optimize edicilerini ayni problemde
karsilastirmak icin:

```bash
python scripts/benchmark_qaoa.py --reps 1 2 3 --optimizers COBYLA SPSA --maxiter 50
```

Not defterlerini baslatmak icin:

```bash
jupyter lab
```

## Not Defterleri

- `notebooks/01_sma_rsi_backtest.ipynb`: Yahoo Finance verisiyle SMA-RSI stratejisini calistirir ve sermaye egrisini cizer.
- `notebooks/02_qaoa_portfolio_optimization.ipynb`: Ornek getiri ve kovaryans matrisiyle QAOA varlik secimini gosterir.

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
