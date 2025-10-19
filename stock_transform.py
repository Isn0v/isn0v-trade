import pandas as pd
import numpy as np


def resample_ohlcv(df: pd.DataFrame, interval):
  agg_dict = {
      'open': 'first',
      'high': 'max',
      'low': 'min',
      'close': 'last',
      'volume': 'sum',
      'value': 'sum',
      'mask_original': 'all'
  }

  df_resampled = df.resample(interval).agg(agg_dict)  # type: ignore
  df_resampled.dropna(how='all', inplace=True)

  return df_resampled.reset_index()


def fill_missing_values(df: pd.DataFrame, freq='1min'):
  df = df.sort_values('timestamp').set_index(pd.to_datetime(df['timestamp']))
  full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
  orig_idx = df.index
  out = df.reindex(full_idx)

  out['mask_original'] = out.index.isin(orig_idx)
  out.drop(['timestamp'], inplace=True, axis=1)

  price_cols = [c for c in ["open", "high", "low", "close"] if c in out]
  vol_cols = [c for c in ["volume", "value"] if c in out]

  if price_cols:
    out[price_cols] = out[price_cols].interpolate(
        method="time", limit_direction="both")
    if {"high", "low"}.issubset(out.columns):
      o = out["open"] if "open" in out else out["close"]
      c = out["close"] if "close" in out else out["open"]
      out["high"] = np.maximum.reduce([out["high"], o, c])
      out["low"] = np.minimum.reduce([out["low"],  o, c])

  if vol_cols:
    out[vol_cols] = out[vol_cols].fillna(0.0)  # или interpolate(method="time")

  out[price_cols] = out[price_cols].ffill().bfill()
  return out


def add_features(df: pd.DataFrame):
  df = df.copy()
  # базовое приведение времени
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  eps = 1e-12
  # лог-цены и доходности
  for c in ['open', 'high', 'low', 'close']:
    df[f'log_{c}'] = np.log(df[c].clip(lower=eps))
  df['ret1'] = df['log_close'].diff().fillna(0.0)
  df['ret_open'] = df['log_open'].diff().fillna(0.0)
  df['ret_high'] = df['log_high'].diff().fillna(0.0)
  df['ret_low'] = df['log_low'].diff().fillna(0.0)
  df['ret_co'] = df['log_close'] - df['log_open']
  df['ret_hl'] = df['log_high'] - df['log_low']

  # геометрия свечи
  rng = (df['high'] - df['low']).clip(lower=0.0)
  df['range'] = rng

  df['body'] = df['close'] - df['open']

  df['upper_wick'] = np.maximum(
      0.0, df['high'] - df[['open', 'close']].max(axis=1))
  df['lower_wick'] = np.maximum(
      0.0, df[['open', 'close']].min(axis=1) - df['low'])

  df['body_pct'] = np.where(rng > 0, df['body']/rng, 0.0)
  df['upper_wick_pct'] = np.where(rng > 0, df['upper_wick'] / rng, 0.0)
  df['lower_wick_pct'] = np.where(rng > 0, df['lower_wick'] / rng, 0.0)

  df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0

  # волатильностные оценки per-bar
  log_hl = np.log(df['high'].clip(lower=eps)) - \
      np.log(df['low'].clip(lower=eps))
  log_co = np.log(df['close'].clip(lower=eps)) - \
      np.log(df['open'].clip(lower=eps))
  df['parkinson_var'] = (log_hl**2) / (4*np.log(2.0))
  df['gk_var'] = 0.5*(log_hl**2) - (2*np.log(2.0)-1)*(log_co**2)
  df['rs_var'] = (np.log(df['high'].clip(lower=eps)/df['close'].clip(lower=eps)) *
                  np.log(df['high'].clip(lower=eps)/df['open'].clip(lower=eps)) +
                  np.log(df['low'].clip(lower=eps)/df['close'].clip(lower=eps)) *
                  np.log(df['low'].clip(lower=eps)/df['open'].clip(lower=eps)))

  for col in ['parkinson_var', 'gk_var', 'rs_var']:
    df[col] = df[col].clip(lower=0)

  # rolling волатильность (напр., N=20 баров)
  N = 20
  df['rv_logret'] = df['ret1']**2
  df['rv_N'] = df['rv_logret'].rolling(N).sum().pow(0.5)
  df['parkinson_vol_N'] = df['parkinson_var'].rolling(N).mean().pow(0.5)
  df['gk_vol_N'] = df['gk_var'].rolling(N).mean().pow(0.5)
  df['rs_vol_N'] = df['rs_var'].rolling(N).mean().pow(0.5)

  # ATR (Wilder 14)
  tr1 = (df['high'] - df['low']).abs()
  tr2 = (df['high'] - df['close'].shift()).abs()
  tr3 = (df['low'] - df['close'].shift()).abs()
  df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
  df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()

  # Полосы Боллинджера
  n_bb, k = 20, 2.0
  ma = df['close'].rolling(n_bb).mean()
  sd = df['close'].rolling(n_bb).std(ddof=0)
  df['bb_mid'] = ma
  df['bb_up'] = ma + k*sd
  df['bb_dn'] = ma - k*sd
  df['bb_width'] = (df['bb_up'] - df['bb_dn']) / ma
  df['bb_percB'] = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
  # # TODO
  # # чтобы сделать работу модели более стабильной, можно ограничить значение этого показателя,
  # # но тогда модель потеряет информацию о величине пробоя
  # df['bb_percB'].clip(0, 1, inplace=True)
  df.replace([np.inf, -np.inf], np.nan, inplace=True)

  # MACD
  ema_fast = df['close'].ewm(span=12, adjust=False).mean()
  ema_slow = df['close'].ewm(span=26, adjust=False).mean()
  macd = ema_fast - ema_slow
  signal = macd.ewm(span=9, adjust=False).mean()
  df['macd'] = macd
  df['macd_signal'] = signal
  df['macd_hist'] = macd - signal

  # RSI (Wilder 14)
  delta = df['close'].diff()
  gain = delta.clip(lower=0)
  loss = -delta.clip(upper=0)
  roll_up = gain.ewm(alpha=1/14, adjust=False).mean()
  roll_down = loss.ewm(alpha=1/14, adjust=False).mean()
  rs = roll_up / (roll_down + 1e-12)
  df['rsi14'] = 100 - (100 / (1 + rs))

  # VWAP — сброс по дням
  date_key = df['timestamp'].dt.tz_convert(None).dt.date if hasattr(
      df['timestamp'].dt, 'tz') and df['timestamp'].dt.tz is not None else df['timestamp'].dt.date
  tp = df['typical_price']
  df['vwap'] = (tp*df['volume']).groupby(date_key).cumsum() / \
      (df['volume'].groupby(date_key).cumsum().replace(0, np.nan))

  # После кода расчета VWAP
  df['vwap'] = df.groupby(date_key)['vwap'].transform(lambda x: x.ffill())
  # Может остаться NaN в самом начале, если первый день начинается без объема
  df['vwap'] = df['vwap'].bfill()  # Заполняем оставшиеся NaN

  # OBV
  direction = np.sign(df['close'].diff().fillna(0.0))
  df['obv'] = (direction * df['volume']).cumsum()

  # MFI (14)
  n_mfi = 14
  mf = tp * df['volume']
  pos_mf = mf.where(tp > tp.shift(), 0.0)
  neg_mf = mf.where(tp < tp.shift(), 0.0)
  pmf = pos_mf.rolling(n_mfi).sum()
  nmf = neg_mf.rolling(n_mfi).sum()
  mr = pmf / (nmf + 1e-12)
  df['mfi14'] = 100 - 100/(1 + mr)

  # календарные признаки и циклические кодировки
  dt = pd.to_datetime(df['timestamp'])
  dow = dt.dt.dayofweek
  hour = dt.dt.hour
  # df['minute'] = dt.dt.minute
  df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
  df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
  df['hour_sin'] = np.sin(2*np.pi*hour/24)
  df['hour_cos'] = np.cos(2*np.pi*hour/24)
  # df['min_sin'] = np.sin(2*np.pi*df['minute']/60)
  # df['min_cos'] = np.cos(2*np.pi*df['minute']/60)

  df.dropna(inplace=True)

  return df
