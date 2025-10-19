import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class OnlineStandardizer:
  def __init__(self, n_channels):
    self.n = 0
    self.mean = np.zeros(n_channels, dtype=np.float64)
    self.M2 = np.zeros(n_channels, dtype=np.float64)

  def update(self, x):  # x: [N, C]
    x = np.asarray(x, dtype=np.float64)
    for row in x:
      self.n += 1
      delta = row - self.mean
      self.mean += delta / self.n
      delta2 = row - self.mean
      self.M2 += delta * delta2

  def finalize(self):
    var = self.M2 / max(self.n - 1, 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    return self.mean.astype(np.float32), std.astype(np.float32)

  @staticmethod
  def transform(x, mean, std):
    return (x - mean) / std


def sliding_indices(n, L, H, stride=1):
  # генерирует начальные индексы окон так, чтобы X: [L], y: [H]
  last = n - (L + H) + 1
  for s in range(0, max(last, 0), stride):
    yield s


class PatchTSTCSVIterableDataset(IterableDataset):
  def __init__(
      self,
      path,
      feature_cols=("open", "high", "low", "close", "value", "volume"),
      target_cols=("close",),
      datetime_col="begin",
      seq_len=512,
      pred_len=96,
      stride=1,
      chunksize=200_000,
      start_time=None,
      end_time=None,
      mean=None,
      std=None,
      dtype=np.float32,
      infer_target_as_delta=False,
  ):
    super().__init__()
    self.path = path
    self.feature_cols = list(feature_cols)
    self.target_cols = list(target_cols)
    self.datetime_col = datetime_col
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.stride = stride
    self.chunksize = chunksize
    self.start_time = start_time
    self.end_time = end_time
    self.mean = mean
    self.std = std
    self.dtype = dtype
    self.infer_target_as_delta = infer_target_as_delta
    self._carry = None  # переносим хвост между чанками

  def _select_time(self, df):
    if self.datetime_col in df.columns:
      if not np.issubdtype(df[self.datetime_col].dtype, np.datetime64):
        df[self.datetime_col] = pd.to_datetime(
            df[self.datetime_col], errors="coerce")
      df = df.dropna(subset=[self.datetime_col]).sort_values(self.datetime_col)
      if self.start_time is not None:
        df = df[df[self.datetime_col] >= self.start_time]
      if self.end_time is not None:
        df = df[df[self.datetime_col] < self.end_time]
    return df

  def _maybe_features(self, df):
    # TODO probably can also add overnight, intraday, range/volatility etc
    # TODO https://www.perplexity.ai/search/u-menia-open-close-high-low-va-SdOZe7dnTs6y4whYSA9Hxw
    # пример: можно добавить лог-доходности и диапазон свечи
    out = df.copy()
    if "close" in out.columns:
      out["log_close"] = np.log(out["close"].clip(lower=1e-12))
      out["ret1"] = out["log_close"].diff()
    if "high" in out.columns and "low" in out.columns:
      out["hl_range"] = (out["high"] - out["low"]) / (out["low"].abs() + 1e-6)
    feats = [c for c in self.feature_cols if c in out.columns]
    # добавим инженерные, если нужны
    for extra in ["ret1", "hl_range"]:
      if extra in out.columns and extra not in feats:
        feats.append(extra)
    return out, feats

  def _yield_windows(self, arrX, arrY):
    n = arrX.shape[0]
    L, H, st = self.seq_len, self.pred_len, self.stride
    for s in sliding_indices(n, L, H, st):
      X = arrX[s: s+L]
      Y = arrY[s+L: s+L+H]
      yield torch.from_numpy(X), torch.from_numpy(Y)

  def __iter__(self):
    self._carry = None
    reader = pd.read_csv(self.path, chunksize=self.chunksize)
    for chunk in reader:
      chunk = self._select_time(chunk)
      if chunk.empty and self._carry is None:
        continue
      if self._carry is not None:
        chunk = pd.concat([self._carry, chunk], axis=0)

      # конструируем признаки и цели
      chunk, feats = self._maybe_features(chunk)
      tcols = [c for c in self.target_cols if c in chunk.columns]
      X = chunk[feats].to_numpy(dtype=self.dtype, copy=False)
      Yraw = chunk[tcols].to_numpy(dtype=self.dtype, copy=False)

      if self.infer_target_as_delta and Yraw.shape[0] > 1:
        Y = np.diff(Yraw, axis=0, prepend=Yraw[:1])
      else:
        Y = Yraw

      # нормализация
      if self.mean is not None and self.std is not None:
        X = (X - self.mean[:X.shape[1]]) / self.std[:X.shape[1]]

      # отдаём окна
      need = self.seq_len + self.pred_len - 1
      if X.shape[0] >= (self.seq_len + self.pred_len):
        usable_rows = X.shape[0]
        # оставим хвост для следующего чанка
        self._carry = chunk.iloc[max(0, usable_rows - need):]
        for x, y in self._yield_windows(X, Y):
          yield x, y
      else:
        # недостаточно строк: накапливаем
        self._carry = chunk

    # финальный слив окон из последнего чанка
    if self._carry is not None:
      chunk = self._carry
      chunk, feats = self._maybe_features(chunk)
      tcols = [c for c in self.target_cols if c in chunk.columns]
      X = chunk[feats].to_numpy(dtype=self.dtype, copy=False)
      Y = chunk[tcols].to_numpy(dtype=self.dtype, copy=False)
      if self.mean is not None and self.std is not None:
        X = (X - self.mean[:X.shape[1]]) / self.std[:X.shape[1]]
      for x, y in self._yield_windows(X, Y):
        yield x, y
