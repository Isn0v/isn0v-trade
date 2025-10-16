import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# Загружаем данные
df = pd.read_csv('SBER.csv')
data = df[['open','close','high','low','value','volume']].values

# Масштабируем
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Формируем последовательности (пример: 128 прошлых баров для прогноза 16)
seq_len = 128
pred_len = 16

X, y = [], []
for i in range(len(data_scaled) - seq_len - pred_len):
    X.append(data_scaled[i:i+seq_len])
    y.append(data_scaled[i+seq_len:i+seq_len+pred_len, :4]) # прогноз open, close, high, low

X = np.array(X)
y = np.array(y)

# В DataLoader
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
