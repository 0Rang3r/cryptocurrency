import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class CryptoFixedWindowDataset(Dataset):
    def __init__(self, df, seq_len, price_cols, emotion_cols, target_col):
        self.seq_len = seq_len
        self.price_cols = price_cols
        self.emotion_cols = emotion_cols
        self.target_col = target_col

        self.price_data = df[price_cols].values
        self.emotion_data = df[emotion_cols].values
        self.target_data = df[target_col].values

    def __len__(self):
        return len(self.price_data) - self.seq_len

    def __getitem__(self, idx):
        x_price = self.price_data[idx : idx + self.seq_len]
        x_emotion = self.emotion_data[idx : idx + self.seq_len]
        y = self.target_data[idx + self.seq_len - 1]

        x_price_tensor = torch.tensor(x_price, dtype=torch.float32)
        x_emotion_tensor = torch.tensor(x_emotion, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_price_tensor, x_emotion_tensor, y_tensor


def get_dataloaders(csv_path, seq_len=24, batch_size=32, target_col="target_trend_4h"):
    print("Loading data...")

    df = pd.read_csv(csv_path)
    df["future_4h_close"] = df["close"].shift(-4)
    df["target_trend_4h"] = (df["future_4h_close"] > df["close"]).astype(int)
    target_col = "target_trend_4h"
    df = df.dropna().reset_index(drop=True)

    price_cols = ["open", "high", "low", "close", "volume", "return", "Vol_t"]
    emotion_cols = ["Info_t", "FOMO_m", "FUD_m", "Euphoria_m", "Panic_m"]

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    price_scaler = StandardScaler()
    emotion_scaler = StandardScaler()

    train_df[price_cols] = price_scaler.fit_transform(train_df[price_cols])
    val_df[price_cols] = price_scaler.transform(val_df[price_cols])
    test_df[price_cols] = price_scaler.transform(test_df[price_cols])

    train_df[emotion_cols] = emotion_scaler.fit_transform(train_df[emotion_cols])
    val_df[emotion_cols] = emotion_scaler.transform(val_df[emotion_cols])
    test_df[emotion_cols] = emotion_scaler.transform(test_df[emotion_cols])

    train_dataset = CryptoFixedWindowDataset(
        train_df, seq_len, price_cols, emotion_cols, target_col
    )
    val_dataset = CryptoFixedWindowDataset(
        val_df, seq_len, price_cols, emotion_cols, target_col
    )
    test_dataset = CryptoFixedWindowDataset(
        test_df, seq_len, price_cols, emotion_cols, target_col
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Train samples: {len(train_dataset)}, "
        f"Val samples: {len(val_dataset)}, "
        f"Test samples: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, len(price_cols), len(emotion_cols)