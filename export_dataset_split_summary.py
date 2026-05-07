import pandas as pd


CSV_PATH = "btc_multimodal_hourly_dataset.csv"
OUT_PATH = "dataset_split_summary.csv"
TARGET_COL = "target_trend_4h"
TARGET_HORIZON = 4
SEQ_LEN = 24

PRICE_COLS = ["open", "high", "low", "close", "volume", "return", "Vol_t"]
EMOTION_COLS = ["Info_t", "FOMO_m", "FUD_m", "Euphoria_m", "Panic_m"]


def get_time_col(df):
    for col in ["date_hour", "timestamp", "datetime", "date"]:
        if col in df.columns:
            return col
    return None


def ensure_target(df):
    df = df.copy()
    if TARGET_COL not in df.columns:
        df["future_4h_close"] = df["close"].shift(-TARGET_HORIZON)
        df[TARGET_COL] = (df["future_4h_close"] > df["close"]).astype(int)
    return df


def describe_split(name, split_df, time_col):
    if len(split_df) == 0:
        start_time = ""
        end_time = ""
    elif time_col is not None:
        start_time = str(split_df[time_col].iloc[0])
        end_time = str(split_df[time_col].iloc[-1])
    else:
        start_time = int(split_df.index.min())
        end_time = int(split_df.index.max())

    positive_rate = float(split_df[TARGET_COL].mean()) if len(split_df) else 0.0

    return {
        "split": name,
        "n_rows_before_window": int(len(split_df)),
        "start_time": start_time,
        "end_time": end_time,
        "target_col": TARGET_COL,
        "positive_class_rate": positive_rate,
        "negative_class_rate": 1.0 - positive_rate,
        "seq_len": SEQ_LEN,
        "target_horizon_hours": TARGET_HORIZON,
        "split_method": "chronological_70_15_15",
        "normalization": "scalers_fit_on_training_set_only",
    }


def main():
    df = pd.read_csv(CSV_PATH).reset_index(drop=True)
    df = ensure_target(df)

    required_cols = PRICE_COLS + EMOTION_COLS + [TARGET_COL]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    time_col = get_time_col(df)
    rows = [
        describe_split("train", df.iloc[:train_end].copy(), time_col),
        describe_split("validation", df.iloc[train_end:val_end].copy(), time_col),
        describe_split("test", df.iloc[val_end:].copy(), time_col),
    ]

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"保存: {OUT_PATH}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
