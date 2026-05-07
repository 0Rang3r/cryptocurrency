import argparse
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CRYPTO_ROOT = os.path.join(PROJECT_ROOT, "crypto_predictor")
if CRYPTO_ROOT not in sys.path:
    sys.path.insert(0, CRYPTO_ROOT)

CSV_PATH = "btc_multimodal_hourly_dataset.csv"
SEQ_LEN = 24
BATCH_SIZE = 32
TARGET_COL = "target_trend_4h"
EMOTION_LAG = 1

V2_OUT_PATH = os.path.join("dataset_cache", "adaptive_dataset_v2.pt")
M5_OUT_PATH = os.path.join("dataset_cache", "adaptive_dataset_m5plus.pt")


def build_v2():
    from dataset_adaptive import get_adaptive_dataloaders

    print("Building adaptive v2 cache...")

    train_loader, val_loader, test_loader, p_dim, e_dim = get_adaptive_dataloaders(
        csv_path=CSV_PATH,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        target_col=TARGET_COL,
        emotion_lag=EMOTION_LAG,
    )

    payload = {
        "train": train_loader.dataset,
        "val": val_loader.dataset,
        "test": test_loader.dataset,
        "p_dim": p_dim,
        "e_dim": e_dim,
        "seq_len": SEQ_LEN,
        "target_col": TARGET_COL,
        "task_horizon": 4,
        "cache_version": "adaptive_v2_mean_last_ohlcaware",
        "emotion_lag": EMOTION_LAG,
        "note": "Adaptive v2 cache.",
    }

    torch.save(payload, V2_OUT_PATH)
    print(f"Saved: {V2_OUT_PATH}")
    print(f"price_dim={p_dim}, emotion_dim={e_dim}")


def build_m5plus():
    from dataset_adaptive_m5plus import get_adaptive_dataloaders

    print("Building adaptive M5-plus cache...")

    train_loader, val_loader, test_loader, p_dim, e_dim = get_adaptive_dataloaders(
        csv_path=CSV_PATH,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        target_col=TARGET_COL,
        emotion_lag=EMOTION_LAG,
    )

    payload = {
        "train": train_loader.dataset,
        "val": val_loader.dataset,
        "test": test_loader.dataset,
        "p_dim": p_dim,
        "e_dim": e_dim,
        "seq_len": SEQ_LEN,
        "target_col": TARGET_COL,
        "task_horizon": 4,
        "cache_version": "adaptive_m5plus_mean_last_delta_zscore_state",
        "emotion_lag": EMOTION_LAG,
        "note": "Adaptive M5-plus cache.",
    }

    torch.save(payload, M5_OUT_PATH)
    print(f"Saved: {M5_OUT_PATH}")
    print(f"price_dim={p_dim}, emotion_dim={e_dim}")


def main():
    parser = argparse.ArgumentParser(description="Build adaptive caches")
    parser.add_argument(
        "--mode",
        choices=["both", "v2", "m5plus"],
        default="both",
        help="both: build both caches; v2: only adaptive_dataset_v2.pt; m5plus: only adaptive_dataset_m5plus.pt",
    )
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Cannot find source CSV: {CSV_PATH}")

    os.makedirs("dataset_cache", exist_ok=True)

    if args.mode in ("both", "v2"):
        build_v2()
        print()

    if args.mode in ("both", "m5plus"):
        build_m5plus()
        print()

    print("Done")


if __name__ == "__main__":
    main()