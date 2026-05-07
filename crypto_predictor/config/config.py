from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    # 数据集划分
    csv_path: str = "btc_multimodal_hourly_dataset.csv"
    seq_len: int = 24
    batch_size: int = 32
    target_col: str = "target_trend"
    train_ratio: float = 0.70
    val_ratio: float = 0.15       

    # 价格特征列
    price_cols: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume', 'return', 'Vol_t'
    ])
    # 情绪特征列
    emotion_cols: List[str] = field(default_factory=lambda: [
        'Info_t', 'FOMO_m', 'FUD_m', 'Euphoria_m', 'Panic_m'
    ])

    # 自适应窗口
    vol_rolling_window: int = 720


@dataclass
class ModelConfig:
    d_model: int = 64            
    num_heads: int = 4               
    num_layers: int = 2                
    hidden_dim: int = 64                 
    lstm_layers: int = 2
    dropout: float = 0.2
    output_dim: int = 1

    # 情绪调制系数
    emotion_alpha: float = 0.5
    learnable_alpha: bool = True


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 2e-4                     
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    early_stopping_metric: str = "f1"
    grad_clip_norm: float = 1.0
    pos_weight: Optional[float] = None

    # 学习率调度器
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"
    lr_min: float = 1e-6
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    seed: int = 42


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # 消融实验
    run_baseline_lstm: bool = True
    run_baseline_gru: bool = True
    run_vanilla_transformer: bool = True
    run_cross_attn_no_modulation: bool = True
    run_innovative_full: bool = True
    use_adaptive_window: bool = True

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
DEFAULT_CONFIG = ExperimentConfig()
