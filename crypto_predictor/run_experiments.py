import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import ExperimentConfig, ModelConfig
from utils.seed import set_seed
from utils.logger import get_logger
from data.dataset import get_dataloaders
from data.adaptive_dataset import get_adaptive_dataloaders
from models.baselines.lstm_gru_transformer import (
    BaselineLSTM,
    MultimodalGRU,
    VanillaTransformer,
)
from models.innovative.dual_attention_model import InnovativeDualAttentionModel
from models.innovative.ablation_models import (
    CrossAttnNoModulation,
    AdditiveModulationModel,
)
from training.trainer import Trainer
from evaluation.reporter import ExperimentReporter

logger = get_logger(__name__)


def run_all_experiments(cfg: ExperimentConfig = None) -> None:
    if cfg is None:
        cfg = ExperimentConfig()

    set_seed(cfg.train.seed)
    device = cfg.device

    logger.info(f"运行设备: {device}")
    logger.info(f"数据集路径: {cfg.data.csv_path}")

    reporter = ExperimentReporter(results_dir=cfg.train.results_dir)

    logger.info("\n加载固定窗口数据集")
    train_loader, val_loader, test_loader, price_dim, emotion_dim = get_dataloaders(
        cfg.data.csv_path,
        cfg=cfg.data,
    )

    adp_train, adp_val, adp_test = None, None, None
    adp_price_dim, adp_emotion_dim = None, None

    if cfg.use_adaptive_window and cfg.run_innovative_full:
        logger.info("\n加载自适应窗口数据集")
        adp_train, adp_val, adp_test, adp_price_dim, adp_emotion_dim = (
            get_adaptive_dataloaders(
                cfg.data.csv_path,
                cfg=cfg.data,
            )
        )

    model_cfg = cfg.model

    if cfg.run_baseline_lstm:
        model = BaselineLSTM(
            price_dim=price_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.lstm_layers,
            dropout=model_cfg.dropout,
        )
        trainer = Trainer(
            model,
            "A_BaselineLSTM",
            train_loader,
            val_loader,
            test_loader,
            device,
            cfg=cfg.train,
        )
        reporter.add(trainer.run())

    if cfg.run_baseline_gru:
        model = MultimodalGRU(
            price_dim=price_dim,
            emotion_dim=emotion_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.lstm_layers,
            dropout=model_cfg.dropout,
        )
        trainer = Trainer(
            model,
            "B_MultimodalGRU",
            train_loader,
            val_loader,
            test_loader,
            device,
            cfg=cfg.train,
        )
        reporter.add(trainer.run())

    if cfg.run_vanilla_transformer:
        model = VanillaTransformer(
            price_dim=price_dim,
            emotion_dim=emotion_dim,
            d_model=model_cfg.d_model,
            nhead=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
        )
        trainer = Trainer(
            model,
            "C_VanillaTransformer",
            train_loader,
            val_loader,
            test_loader,
            device,
            cfg=cfg.train,
        )
        reporter.add(trainer.run())

    if cfg.run_cross_attn_no_modulation:
        model = CrossAttnNoModulation(
            price_dim=price_dim,
            emotion_dim=emotion_dim,
            cfg=model_cfg,
        )
        trainer = Trainer(
            model,
            "D_CrossAttn_NoMod",
            train_loader,
            val_loader,
            test_loader,
            device,
            cfg=cfg.train,
        )
        reporter.add(trainer.run())

    model = AdditiveModulationModel(
        price_dim=price_dim,
        emotion_dim=emotion_dim,
        cfg=model_cfg,
    )
    trainer = Trainer(
        model,
        "E_AdditiveModulation",
        train_loader,
        val_loader,
        test_loader,
        device,
        cfg=cfg.train,
    )
    reporter.add(trainer.run())

    if cfg.run_innovative_full:
        use_adaptive = cfg.use_adaptive_window and adp_train is not None

        current_price_dim = adp_price_dim if use_adaptive else price_dim
        current_emotion_dim = adp_emotion_dim if use_adaptive else emotion_dim
        current_train = adp_train if use_adaptive else train_loader
        current_val = adp_val if use_adaptive else val_loader
        current_test = adp_test if use_adaptive else test_loader

        model = InnovativeDualAttentionModel(
            price_dim=current_price_dim,
            emotion_dim=current_emotion_dim,
            cfg=model_cfg,
        )
        tag = "F_InnovativeDualAttn" + (
            "_AdaptiveWindow" if use_adaptive else "_FixedWindow"
        )
        trainer = Trainer(
            model,
            tag,
            current_train,
            current_val,
            current_test,
            device,
            cfg=cfg.train,
        )
        reporter.add(trainer.run())

    reporter.print_summary()
    reporter.save(tag="btc")


if __name__ == "__main__":
    from config.config import DataConfig, TrainConfig

    cfg = ExperimentConfig(
        data=DataConfig(
            csv_path="btc_multimodal_hourly_dataset.csv",
            seq_len=24,
            batch_size=32,
        ),
        model=ModelConfig(
            d_model=64,
            num_heads=4,
            num_layers=2,
            dropout=0.2,
            learnable_alpha=True,
            emotion_alpha=0.5,
        ),
        train=TrainConfig(
            epochs=50,
            lr=2e-4,
            early_stopping_patience=10,
            early_stopping_metric="f1",
            grad_clip_norm=1.0,
            use_lr_scheduler=True,
            lr_scheduler_type="cosine",
            seed=42,
        ),
        run_baseline_lstm=True,
        run_baseline_gru=True,
        run_vanilla_transformer=True,
        run_cross_attn_no_modulation=True,
        run_innovative_full=True,
        use_adaptive_window=True,
    )

    run_all_experiments(cfg)