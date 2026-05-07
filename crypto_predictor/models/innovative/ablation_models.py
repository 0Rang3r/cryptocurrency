import torch
import torch.nn as nn

from models.components.positional_encoding import PositionalEncoding
from models.components.cross_attention import CrossAttention
from config.config import ModelConfig


class CrossAttnNoModulation(nn.Module):
    def __init__(self, price_dim: int, emotion_dim: int, cfg: ModelConfig = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.d_model = cfg.d_model
        self.cross_attention = CrossAttention(
            price_dim=price_dim,
            emotion_dim=emotion_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            use_emotion_modulation=False,
        )
        self.cross_norm = nn.LayerNorm(cfg.d_model)

        self.pos_encoder = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )
        self.self_norm = nn.LayerNorm(cfg.d_model)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.output_dim),
        )

    def forward(self, x_price: torch.Tensor, x_emotion: torch.Tensor) -> torch.Tensor:
        cross_out = self.cross_norm(self.cross_attention(x_price, x_emotion))
        self_in = self.pos_encoder(cross_out)
        self_out = self.self_attn_encoder(self_in)
        last = self.self_norm(self_out[:, -1, :])
        return self.classifier(last).squeeze(-1)


class AdditiveModulationModel(nn.Module):
    def __init__(self, price_dim: int, emotion_dim: int, cfg: ModelConfig = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.d_model = cfg.d_model
        self.cross_attention = CrossAttention(
            price_dim=price_dim,
            emotion_dim=emotion_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            use_emotion_modulation=True,
            modulation_mode="add",
            learnable_alpha=cfg.learnable_alpha,
            alpha_init=cfg.emotion_alpha,
        )
        self.cross_norm = nn.LayerNorm(cfg.d_model)

        self.pos_encoder = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )
        self.self_norm = nn.LayerNorm(cfg.d_model)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.output_dim),
        )

    def forward(self, x_price: torch.Tensor, x_emotion: torch.Tensor) -> torch.Tensor:
        cross_out = self.cross_norm(self.cross_attention(x_price, x_emotion))
        self_in = self.pos_encoder(cross_out)
        self_out = self.self_attn_encoder(self_in)
        last = self.self_norm(self_out[:, -1, :])
        return self.classifier(last).squeeze(-1)