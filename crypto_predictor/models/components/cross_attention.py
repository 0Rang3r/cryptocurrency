import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.emotion_modulator import EmotionModulator


class CrossAttention(nn.Module):
    def __init__(
        self,
        price_dim: int,
        emotion_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_emotion_modulation: bool = True,
        modulation_mode: str = "mul",
        learnable_alpha: bool = True,
        alpha_init: float = 0.5,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_modulation = use_emotion_modulation

        self.q_proj = nn.Linear(price_dim, d_model)
        self.k_proj = nn.Linear(emotion_dim, d_model)
        self.v_proj = nn.Linear(emotion_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        if use_emotion_modulation:
            self.modulator = EmotionModulator(
                emotion_feature_dim=4,
                alpha_init=alpha_init,
                learnable_alpha=learnable_alpha,
                mode=modulation_mode,
            )

    def forward(
        self,
        x_price: torch.Tensor,
        x_emotion: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, price_len, _ = x_price.size()
        _, emotion_len, _ = x_emotion.size()

        q = self.q_proj(x_price).view(
            batch_size, price_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x_emotion).view(
            batch_size, emotion_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x_emotion).view(
            batch_size, emotion_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if self.use_modulation:
            attn_scores = self.modulator(attn_scores, x_emotion)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, price_len, self.d_model
        )

        return self.out_proj(context)