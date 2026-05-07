import torch
import torch.nn as nn

from models.components.positional_encoding import PositionalEncoding
from models.components.cross_attention import CrossAttention
from config.config import ModelConfig


class InnovativeDualAttentionModel(nn.Module):
    def __init__(
        self,
        price_dim: int,
        emotion_dim: int,
        cfg: ModelConfig = None,
        residual_alpha: float = 0.1,
        learnable_residual_alpha: bool = False,
        upper_num_layers: int = 1,
    ):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.d_model = cfg.d_model
        self.emotion_dim = emotion_dim

        self.price_proj = nn.Linear(price_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=cfg.dropout)

        price_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.price_transformer = nn.TransformerEncoder(
            price_encoder_layer,
            num_layers=cfg.num_layers,
        )
        self.price_norm = nn.LayerNorm(self.d_model)

        self.cross_attention = CrossAttention(
            price_dim=self.d_model,
            emotion_dim=emotion_dim,
            d_model=self.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            use_emotion_modulation=True,
            modulation_mode="mul",
            learnable_alpha=getattr(cfg, "learnable_alpha", True),
            alpha_init=getattr(cfg, "emotion_alpha", 0.5),
        )

        self.delta_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        if learnable_residual_alpha:
            self.residual_alpha = nn.Parameter(torch.tensor(float(residual_alpha)))
        else:
            self.register_buffer("residual_alpha", torch.tensor(float(residual_alpha)))

        self.fusion_norm = nn.LayerNorm(self.d_model)

        upper_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn_encoder = nn.TransformerEncoder(
            upper_encoder_layer,
            num_layers=upper_num_layers,
        )
        self.self_norm = nn.LayerNorm(self.d_model)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.d_model // 2, getattr(cfg, "output_dim", 1)),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def build_price_backbone(self, x_price: torch.Tensor) -> torch.Tensor:
        h = self.price_proj(x_price)
        h = self.pos_encoder(h)
        h = self.price_transformer(h)
        h = self.price_norm(h)
        return h

    def emotion_fusion(self, h_base: torch.Tensor, x_emotion: torch.Tensor) -> torch.Tensor:
        cross_delta = self.cross_attention(h_base, x_emotion)
        delta_e = self.delta_proj(cross_delta)
        fused = h_base + self.residual_alpha * delta_e
        fused = self.fusion_norm(fused)
        return fused

    def forward(self, x_price: torch.Tensor, x_emotion: torch.Tensor) -> torch.Tensor:
        h_base = self.build_price_backbone(x_price)
        h_fused = self.emotion_fusion(h_base, x_emotion)

        upper_in = self.pos_encoder(h_fused)
        upper_out = self.self_attn_encoder(upper_in)
        last_hidden = self.self_norm(upper_out[:, -1, :])

        return self.classifier(last_hidden).squeeze(-1)

    def set_residual_alpha(self, alpha: float) -> None:
        if isinstance(self.residual_alpha, nn.Parameter):
            with torch.no_grad():
                self.residual_alpha.fill_(float(alpha))
        else:
            self.residual_alpha.fill_(float(alpha))

    def freeze_backbone(self) -> None:
        for module in [self.price_proj, self.price_transformer, self.price_norm]:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for module in [self.price_proj, self.price_transformer, self.price_norm]:
            for p in module.parameters():
                p.requires_grad = True

    def unfreeze_backbone_last_layers(self, n_last_layers: int = 1) -> None:
        for p in self.price_proj.parameters():
            p.requires_grad = False
        for p in self.price_transformer.parameters():
            p.requires_grad = False
        for p in self.price_norm.parameters():
            p.requires_grad = False

        for p in self.price_proj.parameters():
            p.requires_grad = True
        for p in self.price_norm.parameters():
            p.requires_grad = True

        layers = list(self.price_transformer.layers)
        if len(layers) == 0:
            return

        n_last_layers = max(1, min(n_last_layers, len(layers)))
        for layer in layers[-n_last_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def load_matching_backbone_from_teacher(self, teacher_ckpt_or_state_dict, verbose: bool = True):
        if isinstance(teacher_ckpt_or_state_dict, str):
            obj = torch.load(
                teacher_ckpt_or_state_dict,
                map_location="cpu",
                weights_only=False,
            )
            teacher_state = obj.get("model_state_dict", obj)
        else:
            teacher_state = teacher_ckpt_or_state_dict

        own_state = self.state_dict()
        copied = []
        skipped = []

        for k, v in teacher_state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v.clone()
                copied.append(k)
            else:
                skipped.append(k)

        self.load_state_dict(own_state, strict=False)

        if verbose:
            print(f"[Teacher Loading] copied {len(copied)} matching tensors.")
            if len(copied) > 0:
                print("  example copied keys:", copied[:8])

        return copied, skipped

    def distill_loss_from_teacher_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        weight: float = 0.2,
    ) -> torch.Tensor:
        student_prob = torch.sigmoid(student_logits)
        teacher_prob = torch.sigmoid(teacher_logits).detach()
        return weight * nn.functional.mse_loss(student_prob, teacher_prob)