import torch
import torch.nn as nn

from models.components.positional_encoding import PositionalEncoding
from models.components.cross_attention import CrossAttention
from config.config import ModelConfig


class InnovativeDualAttentionModelM5Only(nn.Module):
    def __init__(
        self,
        price_dim: int,
        emotion_dim: int,
        cfg: ModelConfig = None,
        residual_alpha: float = 0.08,
        gate_bias_init: float = -2.0,
    ):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.d_model = cfg.d_model
        self.input_dim = price_dim + emotion_dim
        self.emotion_dim = emotion_dim

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )
        self.norm = nn.LayerNorm(self.d_model)

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

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.d_model // 2, 1),
        )

        self.register_buffer("residual_alpha", torch.tensor(float(residual_alpha)))
        self.fusion_norm = nn.LayerNorm(self.d_model)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.d_model // 2, getattr(cfg, "output_dim", 1)),
        )

        self._init_weights(gate_bias_init)

    def _init_weights(self, gate_bias_init: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        last_linear = self.gate_mlp[-1]
        if isinstance(last_linear, nn.Linear) and last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, gate_bias_init)

    def build_teacher_backbone(
        self,
        x_price: torch.Tensor,
        x_emotion: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x_price, x_emotion], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x

    def emotion_fusion(
        self,
        h_base: torch.Tensor,
        x_emotion: torch.Tensor,
    ) -> torch.Tensor:
        cross_delta = self.cross_attention(h_base, x_emotion)
        delta_e = self.delta_proj(cross_delta)

        gate_input = torch.cat([h_base, delta_e], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))

        fused = h_base + self.residual_alpha * gate * delta_e
        fused = self.fusion_norm(fused)
        return fused

    def forward(self, x_price: torch.Tensor, x_emotion: torch.Tensor) -> torch.Tensor:
        h_base = self.build_teacher_backbone(x_price, x_emotion)
        h_fused = self.emotion_fusion(h_base, x_emotion)
        last_hidden = h_fused[:, -1, :]
        return self.classifier(last_hidden).squeeze(-1)

    def set_residual_alpha(self, alpha: float) -> None:
        self.residual_alpha.fill_(float(alpha))

    def freeze_backbone(self, freeze_input_proj: bool = True) -> None:
        if freeze_input_proj:
            for p in self.input_proj.parameters():
                p.requires_grad = False
        else:
            for p in self.input_proj.parameters():
                p.requires_grad = True

        for p in self.transformer.parameters():
            p.requires_grad = False

        for p in self.norm.parameters():
            p.requires_grad = False

    def freeze_backbone_by_copy_info(self, copy_info: dict) -> None:
        freeze_input_proj = bool(copy_info.get("input_proj_weight_copied", False))
        self.freeze_backbone(freeze_input_proj=freeze_input_proj)

    def unfreeze_backbone_last_layers(self, n_last_layers: int = 1) -> None:
        for p in self.input_proj.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False

        for p in self.input_proj.parameters():
            p.requires_grad = True
        for p in self.norm.parameters():
            p.requires_grad = True

        layers = list(self.transformer.layers)
        if len(layers) == 0:
            return

        n_last_layers = max(1, min(n_last_layers, len(layers)))
        for layer in layers[-n_last_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def load_backbone_from_teacher_model(self, teacher_model, verbose: bool = True):
        copied_count = 0
        copied_keys = []
        module_copied = {
            "input_proj": [],
            "pos_encoder": [],
            "transformer": [],
            "norm": [],
        }

        mapping = [
            ("input_proj", self.input_proj),
            ("pos_encoder", self.pos_encoder),
            ("transformer", self.transformer),
            ("norm", self.norm),
        ]

        for teacher_name, student_module in mapping:
            if not hasattr(teacher_model, teacher_name):
                continue

            teacher_module = getattr(teacher_model, teacher_name)
            student_state = student_module.state_dict()
            teacher_state = teacher_module.state_dict()

            for k, v in teacher_state.items():
                if k in student_state and student_state[k].shape == v.shape:
                    student_state[k] = v.clone()
                    copied_count += 1
                    copied_keys.append(f"{teacher_name}.{k}")
                    module_copied[teacher_name].append(k)

            student_module.load_state_dict(student_state, strict=False)

        copy_info = {
            "copied_count": copied_count,
            "copied_keys": copied_keys,
            "input_proj_weight_copied": "weight" in module_copied["input_proj"],
            "input_proj_bias_copied": "bias" in module_copied["input_proj"],
            "transformer_copied_count": len(module_copied["transformer"]),
            "norm_copied_count": len(module_copied["norm"]),
        }

        if verbose:
            print(f"[M5 Teacher Loading] copied {copied_count} tensors from teacher backbone.")
            print(
                f"  input_proj.weight copied: {copy_info['input_proj_weight_copied']} | "
                f"input_proj.bias copied: {copy_info['input_proj_bias_copied']} | "
                f"transformer copied count: {copy_info['transformer_copied_count']} | "
                f"norm copied count: {copy_info['norm_copied_count']}"
            )

        return copy_info

    def load_matching_backbone_from_teacher(self, teacher_state_dict, verbose: bool = True):
        own_state = self.state_dict()
        copied = []
        skipped = []

        for k, v in teacher_state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v.clone()
                copied.append(k)
            else:
                skipped.append(k)

        self.load_state_dict(own_state, strict=False)

        if verbose:
            print(f"[Fallback Teacher Loading] copied {len(copied)} matching tensors.")

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