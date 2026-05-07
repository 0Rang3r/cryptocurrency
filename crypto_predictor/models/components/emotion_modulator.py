import torch
import torch.nn as nn


class EmotionModulator(nn.Module):
    def __init__(
        self,
        emotion_feature_dim: int = 4,
        hidden_dim: int = 16,
        alpha_init: float = 0.5,
        learnable_alpha: bool = True,
        mode: str = "mul",
    ):
        super().__init__()
        assert mode in ("mul", "add"), "mode 必须为 'mul' 或 'add'"
        self.mode = mode

        self.mlp = nn.Sequential(
            nn.Linear(emotion_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_init))

    def _extract_pure_emotions(self, emotion_seq: torch.Tensor) -> torch.Tensor:
        feature_dim = emotion_seq.size(-1)

        if feature_dim == 4:
            return emotion_seq

        if feature_dim == 5:
            idx = [1, 2, 3, 4]
        elif feature_dim == 10:
            idx = [3, 5, 7, 9]
        elif feature_dim == 22:
            idx = [5, 9, 13, 17]
        else:
            raise ValueError(
                f"EmotionModulator 不支持当前 emotion 维度 E={feature_dim}，"
                f"目前仅支持 E in {{4, 5, 10, 22}}"
            )

        idx = torch.tensor(idx, device=emotion_seq.device, dtype=torch.long)
        return torch.index_select(emotion_seq, dim=-1, index=idx)

    def forward(self, attn_scores: torch.Tensor, emotion_seq: torch.Tensor) -> torch.Tensor:
        pure_emotions = self._extract_pure_emotions(emotion_seq)
        modulation = self.mlp(pure_emotions)
        modulation = torch.sigmoid(modulation)

        modulation = modulation.squeeze(-1).unsqueeze(1).unsqueeze(2)

        if self.mode == "mul":
            return attn_scores * (1.0 + self.alpha * modulation)

        return attn_scores + self.alpha * modulation