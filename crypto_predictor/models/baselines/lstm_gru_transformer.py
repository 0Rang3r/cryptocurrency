import torch
import torch.nn as nn

from models.components.positional_encoding import PositionalEncoding


class BaselineLSTM(nn.Module):
    def __init__(
        self,
        price_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        x_price: torch.Tensor,
        x_emotion: torch.Tensor = None,
    ) -> torch.Tensor:
        out, _ = self.lstm(x_price)
        last_hidden = self.norm(out[:, -1, :])
        return self.classifier(last_hidden).squeeze(-1)


class MultimodalGRU(nn.Module):
    def __init__(
        self,
        price_dim: int,
        emotion_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        input_dim = price_dim + emotion_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x_price: torch.Tensor, x_emotion: torch.Tensor) -> torch.Tensor:
        x_concat = torch.cat([x_price, x_emotion], dim=-1)
        out, _ = self.gru(x_concat)
        last_hidden = self.norm(out[:, -1, :])
        return self.classifier(last_hidden).squeeze(-1)


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        price_dim: int,
        emotion_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        input_dim = price_dim + emotion_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x_price: torch.Tensor, x_emotion: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_price, x_emotion], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        last = self.norm(x[:, -1, :])
        return self.classifier(last).squeeze(-1)