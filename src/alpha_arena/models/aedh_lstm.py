import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AEDH_LSTMConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    attn_dim: Optional[int] = None
    head_hidden_dim: int = 64
    dropout: float = 0.2
    use_last_state: bool = True
    cs_feature_dim: int = 0
    cs_feature_mask: bool = False


class TemporalAttention(nn.Module):
    """
    Temporal attention over LSTM outputs.

    Input:
        x: [B, T, H]
    Output:
        context: [B, H]
        attn_weights: [B, T]
    """

    def __init__(self, hidden_dim: int, attn_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        attn_dim = attn_dim or hidden_dim

        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor):
        """x: [B, T, H]

        x 的每一步都是一个交易日的特征向量，没有插入全零的 padding 行，因此不需要 mask。
        """
        scores = self.score(x).squeeze(-1)            # [B, T]
        attn_weights = torch.softmax(scores, dim=1)   # [B, T]
        context = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, H]
        return context, attn_weights


class MLPHead(nn.Module):
    """
    Generic prediction head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionEnhancedDualHeadLSTM(nn.Module):
    """
    Attention-Enhanced Dual-Head LSTM

    Structure:
        1. Input projection
        2. LSTM encoder
        3. Temporal attention pooling
        4. Feature fusion
        5. Dual heads:
           - return_head: predict expected return
           - risk_head:   predict log-variance / uncertainty

    Args:
        input_dim: feature dimension per timestep
        hidden_dim: LSTM hidden size
        num_layers: number of LSTM layers
        attn_dim: hidden size in attention scorer
        head_hidden_dim: hidden size of output heads
        dropout: dropout rate
        bidirectional: whether to use bidirectional LSTM
        use_last_state: whether to concatenate the last timestep hidden state
    """

    def __init__(
        self,
        config: AEDH_LSTMConfig | None = None,
    ):
        super().__init__()

        self.config = config or AEDH_LSTMConfig(input_dim=128)

        self.input_dim = self.config.input_dim
        self.hidden_dim = self.config.hidden_dim
        self.num_layers = self.config.num_layers
        self.use_last_state = self.config.use_last_state

        lstm_out_dim = self.hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attention = TemporalAttention(
            hidden_dim=lstm_out_dim,
            attn_dim=self.config.attn_dim,
            dropout=self.config.dropout,
        )

        seq_dim = lstm_out_dim
        if self.use_last_state:
            seq_dim += lstm_out_dim

        cs_raw_dim = 0
        if self.config.cs_feature_dim > 0:
            cs_raw_dim = self.config.cs_feature_dim
            if self.config.cs_feature_mask:
                cs_raw_dim += self.config.cs_feature_dim

        fused_dim = seq_dim + cs_raw_dim

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

        self.return_head = MLPHead(
            input_dim=fused_dim,
            hidden_dim=self.config.head_hidden_dim,
            output_dim=1,
            dropout=self.config.dropout,
            use_layernorm=True,
        )

        self.risk_head = MLPHead(
            input_dim=fused_dim,
            hidden_dim=self.config.head_hidden_dim,
            output_dim=1,
            dropout=self.config.dropout,
            use_layernorm=True,
        )

    def __call__(self,
        x_seq: torch.Tensor,
        x_cs: torch.Tensor,
        x_cs_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(x_seq, x_cs, x_cs_mask)

    def forward(self,
        x_seq: torch.Tensor,
        x_cs: torch.Tensor,
        x_cs_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, F]

        Returns:
            {
                "pred_return": [B],
                "pred_logvar": [B],
                "pred_var": [B],
                "pred_vol": [B],
                "attn_weights": [B, T],
                "features": [B, D],
            }
        """
        x = self.input_proj(x_seq)                 # [B, T, F]
        lstm_out, _ = self.lstm(x)             # [B, T, H]

        context, attn_weights = self.attention(lstm_out)   # [B, H], [B, T]

        parts = [context]
        if self.use_last_state:
            last_state = lstm_out[:, -1, :]   # [B, H]
            parts.append(last_state)
        parts.append(x_cs)
        if self.config.cs_feature_mask and x_cs_mask is not None:
            parts.append(x_cs_mask)

        features = torch.cat(parts, dim=-1)

        features = self.fusion(features)

        pred_return = self.return_head(features).squeeze(-1)   # [B]

        raw_var = self.risk_head(features).squeeze(-1)
        pred_var = F.softplus(raw_var) + 1e-6
        pred_logvar = torch.log(pred_var)
        pred_vol = torch.sqrt(pred_var)

        return {
            "pred_return": pred_return,
            "pred_logvar": pred_logvar,
            "pred_var": pred_var,
            "pred_vol": pred_vol,
            "attn_weights": attn_weights,
            "features": features,
        }

    @staticmethod
    def gaussian_nll_loss(
        pred_return: torch.Tensor,
        target_return: torch.Tensor,
        pred_logvar: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Heteroscedastic Gaussian negative log-likelihood:
            0.5 * [ log_var + (y - mu)^2 / exp(log_var) ]
        """
        loss = 0.5 * (
            pred_logvar + (target_return - pred_return) ** 2 / torch.exp(pred_logvar)
        )

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def mse_loss(
        pred_return: torch.Tensor,
        target_return: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        loss = (pred_return - target_return) ** 2

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def rank_ic_loss(
        pred_return: torch.Tensor,
        target_return: torch.Tensor,
        mask: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Differentiable Pearson IC loss used as a proxy for RankIC.

        Assumption:
            One batch corresponds to one label_date cross-section.
        """
        if mask is not None:
            mask = mask.bool()
            pred_return = pred_return[mask]
            target_return = target_return[mask]

        if pred_return.numel() < 2:
            return pred_return.new_zeros(())

        x = pred_return - pred_return.mean()
        y = target_return - target_return.mean()

        x_var = (x * x).mean()
        y_var = (y * y).mean()

        if x_var.detach() < eps or y_var.detach() < eps:
            return pred_return.new_zeros(())

        corr = (x * y).mean() / torch.sqrt(x_var * y_var + eps)

        return -corr


    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_return: torch.Tensor,
        loss_type: str = "gaussian_nll",
        alpha_rank: float = 0.0,
        alpha_mse: float = 0.0,   # ⭐ 新增
        var_min: float = 1e-4,    # ⭐ 防止 variance collapse
    ) -> Dict[str, torch.Tensor]:

        pred_return = outputs["pred_return"]
        pred_var = outputs["pred_var"]

        # =====================
        # 1. clamp variance（非常重要）作用：防止模型预测的方差过小，导致训练不稳定甚至 NaN。
        # =====================
        pred_var = torch.clamp(pred_var, min=var_min)

        # =====================
        # 2. MSE regularization（推荐加）
        # =====================
        mse_loss = F.mse_loss(pred_return, target_return)
        # =====================
        # 3. base loss
        # =====================
        if loss_type == "gaussian_nll":
            nll_loss = F.gaussian_nll_loss(
                input=pred_return,
                target=target_return,
                var=pred_var,
                full=False,
                reduction="mean"
            )
            base_loss = nll_loss

        elif loss_type == "mse":
            nll_loss = pred_return.new_zeros(())
            base_loss = mse_loss

        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # =====================
        # 4. rank loss
        # =====================
        if alpha_rank > 0:
            rank_loss = self.rank_ic_loss(pred_return, target_return)
        else:
            rank_loss = torch.tensor(0.0, device=target_return.device)

        # =====================
        # 5. total loss
        # =====================
        total_loss = base_loss \
                + alpha_mse * mse_loss \
                + alpha_rank * rank_loss

        # =====================
        # 6. 额外监控（强烈推荐）
        # =====================
        pred_std = torch.sqrt(pred_var)

        return {
            "loss": total_loss,
            "base_loss": base_loss.detach(),
            "nll_loss": nll_loss.detach(),
            "mse_loss": mse_loss.detach(),
            "rank_loss": rank_loss.detach(),
            "pred_var_mean": pred_var.mean().detach(),
            "pred_var_min": pred_var.min().detach(),
            "pred_var_max": pred_var.max().detach(),
            "pred_std_mean": pred_std.mean().detach(),
            "pred_return_mean": pred_return.mean().detach(),
            "pred_return_std": pred_return.std(unbiased=False).detach(),
        }
    

    def compute_loss_legacy(
        self,
        outputs: Dict[str, torch.Tensor],
        target_return: torch.Tensor,
        loss_type: str = "gaussian_nll",
        alpha_rank: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        pred_return = outputs["pred_return"]
        pred_var = outputs["pred_var"]

        if loss_type == "gaussian_nll":
            base_loss = F.gaussian_nll_loss(
                input=pred_return,
                target=target_return,
                var=pred_var,
                full=False,
                reduction="mean"
            )
        elif loss_type == "mse":
            base_loss = self.mse_loss(pred_return, target_return)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        if alpha_rank > 0:
            rank_loss = self.rank_ic_loss(pred_return, target_return)
        else:
            rank_loss = torch.tensor(0.0, device=target_return.device)

        total_loss = base_loss + alpha_rank * rank_loss

        return {
            "loss": total_loss,
            "base_loss": base_loss.detach(),
            "rank_loss": rank_loss.detach(),
        }

    def compute_loss_legacy2(
        self,
        outputs,
        target_return,
        target_risk=None,
        loss_type="gaussian_nll",
        alpha_rank=0.0,
        alpha_risk=0.0,
    ):
        pred_return = outputs["pred_return"]
        pred_var = outputs["pred_var"]
        pred_vol = outputs["pred_vol"]

        if loss_type == "gaussian_nll":
            base_loss = F.gaussian_nll_loss(
                input=pred_return,
                target=target_return,
                var=pred_var,
                full=False,
                reduction="mean",
            )
        elif loss_type == "mse":
            base_loss = F.mse_loss(pred_return, target_return)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        if alpha_rank > 0:
            rank_loss = self.rank_ic_loss(pred_return, target_return)
        else:
            rank_loss = pred_return.new_zeros(())

        if alpha_risk > 0:
            if target_risk is None:
                raise ValueError("target_risk is required when alpha_risk > 0")
            risk_loss = F.mse_loss(pred_vol, target_risk)
        else:
            risk_loss = pred_return.new_zeros(())

        total_loss = base_loss + alpha_rank * rank_loss + alpha_risk * risk_loss

        return {
            "loss": total_loss,
            "base_loss": base_loss.detach(),
            "rank_loss": rank_loss.detach(),
            "risk_loss": risk_loss.detach(),
        }