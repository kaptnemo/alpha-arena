import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# =========================
# Config
# =========================

@dataclass
class MemoryCacheConfig:
    hidden_dim: int
    segment_len: int = 10
    cache_mode: str = "gated_residual"   # ["residual", "gated_residual", "mean", "topk_gated"]
    max_cached_segments: Optional[int] = None
    topk: int = 4
    dropout: float = 0.1
    use_segment_context: bool = True
    detach_cached_memory: bool = False


class alpha_arenaConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    attn_dim: Optional[int] = None
    head_hidden_dim: int = 64
    dropout: float = 0.2
    segment_len: int = 10
    max_cached_segments: Optional[int] = None
    cache_mode: str = "gated_residual"
    topk: int = 4
    detach_cached_memory: bool = False


# =========================
# Temporal Attention
# =========================

class TemporalAttention(nn.Module):
    """
    Temporal attention pooling over sequence output.

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x).squeeze(-1)            # [B, T]
        attn_weights = torch.softmax(scores, dim=1)   # [B, T]
        context = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, H]
        return context, attn_weights


# =========================
# Segment Memory Retriever
# =========================

class SegmentMemoryRetriever(nn.Module):
    """
    Retrieve from cached segment memories.
    """

    def __init__(self, cfg: MemoryCacheConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        self.query_proj = nn.Linear(h, h, bias=False)
        self.key_proj = nn.Linear(h, h, bias=False)
        self.value_proj = nn.Linear(h, h, bias=False)

        self.gate_net = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, h),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(h, h),
            nn.Dropout(cfg.dropout),
        )

    def _stack_cache(
        self,
        cached_memory: List[torch.Tensor],
        cached_context: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mem = torch.stack(cached_memory, dim=1)  # [B, S, H]
        ctx = None
        if cached_context is not None and len(cached_context) > 0:
            ctx = torch.stack(cached_context, dim=1)  # [B, S, H]
        return mem, ctx

    def _relevance_scores(
        self,
        query: torch.Tensor,                      # [B, H]
        memory_bank: torch.Tensor,               # [B, S, H]
        context_bank: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.query_proj(query).unsqueeze(1)  # [B, 1, H]

        if self.cfg.use_segment_context and context_bank is not None:
            keys = self.key_proj(context_bank)   # [B, S, H]
        else:
            keys = self.key_proj(memory_bank)    # [B, S, H]

        scores = torch.sum(q * keys, dim=-1) / math.sqrt(query.size(-1))  # [B, S]
        return scores

    def forward(
        self,
        query: torch.Tensor,                     # [B, H]
        cached_memory: List[torch.Tensor],
        cached_context: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        device = query.device
        B, H = query.shape

        if len(cached_memory) == 0:
            zero = torch.zeros(B, H, device=device, dtype=query.dtype)
            return zero, None, {
                "cache_scores": None,
                "cache_indices": None,
            }

        memory_bank, context_bank = self._stack_cache(cached_memory, cached_context)  # [B, S, H]
        scores = self._relevance_scores(query, memory_bank, context_bank)              # [B, S]

        mode = self.cfg.cache_mode

        if mode in {"residual", "gated_residual", "mean"}:
            attn = torch.softmax(scores, dim=-1)   # [B, S]
            values = self.value_proj(memory_bank)  # [B, S, H]
            retrieved = torch.sum(attn.unsqueeze(-1) * values, dim=1)  # [B, H]

            if mode == "mean":
                out = self.out_proj(retrieved)
                return out, attn, {
                    "cache_scores": scores,
                    "cache_indices": None,
                }

            if mode == "residual":
                out = self.out_proj(query + retrieved)
                return out, attn, {
                    "cache_scores": scores,
                    "cache_indices": None,
                }

            gate_in = torch.cat([query, retrieved, query - retrieved], dim=-1)  # [B, 3H]
            gate = self.gate_net(gate_in)                                        # [B, H]
            fused = gate * query + (1.0 - gate) * retrieved
            out = self.out_proj(fused)
            return out, attn, {
                "cache_scores": scores,
                "cache_indices": None,
                "gate": gate,
            }

        elif mode == "topk_gated":
            k = min(self.cfg.topk, scores.size(1))
            topk_scores, topk_idx = torch.topk(scores, k=k, dim=-1)  # [B, K], [B, K]

            values = self.value_proj(memory_bank)  # [B, S, H]
            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, values.size(-1))
            topk_values = torch.gather(values, dim=1, index=gather_idx)  # [B, K, H]

            topk_attn = torch.softmax(topk_scores, dim=-1)
            retrieved = torch.sum(topk_attn.unsqueeze(-1) * topk_values, dim=1)  # [B, H]

            gate_in = torch.cat([query, retrieved, query - retrieved], dim=-1)
            gate = self.gate_net(gate_in)
            fused = gate * query + (1.0 - gate) * retrieved
            out = self.out_proj(fused)

            return out, topk_attn, {
                "cache_scores": scores,
                "cache_indices": topk_idx,
                "gate": gate,
            }

        else:
            raise ValueError(f"Unsupported cache_mode: {mode}")


# =========================
# Multi-Layer Memory Caching LSTM
# =========================

class MultiLayerMemoryCachingLSTM(nn.Module):
    """
    Multi-layer backbone with memory caching only on the top layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
        cache_cfg: Optional[MemoryCacheConfig] = None,
    ):
        super().__init__()

        if bidirectional:
            raise ValueError(
                "MultiLayerMemoryCachingLSTM is intended for causal modeling. "
                "Set bidirectional=False."
            )
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
            self.cells.append(nn.LSTMCell(layer_input_dim, hidden_dim))

        self.state_dropout = nn.Dropout(dropout)

        self.cache_cfg = cache_cfg or MemoryCacheConfig(hidden_dim=hidden_dim)
        if self.cache_cfg.hidden_dim != hidden_dim:
            raise ValueError("cache_cfg.hidden_dim must equal hidden_dim")

        self.retriever = SegmentMemoryRetriever(self.cache_cfg)

        self.fuse_current = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _truncate_cache(
        self,
        cached_memory: List[torch.Tensor],
        cached_context: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        max_cached = self.cache_cfg.max_cached_segments
        if max_cached is None or len(cached_memory) <= max_cached:
            return cached_memory, cached_context
        return cached_memory[-max_cached:], cached_context[-max_cached:]

    def _init_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        hx: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if hx is None:
            h_list = [
                torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
            c_list = [
                torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h_list, c_list = hx
            if len(h_list) != self.num_layers or len(c_list) != self.num_layers:
                raise ValueError("hx must contain num_layers hidden states and cell states")
        return h_list, c_list

    def forward(
        self,
        x: torch.Tensor,  # [B, T, F]
        hx: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    ) -> Dict[str, object]:
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        x = self.input_proj(x)
        h_list, c_list = self._init_states(B, device, dtype, hx)

        seg_len = self.cache_cfg.segment_len

        outputs: List[torch.Tensor] = []
        cache_weights: List[Optional[torch.Tensor]] = []
        cache_aux: List[Dict[str, torch.Tensor]] = []

        cached_memory: List[torch.Tensor] = []
        cached_context: List[torch.Tensor] = []
        current_segment_states: List[torch.Tensor] = []

        for t in range(T):
            layer_input = x[:, t, :]  # [B, F]

            # Stacked LSTM
            for layer_idx, cell in enumerate(self.cells):
                h_t, c_t = cell(layer_input, (h_list[layer_idx], c_list[layer_idx]))
                h_t = self.state_dropout(h_t)

                h_list[layer_idx] = h_t
                c_list[layer_idx] = c_t
                layer_input = h_t

            # Top layer state
            top_h_t = h_list[-1]

            # Memory retrieval on top layer only
            retrieved_t, weights_t, aux_t = self.retriever(
                query=top_h_t,
                cached_memory=cached_memory,
                cached_context=cached_context if len(cached_context) > 0 else None,
            )

            enhanced_t = self.fuse_current(torch.cat([top_h_t, retrieved_t], dim=-1))  # [B, H]

            outputs.append(enhanced_t)
            cache_weights.append(weights_t)
            cache_aux.append(aux_t)

            current_segment_states.append(top_h_t)

            segment_end = ((t + 1) % seg_len == 0) or (t == T - 1)
            if segment_end:
                seg_states = torch.stack(current_segment_states, dim=1)  # [B, Ls, H]

                seg_memory = seg_states[:, -1, :]    # [B, H]
                seg_context = seg_states.mean(dim=1) # [B, H]

                if self.cache_cfg.detach_cached_memory:
                    seg_memory = seg_memory.detach()
                    seg_context = seg_context.detach()

                cached_memory.append(seg_memory)
                cached_context.append(seg_context)

                cached_memory, cached_context = self._truncate_cache(
                    cached_memory, cached_context
                )
                current_segment_states = []

        sequence_output = torch.stack(outputs, dim=1)  # [B, T, H]

        return {
            "sequence_output": sequence_output,
            "final_hidden": sequence_output[:, -1, :],
            "final_cell": c_list[-1],
            "all_layer_hidden": h_list,
            "all_layer_cell": c_list,
            "cache_weights": cache_weights,
            "cache_aux": cache_aux,
            "num_cached_segments": len(cached_memory),
        }


# =========================
# Generic Head
# =========================

class MLPHead(nn.Module):
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


# =========================
# Full Model
# =========================

class AttentionEnhancedDualHeadalpha_arena(nn.Module):
    """
    Full architecture:
        MultiLayerMemoryCachingLSTM
        -> TemporalAttention
        -> Fusion(context + final_hidden)
        -> Dual Heads (return / risk)

    Input:
        x: [B, T, F]

    Output:
        {
            "pred_return": [B]
            "pred_logvar": [B]
            "pred_var": [B]
            "pred_vol": [B]
            "attn_weights": [B, T]
            "cache_weights": list
            "cache_aux": list
            "features": [B, H]
            "sequence_output": [B, T, H]
        }
    """

    def __init__(
        self,
        config: alpha_arenaConfig | None = None,
    ):
        super().__init__()

        self.config = config or alpha_arenaConfig(input_dim=128)
        self.input_dim = self.config.input_dim
        self.hidden_dim = self.config.hidden_dim
        self.num_layers = self.config.num_layers

        cache_cfg = MemoryCacheConfig(
            hidden_dim=self.hidden_dim,
            segment_len=self.config.segment_len,
            cache_mode=self.config.cache_mode,
            max_cached_segments=self.config.max_cached_segments,
            topk=self.config.topk,
            dropout=self.config.dropout,
            use_segment_context=True,
            detach_cached_memory=self.config.detach_cached_memory,
        )

        self.backbone = MultiLayerMemoryCachingLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.config.dropout,
            bidirectional=False,
            cache_cfg=cache_cfg,
        )

        self.attention = TemporalAttention(
            hidden_dim=self.hidden_dim,
            attn_dim=self.config.attn_dim,
            dropout=self.config.dropout,
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

        self.return_head = MLPHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.config.head_hidden_dim,
            output_dim=1,
            dropout=self.config.dropout,
            use_layernorm=True,
        )

        self.risk_head = MLPHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.config.head_hidden_dim,
            output_dim=1,
            dropout=self.config.dropout,
            use_layernorm=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(x, hx=hx)

        sequence_output = backbone_out["sequence_output"]  # [B, T, H]
        final_hidden = backbone_out["final_hidden"]        # [B, H]

        context, attn_weights = self.attention(sequence_output)  # [B, H], [B, T]

        features = self.feature_fusion(torch.cat([context, final_hidden], dim=-1))  # [B, H]

        pred_return = self.return_head(features).squeeze(-1)   # [B]
        pred_logvar = self.risk_head(features).squeeze(-1)     # [B]
        pred_logvar = torch.clamp(pred_logvar, min=-10.0, max=10.0)

        pred_var = torch.exp(pred_logvar)
        pred_vol = torch.sqrt(pred_var + 1e-8)

        return {
            "pred_return": pred_return,
            "pred_logvar": pred_logvar,
            "pred_var": pred_var,
            "pred_vol": pred_vol,
            "attn_weights": attn_weights,
            "cache_weights": backbone_out["cache_weights"],
            "cache_aux": backbone_out["cache_aux"],
            "features": features,
            "sequence_output": sequence_output,
            "final_hidden": final_hidden,
            "num_cached_segments": backbone_out["num_cached_segments"],
        }

    @staticmethod
    def gaussian_nll_loss(
        pred_return: torch.Tensor,
        target_return: torch.Tensor,
        pred_logvar: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
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
        eps: float = 1e-8,
    ) -> torch.Tensor:
        x = pred_return - pred_return.mean()
        y = target_return - target_return.mean()

        corr = (x * y).mean() / (x.std(unbiased=False) * y.std(unbiased=False) + eps)
        return -corr

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_return: torch.Tensor,
        loss_type: str = "gaussian_nll",
        alpha_rank: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        pred_return = outputs["pred_return"]
        pred_logvar = outputs["pred_logvar"]

        if loss_type == "gaussian_nll":
            base_loss = self.gaussian_nll_loss(pred_return, target_return, pred_logvar)
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