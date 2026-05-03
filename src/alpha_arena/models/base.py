from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


Batch = dict[str, Any]
ModelOutput = dict[str, Any]
LossOutput = dict[str, Any]
PredictOutput = dict[str, Any]


class BaseAlphaModel(nn.Module, ABC):
    required_batch_keys: tuple[str, ...] = (
        "x_seq",
        "x_cs",
        "x_cs_mask",
        "y_return",
        "y_risk",
        "label_date",
    )
    required_forward_keys: tuple[str, ...] = ("pred_return", "pred_var")
    required_loss_keys: tuple[str, ...] = ("loss",)
    required_predict_keys: tuple[str, ...] = ("pred",)

    def __init__(self, enable_validation: bool = True):
        super().__init__()
        self.enable_validation = enable_validation

    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput:
        raise NotImplementedError

    def compute_loss(
        self,
        batch: Batch,
        outputs: ModelOutput | None = None,
        **kwargs,
    ) -> LossOutput:
        if self.enable_validation:
            self.validate_batch(batch)

        if outputs is None:
            outputs = self.forward(batch)

        if self.enable_validation:
            self.validate_forward_output(outputs)

        loss_dict = self._compute_loss(batch, outputs, **kwargs)

        if self.enable_validation:
            self.validate_loss_output(loss_dict)

        return loss_dict

    @abstractmethod
    def _compute_loss(
        self,
        batch: Batch,
        outputs: ModelOutput,
        **kwargs,
    ) -> LossOutput:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, batch: Batch) -> PredictOutput:
        self.eval()

        if self.enable_validation:
            self.validate_batch(batch)

        outputs = self.forward(batch)

        if self.enable_validation:
            self.validate_forward_output(outputs)

        pred = outputs["pred_return"]

        if pred.ndim == 2 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)

        pred_dict = {
            "pred": pred.detach(),
        }

        if self.enable_validation:
            self.validate_predict_output(pred_dict)

        return pred_dict

    def validate_batch(self, batch: Batch) -> None:
        missing = [k for k in self.required_batch_keys if k not in batch]
        if missing:
            raise KeyError(f"Missing required batch keys: {missing}")

    def validate_forward_output(self, outputs: ModelOutput) -> None:
        missing = [k for k in self.required_forward_keys if k not in outputs]
        if missing:
            raise KeyError(f"Missing required forward output keys: {missing}")

        pred = outputs["pred"]
        if not isinstance(pred, Tensor):
            raise TypeError(f"outputs['pred'] must be Tensor, got {type(pred)}")

    def validate_loss_output(self, loss_dict: LossOutput) -> None:
        missing = [k for k in self.required_loss_keys if k not in loss_dict]
        if missing:
            raise KeyError(f"Missing required loss output keys: {missing}")

        loss = loss_dict["loss"]

        if not isinstance(loss, Tensor):
            raise TypeError(f"loss_dict['loss'] must be Tensor, got {type(loss)}")

        if loss.ndim != 0:
            raise ValueError(
                f"loss_dict['loss'] must be scalar, got {tuple(loss.shape)}"
            )

        if not torch.isfinite(loss):
            raise ValueError("loss_dict['loss'] contains NaN or Inf")

    def validate_predict_output(self, pred_dict: PredictOutput) -> None:
        missing = [k for k in self.required_predict_keys if k not in pred_dict]
        if missing:
            raise KeyError(f"Missing required predict output keys: {missing}")
