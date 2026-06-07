from __future__ import annotations

from typing import Any, Dict, Iterable

from .base_dataset import BaseDataset
from .base_model import BaseModel


class BaseRunner:
    """Common training/evaluation loop logic."""

    def __init__(self, model: BaseModel, dataset: BaseDataset) -> None:
        self.model = model
        self.dataset = dataset

    def run(self) -> None:
        for batch in self.dataset:
            outputs = self.model.forward(batch)
            loss = self.model.loss(outputs, batch)
            self.after_step(loss)

    def after_step(self, loss: Any) -> None:
        print(f"loss={loss}")
