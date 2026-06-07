from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    """Common model interface."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Model inference logic."""

    @abstractmethod
    def loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Any:
        """Compute loss."""
