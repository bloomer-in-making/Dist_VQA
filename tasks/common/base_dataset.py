from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class BaseDataset(ABC):
    """Common dataset interface."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def __iter__(self) -> Iterable[Dict[str, Any]]:
        """Build a batch."""
