from __future__ import annotations

import os
from typing import Any, Dict, Iterable

import numpy as np
import torch

from ..common import BaseDataset


def convert_array(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a torch tensor."""
    if array.dtype == np.float64:
        return torch.tensor(array, dtype=torch.float32)
    return torch.tensor(array)


def custom_collate_fn(samples: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Batch collate function."""
    x = [convert_array(sample["x"]) for sample in samples]
    y = [convert_array(sample["y"]) for sample in samples]

    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return {"x": x, "y": y}


class XYZDataset(BaseDataset):
    """XYZ Hamiltonian dataset."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        qubit = config.get("qubit", 20)
        data_dir = config.get("data_dir", "xyz_ham")
        self.file = os.path.join(data_dir, f"{qubit}_data_continuous_3000.csv")
        if not os.path.exists(self.file):
            raise FileNotFoundError(f"data file not found: {self.file}")

        self.data = np.loadtxt(self.file, delimiter=",")

    def __len__(self) -> int:
        return self.data.shape[-1]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        x = self.data[0:3, idx]
        y = self.data[-1, idx]
        return {"x": x, "y": y}

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        """For BaseDataset interface compatibility."""
        for idx in range(len(self)):
            yield self[idx]

class XYZDatasetTest(BaseDataset):
    """XYZ Hamiltonian dataset."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        qubit = config.get("qubit", 20)
        data_dir = config.get("data_dir", "xyz_ham")
        self.file = os.path.join(data_dir, f"{qubit}_data_continuous_3000_test.csv")
        if not os.path.exists(self.file):
            raise FileNotFoundError(f"data file not found: {self.file}")

        self.data = np.loadtxt(self.file, delimiter=",")

    def __len__(self) -> int:
        return self.data.shape[-1]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        x = self.data[0:3, idx]
        y = self.data[-1, idx]
        return {"x": x, "y": y}

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        """For BaseDataset interface compatibility."""
        for idx in range(len(self)):
            yield self[idx]
