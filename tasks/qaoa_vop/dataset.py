from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import BaseDataset


def maxcut_bruteforce(A: np.ndarray, W: np.ndarray) -> float:
    """Compute the MaxCut optimum by brute force."""
    n = A.shape[0]
    best = -1e18

    for mask in range(1 << n):
        z = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)], dtype=float)
        val = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] != 0:
                    wij = float(W[i, j])
                    val += 0.5 * wij * (1.0 - z[i] * z[j])
        if val > best:
            best = val
    return best


def collate_graph(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for graph batches."""
    xs = torch.stack([it["x"] for it in batch], dim=0)
    Hc_list = [it["Hc"] for it in batch]
    out = {"x": xs, "Hc": Hc_list}
    if "opt" in batch[0]:
        out["opt"] = torch.tensor([it["opt"] for it in batch], dtype=torch.float32)
    return out

class QAOADataset(BaseDataset):
    """QAOA MaxCut dataset."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        n = config.get("n", 10)
        num_graphs = config.get("num_graphs", 256)
        p_range = tuple(config.get("p_range", [0.2, 0.8]))
        weighted = config.get("weighted", "uniform")
        seed = config.get("seed", 7)

        rng = np.random.default_rng(seed)
        self.items = []

        for _ in range(num_graphs):

            edge_p = float(rng.uniform(*p_range))
            A = (rng.uniform(size=(n, n)) < edge_p).astype(int)
            A = np.triu(A, 1)
            A = A + A.T
            np.fill_diagonal(A, 0)


            if weighted == "none":
                W = A.astype(float)
            elif weighted == "pm1":
                signs = rng.choice([-1.0, 1.0], size=(n, n))
                W = A * signs
            elif weighted == "uniform":
                w = rng.uniform(0.5, 2.0, size=(n, n))
                W = A * w
            else:
                raise ValueError(f"weighted must be one of {{'none','pm1','uniform'}}")

            import pennylane as qml


            n_h = A.shape[0]
            obs, coeffs = [], []
            for i in range(n_h):
                for j in range(i + 1, n_h):
                    if A[i, j] != 0:
                        w = float(W[i, j])
                        obs.append(qml.Identity(wires=[i, j] if i != j else i))
                        coeffs.append(0.5 * w)
                        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                        coeffs.append(-0.5 * w)
            Hc = qml.Hamiltonian(coeffs, obs)

            iu = np.triu_indices(n, k=1)
            adj_vec = A[iu].astype(np.float32)
            x = torch.from_numpy(adj_vec)

            opt_val = maxcut_bruteforce(A, W)

            self.items.append({"x": x, "Hc": Hc, "opt": opt_val})

        self.x_dim = self.items[0]["x"].numel()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        """For BaseDataset interface compatibility."""
        for idx in range(len(self)):
            yield self[idx]

class QAOADatasetTest(BaseDataset):
    """QAOA MaxCut dataset."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        n = config.get("n", 10)
        num_graphs = config.get("num_graphs", 256)
        p_range = tuple(config.get("p_range", [0.2, 0.8]))
        weighted = config.get("weighted", "uniform")
        seed = config.get("seed", 7)
        seed = seed + 10

        rng = np.random.default_rng(seed)
        self.items = []

        for _ in range(num_graphs):

            edge_p = float(rng.uniform(*p_range))
            A = (rng.uniform(size=(n, n)) < edge_p).astype(int)
            A = np.triu(A, 1)
            A = A + A.T
            np.fill_diagonal(A, 0)


            if weighted == "none":
                W = A.astype(float)
            elif weighted == "pm1":
                signs = rng.choice([-1.0, 1.0], size=(n, n))
                W = A * signs
            elif weighted == "uniform":
                w = rng.uniform(0.5, 2.0, size=(n, n))
                W = A * w
            else:
                raise ValueError(f"weighted must be one of {{'none','pm1','uniform'}}")

            import pennylane as qml


            n_h = A.shape[0]
            obs, coeffs = [], []
            for i in range(n_h):
                for j in range(i + 1, n_h):
                    if A[i, j] != 0:
                        w = float(W[i, j])
                        obs.append(qml.Identity(wires=[i, j] if i != j else i))
                        coeffs.append(0.5 * w)
                        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                        coeffs.append(-0.5 * w)
            Hc = qml.Hamiltonian(coeffs, obs)

            iu = np.triu_indices(n, k=1)
            adj_vec = A[iu].astype(np.float32)
            x = torch.from_numpy(adj_vec)

            opt_val = maxcut_bruteforce(A, W)

            self.items.append({"x": x, "Hc": Hc, "opt": opt_val})

        self.x_dim = self.items[0]["x"].numel()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        """For BaseDataset interface compatibility."""
        for idx in range(len(self)):
            yield self[idx]
