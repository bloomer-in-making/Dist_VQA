from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from ..common import BaseModel


def maxcut_hamiltonian(adj: np.ndarray, weights: np.ndarray | None = None) -> qml.Hamiltonian:
    """Build the MaxCut Hamiltonian."""
    n = adj.shape[0]
    obs, coeffs = [], []
    W = weights if weights is not None else np.ones_like(adj, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] != 0:
                w = float(W[i, j])
                obs.append(qml.Identity(wires=[i, j] if i != j else i))
                coeffs.append(0.5 * w)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coeffs.append(-0.5 * w)

    return qml.Hamiltonian(coeffs, obs)


class MLP(nn.Module):
    """Simple MLP."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HyperNetQAOA(nn.Module):
    """HyperNetwork for QAOA: adjacency vector -> (gammas, betas)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        p: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        self.p = p
        self.device = torch.device(device)

        self.head = MLP(in_dim, hidden_dim, out_dim=2 * p)

        self.to(self.device)
        for name, param in self.named_parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Input x -> (gammas, betas)."""
        raw = self.head(x)
        gammas = raw[..., : self.p]
        betas = raw[..., self.p :]
        return gammas, betas


def build_qaoa_qnode(n: int, p: int, dev_name: str = "lightning.gpu", mixer_type: str = "x"):
    """Build the QAOA QNode."""
    dev = qml.device(dev_name, wires=n)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def qaoa_energy(gammas: torch.Tensor, betas: torch.Tensor, Hc: qml.Hamiltonian):

        for w in range(n):
            qml.Hadamard(wires=w)

        for layer in range(p):
            g = gammas[layer]
            b = betas[layer]

            qml.templates.ApproxTimeEvolution(Hc, g, 1)

            if mixer_type == "x":
                for w in range(n):
                    qml.RX(2.0 * b, wires=w)
            else:
                raise ValueError(f"Unknown mixer_type: {mixer_type}")

        return qml.expval(Hc)

    return qaoa_energy


class QAOAModel(BaseModel):
    """Model wrapper for the QAOA task."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        model_cfg = config.get("model", {})
        n = config.get("n", 10)
        p = config.get("p", 4)

        in_dim = n * (n - 1) // 2

        self.hypernet = HyperNetQAOA(
            in_dim=in_dim,
            hidden_dim=model_cfg.get("hidden_size", 256),
            p=p,
            device=config.get("device", "cuda"),
        )

        self.n = n
        self.p = p
        self.qaoa = build_qaoa_qnode(n=n, p=p, dev_name="lightning.gpu")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Model inference."""
        data_inputs = batch["x"].to(self.hypernet.device)
        Hcs = batch["Hc"]

        gammas, betas = self.hypernet(data_inputs)

        energies = []
        for b in range(data_inputs.shape[0]):
            E = self.qaoa(gammas[b], betas[b], Hcs[b])
            energies.append(E)

        expvals = torch.stack(energies)
        return {"pred": expvals}

    def loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss (energy maximization = minimize the negative)."""
        return -outputs["pred"].mean()
