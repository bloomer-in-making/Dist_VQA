from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from ..common import BaseModel


def create_neel_st(num_qubits: int) -> np.ndarray:
    """Initialize the Neel state."""
    assert num_qubits % 2 == 0
    st_01 = np.array([0.0, 1.0, 0.0, 0.0])
    st_10 = np.array([0.0, 0.0, 1.0, 0.0])
    st_01_N_2 = np.array([1.0])
    st_10_N_2 = np.array([1.0])

    for i in range(num_qubits // 2):
        st_01_N_2 = np.kron(st_01_N_2, st_01)
        st_10_N_2 = np.kron(st_10_N_2, st_10)

    return (st_01_N_2 + st_10_N_2) / math.sqrt(2)


def xyz_ham(delta: torch.Tensor, n_wires: int) -> qml.Hamiltonian:
    """Build the XYZ Hamiltonian."""
    obs = []
    coeffs = []
    for w in range(n_wires):
        obs.append(qml.PauliX(w) @ qml.PauliX((w + 1) % n_wires))
        coeffs.append(delta[0])
    for w in range(n_wires):
        obs.append(qml.PauliY(w) @ qml.PauliY((w + 1) % n_wires))
        coeffs.append(delta[1])
    for w in range(n_wires):
        obs.append(qml.PauliZ(w) @ qml.PauliZ((w + 1) % n_wires))
        coeffs.append(delta[2])

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


class HyperNet(nn.Module):
    """HyperNetwork: generate theta parameters from Jxyz and run the quantum circuit."""

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 256,
        num_wire: int = 10,
        gates_per_block: int = 3,
        device: str = "cpu",
        shots: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_wire = num_wire
        self.gates_per_block = gates_per_block
        self.output_dim = num_wire * gates_per_block
        self.device_str = device
        self.device = torch.device(device)
        self.ini_st = create_neel_st(self.num_wire)

        self.net = MLP(self.input_size, self.hidden_size, self.output_dim)

        self.dev_analytic = qml.device("lightning.gpu", wires=self.num_wire)
        self.qnode_analytic = qml.QNode(
            self._circuit, self.dev_analytic, interface="torch", diff_method="adjoint"
        )

        self._shot_qnodes: Dict[int, qml.QNode] = {}

        if isinstance(shots, int) and shots > 0:
            self._get_shot_qnode(shots)

        self.to(self.device)

        for name, param in self.named_parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device)

    def _circuit(
        self,
        weight: torch.Tensor,
        ham: qml.Hamiltonian,
        ini_st: np.ndarray,
        n_wires: int,
        gates_per_block: int,
    ) -> torch.Tensor:
        """Quantum circuit body."""
        qml.QubitStateVector(ini_st, wires=range(n_wires))
        for k in range(n_wires):
            for w in range(n_wires):
                qml.IsingXX(
                    2 * weight[gates_per_block * k + 0], wires=[w, (w + 1) % n_wires]
                )
            for w in range(n_wires):
                qml.IsingYY(
                    2 * weight[gates_per_block * k + 1], wires=[w, (w + 1) % n_wires]
                )
            for w in range(n_wires):
                qml.IsingZZ(
                    2 * weight[gates_per_block * k + 2], wires=[w, (w + 1) % n_wires]
                )
        return qml.expval(ham)

    def _get_shot_qnode(self, shots: int) -> qml.QNode:
        """Create/return a per-shots QNode cache."""
        if shots not in self._shot_qnodes:
            dev_s = qml.device("lightning.gpu", wires=self.num_wire, shots=shots)
            qnode_s = qml.QNode(
                self._circuit, dev_s, interface="torch", diff_method="parameter-shift"
            )
            self._shot_qnodes[shots] = qnode_s
        return self._shot_qnodes[shots]

    def forward(self, in_val: torch.Tensor) -> torch.Tensor:
        """
        in_val: [B, 3] = [Jx, Jy, Jz]
        returns: expvals [B]
        """
        B = in_val.shape[0]
        theta = self.net(in_val)

        outs = []
        for b in range(B):
            ham = xyz_ham(in_val[b], self.num_wire)
            val = self.qnode_analytic(
                theta[b], ham, self.ini_st, self.num_wire, self.gates_per_block
            )
            outs.append(val)

        return torch.stack(outs)


class XYZModel(BaseModel):
    """Model wrapper for the XYZ task."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.hypernet = HyperNet(
            input_size=3,
            hidden_size=model_cfg.get("hidden_size", 256),
            num_wire=config.get("qubit", 20),
            gates_per_block=config.get("gpb", 3),
            device=config.get("device", "cuda"),
            shots=None,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Model inference."""
        data_inputs = batch["x"]

        if not isinstance(data_inputs, torch.Tensor):
            data_inputs = torch.tensor(data_inputs)

        pred = self.hypernet(data_inputs)
        return {"pred": pred}

    def loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss (energy minimization)."""
        return outputs["pred"].mean()
