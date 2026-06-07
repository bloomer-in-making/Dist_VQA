import torch
import torch.nn as nn
import pennylane as qml
import math
import numpy as np
import torch.nn.functional as F


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


def make_xyz_ham(delta, NUM_WIRES):
    obs = []
    coeffs = []
    for w in range(NUM_WIRES):
        obs.append(qml.PauliX(w) @ qml.PauliX((w + 1) % NUM_WIRES))
        coeffs.append(delta[0])
    for w in range(NUM_WIRES):
        obs.append(qml.PauliY(w) @ qml.PauliY((w + 1) % NUM_WIRES))
        coeffs.append(delta[1])
    for w in range(NUM_WIRES):
        obs.append(qml.PauliZ(w) @ qml.PauliZ((w + 1) % NUM_WIRES))
        coeffs.append(delta[2])

    return qml.Hamiltonian(coeffs, obs)


class QVoPInit(nn.Module):
    """
    Hypernet that takes a task (phi: Jx,Jy,Jz) and generates a PQC parameter theta init vector.
    phi -> (mu, log sigma) -> theta = mu + sigma (*) eps  (variational hypernet)
    """
    def __init__(self, input_size=3, hidden_size=256, num_wires=20, layerpb=3, dev=None):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_wires   = num_wires
        self.layerpb     = layerpb
        self.output_dim  = num_wires * layerpb
        self.dev         = dev

        self.model_float = torch.get_default_dtype()

        self.qnode_float = torch.float64
        self.qnode_complex = torch.complex128

        self.backbone = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, dtype=self.model_float),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, dtype=self.model_float),
            nn.ReLU()
        )

        self.head = nn.Linear(self.hidden_size, 2 * self.output_dim, dtype=self.model_float)

        self.qnode = self.circuit_gen()

        ini_np = create_neel_st(self.num_wires)
        ini_t  = torch.tensor(ini_np, dtype=self.qnode_complex)
        self.register_buffer("ini_st_buf", ini_t)

    def circuit_gen(self):
        @qml.qnode(self.dev, diff_method="adjoint", interface="torch")
        def circuit(weight, ham, ini_st, n_wires, LAYERS_PER_BLOCK):
            qml.QubitStateVector(ini_st, wires=range(n_wires))
            for k in range(n_wires):
                for w in range(n_wires):
                    qml.IsingXX(2 * weight[LAYERS_PER_BLOCK * k + 0], wires=[w, (w + 1) % n_wires])
                for w in range(n_wires):
                    qml.IsingYY(2 * weight[LAYERS_PER_BLOCK * k + 1], wires=[w, (w + 1) % n_wires])
                for w in range(n_wires):
                    qml.IsingZZ(2 * weight[LAYERS_PER_BLOCK * k + 2], wires=[w, (w + 1) % n_wires])

            return qml.expval(ham)

        return circuit

    def _encode_mu_and_logsigma(self, phi_batch):
        device = self.head.weight.device
        phi_batch = phi_batch.to(dtype=self.model_float, device=device)

        h = self.backbone(phi_batch)
        mu_logsig = self.head(h)
        mu, s_raw = mu_logsig.chunk(2, dim=-1)

        sigma_min = 1e-6
        sigma = F.softplus(s_raw) + sigma_min
        log_sigma = torch.log(sigma + 1e-12)
        return mu, log_sigma

    @property
    def ini_st(self):
        return self.ini_st_buf

    def forward(self, phi_batch, sample=True):
        mu, log_sigma = self._encode_mu_and_logsigma(phi_batch)

        if sample:
            eps = torch.randn_like(mu) / math.sqrt(mu.shape[-1])
            theta = mu + torch.exp(log_sigma) * eps
        else:
            theta = mu

        B = phi_batch.shape[0]
        expvals = []
        theta_eff_list = []

        ini_st_q = self.ini_st_buf.to(device=theta.device, dtype=self.qnode_complex)

        for b in range(B):
            ham = make_xyz_ham(phi_batch[b].to(self.qnode_float), self.num_wires)

            theta_eff = theta[b].to(dtype=self.model_float)

            theta_q = theta_eff.to(dtype=self.qnode_float, device=phi_batch.device)
            val = self.qnode(theta_q, ham, ini_st_q, self.num_wires, self.layerpb)

            expvals.append(val.to(self.model_float))
            theta_eff_list.append(theta_eff)

        expvals = torch.stack(expvals).to(dtype=self.model_float)
        theta_eff_all = torch.stack(theta_eff_list).to(dtype=self.model_float)

        return expvals, theta, theta_eff_all, (mu, log_sigma)

    def net(self, phi_batch):
        """
        For adaptation: extracts only phi -> mu.
        Used as the theta0 initial value in AdaptVoPRunner.
        """
        mu, _ = self._encode_mu_and_logsigma(phi_batch)
        return mu

    def qnode_analytic(self, weight, ham, ini_st, num_wires, LAYERS_PER_BLOCK):
        """Wrapper to keep the existing HyperNet interface for Phase 2."""
        return self.qnode(weight, ham, ini_st, num_wires, LAYERS_PER_BLOCK)
