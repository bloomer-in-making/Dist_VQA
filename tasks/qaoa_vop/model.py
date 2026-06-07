from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import pennylane as qml

from .qvop_init import QVoPInit


def build_qaoa_qnode(n: int, p: int, dev_name: str = "lightning.gpu", mixer_type: str = "x"):
    dev = qml.device(dev_name, wires=n)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def qaoa_energy(theta: torch.Tensor, Hc: qml.Hamiltonian):
        """theta: [2p] = [gammas | betas]"""
        gammas = theta[:p]
        betas  = theta[p:]

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


def qaoa_ham(cost_params, n_wires: int) -> qml.Hamiltonian:
    """Build a Z_i Z_j cost Hamiltonian from an edge list with weights."""
    obs = []
    coeffs = []
    for (i, j, w_ij) in cost_params:
        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        coeffs.append(w_ij)
    return qml.Hamiltonian(coeffs, obs)


def kl_diag_gauss(mu_q, log_sig_q, mu_p, log_sig_p):
    D = mu_q.shape[-1]
    var_q = torch.exp(2 * log_sig_q)
    var_p = torch.exp(2 * log_sig_p)

    term = (
        var_q / var_p
        + (mu_p - mu_q) ** 2 / var_p
        - 1.0
        + (2 * log_sig_p - 2 * log_sig_q)
    )
    return 0.5 * term.sum(dim=-1).mean() / D


def batch_prototype_prior(phi, mu, log_sig, K=5, prior_sigma0=0.1, detach_proto=True):
    """KNN-based prototype prior."""
    B = phi.size(0)
    if B == 1:
        mu_p = torch.zeros_like(mu)
        log_sig_p = torch.zeros_like(log_sig)
        return (
            mu_p.detach() if detach_proto else mu_p,
            log_sig_p.detach() if detach_proto else log_sig_p,
        )

    with torch.no_grad():
        dist = torch.cdist(phi, phi, p=2)
        dist = dist + torch.eye(B, device=phi.device, dtype=phi.dtype) * 1e6
        K_eff = min(K, B - 1)
        idx = dist.topk(k=K_eff, largest=False).indices

    mu_knn = mu[idx]
    var_knn = torch.exp(2 * log_sig[idx])

    if detach_proto:
        mu_knn = mu_knn.detach()
        var_knn = var_knn.detach()

    mu_p = mu_knn.mean(dim=1)
    var_p0 = var_knn.mean(dim=1)

    var_p = var_p0 + prior_sigma0 ** 2
    log_p = 0.5 * torch.log(var_p + 1e-12)
    return mu_p, log_p


class QAOAVoPModel(nn.Module):
    """
    VoP hypernet for QAOA:
      graph descriptor phi (adjacency vector) -> (mu(phi), log sigma(phi)) -> sample theta
    The dimension of theta equals the QAOA parameters 2p (gammas, betas).
    """

    def __init__(self, config):
        super().__init__()
        n = config.get("n", 10)
        p = config.get("p", 4)

        in_dim = n * (n - 1) // 2

        self.hypernet = QVoPInit(
            in_dim=in_dim,
            hidden_size=config.get("hidden_size", 256),
            p=p,
            device=config.get("device", "cuda"),
        )

        self.n = n
        self.p = p
        self.qaoa = build_qaoa_qnode(n=n, p=p, dev_name="lightning.gpu")

    def forward(self, batch: Dict[str, Any], sample: bool = True) -> Dict[str, Any]:
        """
        batch["x"]: [B, in_dim]  (adjacency vector)
        batch["Hc"]: list of Hamiltonians (length B)
        """
        phi = batch["x"]
        Hcs = batch["Hc"]

        theta, (mu, log_sigma) = self.hypernet(phi, sample=sample)

        energies = []
        for b in range(phi.shape[0]):
            E = self.qaoa(theta[b], Hcs[b])
            energies.append(E)

        expvals = torch.stack(energies)

        return {
            "expval": expvals,
            "theta": theta,
            "mu": mu,
            "log_sigma": log_sigma,
        }

    def loss(self, outputs, batch, beta=0.05, use_proto_prior=True, K=5, prior_sigma0=0.1):
        pred = outputs["expval"]
        loss_energy = pred.mean()

        mu = outputs["mu"]
        log_sigma = outputs["log_sigma"]

        if use_proto_prior:
            phi = batch["x"].to(mu.device, dtype=mu.dtype)
            mu_p, log_sig_p = batch_prototype_prior(
                phi=phi,
                mu=mu,
                log_sig=log_sigma,
                K=K,
                prior_sigma0=prior_sigma0,
                detach_proto=True,
            )
        else:
            mu_p      = torch.zeros_like(mu)
            log_sig_p = torch.zeros_like(log_sigma)

        kl = kl_diag_gauss(
            mu_q=mu,
            log_sig_q=log_sigma,
            mu_p=mu_p,
            log_sig_p=log_sig_p,
        )

        loss = -loss_energy + beta * kl
        return loss, loss_energy, kl

    def loss_energy_only(self, outputs, batch):
        """For validation: energy only, without KL."""
        return outputs["expval"].mean()
