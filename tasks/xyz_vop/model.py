from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from .qvop_init import QVoPInit


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


def kl_diag_gauss(mu_q, log_sig_q, mu_p, log_sig_p):
    """
    KL( N(mu_q, diag(sig_q^2)) || N(mu_p, diag(sig_p^2)) )
    returns: sum over parameter dims then batch mean (scalar)
    """
    D = mu_q.shape[-1]
    var_q = torch.exp(2 * log_sig_q)
    var_p = torch.exp(2 * log_sig_p)

    term = var_q / var_p \
         + (mu_p - mu_q) ** 2 / var_p \
         - 1.0 \
         + (2 * log_sig_p - 2 * log_sig_q)

    return 0.5 * term.sum(dim=-1).mean() / D


def batch_prototype_prior(phi, mu, log_sig, K=5, prior_sigma0=0.1, detach_proto=True):
    """KNN prototype prior in theta-space (mu)."""
    B = mu.size(0)

    if B == 1:
        mu_p = torch.zeros_like(mu)
        log_sig_p = torch.zeros_like(log_sig)
        return (
            mu_p.detach() if detach_proto else mu_p,
            log_sig_p.detach() if detach_proto else log_sig_p,
        )

    with torch.no_grad():
        dist = torch.cdist(mu, mu, p=2)
        dist = dist + torch.eye(B, device=mu.device) * 1e6
        K_eff = min(K, B - 1)
        idx = dist.topk(k=K_eff, largest=False).indices

    mu_knn  = mu[idx]
    var_knn = torch.exp(2 * log_sig[idx])

    if detach_proto:
        mu_knn  = mu_knn.detach()
        var_knn = var_knn.detach()

    mu_p   = mu_knn.mean(dim=1)
    var_p0 = var_knn.mean(dim=1)

    var_p  = var_p0 + prior_sigma0 ** 2
    log_p  = 0.5 * torch.log(var_p + 1e-12)

    return mu_p, log_p


class XYZVoPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_wires   = config.get("qubit", 20)
        layerpb     = config.get("layerpb", 3)
        hidden_size = config.get("hidden_size", 256)

        dev = qml.device("lightning.gpu", wires=num_wires, shots=None)

        self.hypernet = QVoPInit(
            input_size=3,
            hidden_size=hidden_size,
            num_wires=num_wires,
            layerpb=layerpb,
            dev=dev,
        )

    def forward(self, batch, sample: bool = True):
        phi_batch = batch["x"]
        expvals, theta, theta_eff, (mu, log_sigma) = self.hypernet(phi_batch, sample=sample)
        return {
            "expval": expvals,
            "theta": theta,
            "theta_eff": theta_eff,
            "mu": mu,
            "log_sigma": log_sigma,
        }

    def loss(self, outputs, batch, beta=0.05, use_proto_prior=True, K=5, prior_sigma0=0.1):
        """
        VoP-style loss:
          L = E[expval]  +  beta * KL(q(theta|x) || p_prototype)
        """
        pred = outputs["expval"]
        loss_energy = pred.mean()

        mu        = outputs["mu"]
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

        loss = loss_energy + beta * kl
        return loss, loss_energy, kl

    def loss_energy_only(self, outputs, batch):
        """For validation: return pure energy loss only."""
        pred = outputs["expval"]
        return pred.mean()
