import torch
import torch.nn as nn
import math


class QVoPInit(nn.Module):
    """
    VoP hypernet for QAOA:
      graph descriptor phi (e.g. adjacency vector) -> (mu(phi), log sigma(phi)) -> sample theta
    The dimension of theta equals the QAOA parameters 2p (gammas, betas).
    """

    def __init__(
        self,
        in_dim: int,
        p: int,
        hidden_size: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.p = p
        self.output_dim = 2 * p
        self.hidden_size = hidden_size
        self.device = torch.device(device)

        self.model_float = torch.get_default_dtype()

        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_size, dtype=self.model_float),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, dtype=self.model_float),
            nn.ReLU(),
        )

        self.head = nn.Linear(self.hidden_size, 2 * self.output_dim, dtype=self.model_float)

        self.to(self.device)

    def _encode_mu_and_logsigma(self, phi_batch):
        """Shared encoder computing phi -> (mu, log sigma)."""
        phi_batch = phi_batch.to(dtype=self.model_float, device=self.head.weight.device)

        h = self.backbone(phi_batch)
        mu_logsig = self.head(h)
        mu, s_raw = mu_logsig.chunk(2, dim=-1)

        sigma_min = 1e-6
        sigma = torch.nn.functional.softplus(s_raw) + sigma_min
        log_sigma = torch.log(sigma + 1e-12)
        return mu, log_sigma

    def forward(self, phi_batch: torch.Tensor, sample: bool = True):
        """
        phi_batch: [B, in_dim]
        returns:
          theta: [B, 2p] (sampled QAOA parameters)
          (mu, log_sigma): [B, 2p], [B, 2p]
        """
        mu, log_sigma = self._encode_mu_and_logsigma(phi_batch)

        if sample:
            eps = torch.randn_like(mu) / math.sqrt(mu.shape[-1])
            theta = mu + torch.exp(log_sigma) * eps
        else:
            theta = mu

        return theta, (mu, log_sigma)

    def net(self, phi_batch):
        """
        For adaptation: extracts only phi -> mu.
        Used as the theta0 initial value in AdaptQAOAVoPRunner.
        """
        mu, _ = self._encode_mu_and_logsigma(phi_batch)
        return mu
