from __future__ import annotations

import csv
import os
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..common import BaseRunner
from .dataset import XYZDataset, XYZDatasetTest, custom_collate_fn
from .model import XYZVoPModel, xyz_ham


class PretrainVoPRunner(BaseRunner):
    """VoP HyperNet pretraining (energy + prototype-prior KL)."""

    def __init__(self, model: XYZVoPModel, dataset: XYZDataset, config: Dict[str, Any]) -> None:
        super().__init__(model, dataset)
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.qubit = config.get("qubit", 20)
        self.beta = self.config.get("beta", 0.05)
        self.K = self.config.get("K_proto", 5)
        self.prior_sigma0 = self.config.get("prior_sigma0", 0.1)
        self.use_proto_prior = self.config.get("use_proto_prior", True)
        self.model.to(self.device)

        seed = config.get("seed", 2025)
        torch.manual_seed(seed)
        train_ratio = config.get("train_ratio", 0.9)
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        batch_size = config.get("batch_size", 16)
        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )

        lr = float(config.get("learning_rate", 1e-3))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.num_epoch = config.get("epoch", 30)

        self.out_dir = config.get(
            "out_dir",
            f"./results/xyz_vop_grad/n{self.qubit}/beta_{self.beta}_K_{self.K}/seed{seed}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.best_loss = float("inf")
        self.best_path = os.path.join(self.out_dir, "best.pt")

        self.train_loss_epoch = []
        self.train_perf_epoch = []
        self.val_loss_epoch = []
        self.val_perf_epoch = []

        self.criterion = torch.nn.L1Loss()

    def _extract_pred(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor | None:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if "expval" in outputs:
            return outputs["expval"]
        if "pred" in outputs:
            return outputs["pred"]
        return None

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        loss_list = []
        perf_list = []

        for batch in tqdm(self.train_loader, desc=f"[VoP-Pretrain] Epoch {epoch}"):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(batch)

            loss, loss_energy, kl = self.model.loss(
                outputs, batch,
                beta=self.beta,
                use_proto_prior=self.use_proto_prior,
                K=self.K,
                prior_sigma0=self.prior_sigma0,
            )
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss_energy.item())

            pred = self._extract_pred(outputs)
            if pred is not None:
                y = batch["y"].to(pred.device).to(pred.dtype)
                tmp = torch.abs(y - pred).log10()
                tmp = tmp[~torch.isinf(tmp)]
                if tmp.numel() > 0:
                    perf_list.append(tmp.mean().item())

        mean_loss = float(np.mean(loss_list)) if loss_list else float("nan")
        mean_perf = float(np.mean(perf_list)) if perf_list else float("nan")
        return mean_loss, mean_perf

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        loss_list = []
        perf_list = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="[XYZ-VoP] Valid"):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                outputs = self.model.forward(batch, sample=False)
                loss_energy = self.model.loss_energy_only(outputs, batch)
                loss_list.append(loss_energy.item())

                pred = self._extract_pred(outputs)
                if pred is not None:
                    y = batch["y"].to(pred.device).to(pred.dtype)
                    tmp = torch.abs(y - pred).log10()
                    tmp = tmp[~torch.isinf(tmp)]
                    if tmp.numel() > 0:
                        perf_list.append(tmp.mean().item())

        return float(np.mean(loss_list)), float(np.mean(perf_list))

    def run(self) -> None:
        hypernet = self.model.hypernet

        for epoch in range(1, self.num_epoch + 1):
            if epoch == 1:
                print(f"[VoP-Pretrain], beta={self.beta}")

            print("train")
            train_loss, train_perf = self.train_epoch(epoch)
            self.train_loss_epoch.append(train_loss)
            self.train_perf_epoch.append(train_perf)
            print(f"Hamiltonian(train) = {train_loss}, log10(|pred - y|) = {train_perf}")

            print("valid")
            val_loss, val_perf = self.validate()
            self.val_loss_epoch.append(val_loss)
            self.val_perf_epoch.append(val_perf)
            print(f"valid loss = {val_loss}, valid log10(|pred - y|) = {val_perf}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": hypernet.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "rng_state": torch.get_rng_state(),
                    },
                    self.best_path,
                )
                print(f"[VoP BEST] epoch={epoch}, val_loss={val_loss:.6f} -> saved to {self.best_path}")

        np.savetxt(os.path.join(self.out_dir, "train_epoch_loss.csv"), np.array(self.train_loss_epoch), delimiter=",")
        np.savetxt(os.path.join(self.out_dir, "train_epoch_perf.csv"), np.array(self.train_perf_epoch), delimiter=",")
        np.savetxt(os.path.join(self.out_dir, "val_epoch_loss.csv"), np.array(self.val_loss_epoch), delimiter=",")
        np.savetxt(os.path.join(self.out_dir, "val_epoch_perf.csv"), np.array(self.val_perf_epoch), delimiter=",")

        print("[VoP-Pretrain] Save files!")


class AdaptVoPRunner(BaseRunner):
    """Per-task adaptation based on the VoP HyperNet."""

    def __init__(self, model: XYZVoPModel, dataset: XYZDatasetTest, config: Dict[str, Any]) -> None:
        super().__init__(model, dataset)
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model.to(self.device)

        ckpt_path = config.get("checkpoint_path")
        if ckpt_path is None:
            raise ValueError("config['checkpoint_path'] is required for VoP adaptation.")
        ckpt_path = os.path.expanduser(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.hypernet.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[VoP-Adapt] Loaded checkpoint: {ckpt_path} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')})")

        self.model.eval()

        seed = config.get("seed", 2025)
        torch.manual_seed(seed)

        num_tasks = config.get("num", 16)
        subset, _ = random_split(
            dataset,
            [num_tasks, len(dataset) - num_tasks],
            generator=torch.Generator().manual_seed(seed),
        )
        self.loader = DataLoader(
            subset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn
        )

        model_path = os.path.dirname(ckpt_path)
        subdir = config.get("output_dir", "pergraph_runs")
        self.out_dir = os.path.join(model_path, subdir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.qubit = config.get("qubit", 20)
        self.lpb = config.get("layerpb", 3)
        self.lr = float(config.get("learning_rate", 1e-3))
        self.step_per_point = config.get("step_per_point", 2000)
        self.shots = config.get("shots", 2048)

        self.csv_path = os.path.join(self.out_dir, "train_log.csv")
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "idx", "Jx", "Jy", "Jz", "label_gs",
                "final_E", "final_rel_err", "final_log10_abs_err",
            ])

    def adapt_one_point(
        self,
        Jxyz: torch.Tensor,
        y_gs: torch.Tensor,
        lr: float,
        steps: int,
    ) -> tuple[torch.Tensor, list[float], list[float]]:
        """Adapt to a single data point."""
        Jxyz = Jxyz.to(self.device, dtype=torch.float32)
        y_gs = y_gs.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            theta0 = self.model.hypernet.net(Jxyz.unsqueeze(0)).squeeze(0)

        phi = theta0.clone().detach().requires_grad_(True)

        ham = xyz_ham(Jxyz, self.model.hypernet.num_wires)
        optimizer = torch.optim.Adam([phi], lr=lr)

        losses = []
        relerrs = []

        pbar = tqdm(range(steps), desc="adapt_vop")
        for _ in pbar:
            optimizer.zero_grad()
            expval = self.model.hypernet.qnode_analytic(
                phi,
                ham,
                self.model.hypernet.ini_st,
                self.model.hypernet.num_wires,
                self.model.hypernet.layerpb,
            )
            loss = expval
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                eps = 1e-12
                rel_err = (expval - y_gs) / (y_gs + eps)
                err_log10 = torch.log10(torch.clamp(torch.abs(expval - y_gs), min=1e-12))

            pbar.set_postfix(
                E=float(expval.item()),
                rel=float(rel_err.item()),
                log10_abs_err=float(err_log10.item()),
            )

            losses.append(float(loss.item()))
            relerrs.append(float(rel_err.item()))

        return phi.detach(), losses, relerrs

    def run(self) -> None:
        """Run Phase 2 based on VoP."""
        ini_st = self.model.hypernet.ini_st

        for idx, batch in enumerate(self.loader):
            x = batch["x"][0].to(self.device)
            y = batch["y"][0].to(self.device)

            if torch.all(x == 0):
                print(f"[{idx}] INVALID x (all zeros) -> skip")
                continue

            print(f"\n=== [VoP adapt] idx={idx}  x={x.cpu().numpy()},  y={float(y.item())} ===")

            phi, loss_traj, relerr_traj = self.adapt_one_point(
                Jxyz=x,
                y_gs=y,
                lr=self.lr,
                steps=self.step_per_point,
            )

            with torch.no_grad():
                ham = xyz_ham(x, self.model.hypernet.num_wires)
                E = self.model.hypernet.qnode(
                    phi, ham, ini_st, self.model.hypernet.num_wires, self.model.hypernet.layerpb
                )
                eps = 1e-12
                final_rel_err = (E - y) / (y + eps)
                log10_abs_err = torch.log10(torch.clamp(torch.abs(E - y), min=1e-12))

            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    idx, float(x[0]), float(x[1]), float(x[2]),
                    float(y.item()), float(E.item()),
                    float(final_rel_err.item()), float(log10_abs_err.item()),
                ])

            torch.save(phi.cpu(), os.path.join(self.out_dir, f"phi_{idx}.pt"))
            np.savetxt(os.path.join(self.out_dir, f"loss_traj_{idx}.csv"), np.array(loss_traj), delimiter=",")
            np.savetxt(os.path.join(self.out_dir, f"relerr_traj_{idx}.csv"), np.array(relerr_traj), delimiter=",")

        print("[VoP-Adapt] Done.")


def build_runner(config: Dict[str, Any]) -> BaseRunner:
    """Return the appropriate runner based on the phase in the config."""
    model = XYZVoPModel(config)

    phase = config.get("phase", "phase1")
    if phase == "phase1":
        dataset = XYZDataset(config)
        return PretrainVoPRunner(model, dataset, config)
    elif phase == "phase2":
        dataset = XYZDatasetTest(config)
        return AdaptVoPRunner(model, dataset, config)
    else:
        raise ValueError(f"Unknown phase: {phase}")
