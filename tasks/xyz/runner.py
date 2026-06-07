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
from .model import XYZModel, xyz_ham


class Phase1Runner(BaseRunner):
    """Phase 1: HyperNet pretraining."""

    def __init__(self, model: XYZModel, dataset: XYZDataset, config: Dict[str, Any]) -> None:
        super().__init__(model, dataset)
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model.hypernet.to(self.device)

        seed = config.get("seed", 2025)
        torch.manual_seed(seed)
        train_dataset, val_dataset = random_split(
            dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(seed)
        )

        batch_size = config.get("batch_size", 16)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )

        lr = float(config.get("learning_rate", 1e-3))
        self.optimizer = torch.optim.Adam(
            self.model.hypernet.parameters(), lr=lr, weight_decay=1e-4
        )

        self.criterion = torch.nn.L1Loss()
        self.num_epoch = config.get("epoch", 30)

        self.model_path = self._build_model_path()
        os.makedirs(self.model_path, exist_ok=True)

        self.best_val = float("inf")
        self.best_path = os.path.join(self.model_path, "best.pt")

    def _build_model_path(self) -> str:
        """Build the model save path."""
        path = os.getcwd()
        return os.path.join(
            path,
            "results", "xyz_continuous_grad",
            f"qubit{self.config.get('qubit', 20)}",
            "MLP",
            f"seed{self.config.get('seed', 2025)}",
        )

    def train_epoch(self, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Train one epoch."""
        self.model.hypernet.train()
        pred_log = []
        perf_log = []

        with tqdm(self.train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for batch in tepoch:
                data_inputs = batch["x"].to(self.device)
                data_labels = batch["y"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model.forward({"x": data_inputs})
                pred = outputs["pred"]
                pred_log.extend(pred.cpu().detach())

                loss = self.model.loss(outputs, batch)
                loss.backward()
                self.optimizer.step()

                performance = self.criterion(pred, data_labels)
                temp = torch.abs(data_labels - pred).log10()
                temp = temp[~torch.isinf(temp)]
                perf_log.extend(temp)

                tepoch.set_postfix(loss=loss.item(), performance=performance.item())

        return torch.tensor(pred_log), torch.tensor(perf_log)

    def validate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Validation."""
        self.model.hypernet.eval()
        pred_log = []
        perf_log = []

        with tqdm(self.val_loader, unit="batch") as tepoch:
            for batch in tepoch:
                data_inputs = batch["x"].to(self.device)
                data_labels = batch["y"].to(dtype=torch.float64)

                with torch.no_grad():
                    pred = self.model.hypernet(data_inputs)

                pred_log.extend(pred.cpu().detach())
                loss = pred.mean()
                performance = self.criterion(pred, data_labels)
                temp = torch.abs(data_labels - pred).log10()
                temp = temp[~torch.isinf(temp)]
                perf_log.extend(temp)

                tepoch.set_postfix(loss=loss.item(), performance=performance.item())

        loss_mean = torch.tensor(pred_log).mean()
        perf_mean = torch.tensor(perf_log).mean()
        return loss_mean, perf_mean

    def run(self) -> None:
        """Run Phase 1."""
        t_loss_epoch_list = []
        t_perf_epoch_list = []
        v_loss_epoch_list = []
        v_perf_epoch_list = []

        for epoch in range(self.num_epoch):
            print("train")
            loss, performance = self.train_epoch(epoch)

            epoch_loss = loss.mean()
            epoch_perf = performance.mean()
            t_loss_epoch_list.append(epoch_loss.item())
            t_perf_epoch_list.append(epoch_perf.item())

            print(
                f"Hamiltonian = {epoch_loss.item()}, "
                f"log10(|pred - y|) = {epoch_perf.item()}"
            )

            print("valid")
            loss, performance = self.validate()
            print(f"valid loss = {loss}")
            v_loss_epoch_list.append(loss)
            v_perf_epoch_list.append(performance)

            torch.save(
                self.model.hypernet.state_dict(),
                os.path.join(self.model_path, f"{epoch}.pt"),
            )

            if loss < self.best_val:
                self.best_val = loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.hypernet.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "val_loss": loss,
                        "train_loss": epoch_loss,
                        "rng_state": torch.get_rng_state(),
                    },
                    self.best_path,
                )
                print(f"[BEST] epoch={epoch}, val_loss={loss:.6f} -> saved to {self.best_path}")

        np.savetxt(
            os.path.join(self.model_path, "train_epoch_perf.csv"),
            np.array(t_perf_epoch_list),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.model_path, "train_epoch_loss.csv"),
            np.array(t_loss_epoch_list),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.model_path, "val_epoch_perf.csv"),
            np.array(v_perf_epoch_list),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.model_path, "val_epoch_loss.csv"),
            np.array(v_loss_epoch_list),
            delimiter=",",
        )

        print("Save files!")


class Phase2Runner(BaseRunner):
    """Phase 2: per-point fine-tuning from a Phase-1 checkpoint."""

    def __init__(self, model: XYZModel, dataset: XYZDatasetTest, config: Dict[str, Any]) -> None:
        super().__init__(model, dataset)
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model.hypernet.to(self.device)
        self.model.hypernet.eval()

        ckpt_path = config.get("checkpoint_path")
        if not ckpt_path:
            raise ValueError("checkpoint_path is required to run Phase 2.")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.hypernet.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']}")

        seed = config.get("seed", 2025)
        torch.manual_seed(seed)
        num_samples = config.get("num", 16)
        subset, _ = random_split(
            dataset, [num_samples, len(dataset) - num_samples],
            generator=torch.Generator().manual_seed(seed)
        )
        self.dataloader = DataLoader(subset, shuffle=False)

        model_path = os.path.dirname(ckpt_path)
        subdir = config.get("output_dir", "finetune_grad_check")
        self.out_dir = os.path.join(model_path, subdir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(self.out_dir, "train_log.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "idx", "Jx", "Jy", "Jz", "label_gs", "final_E",
                "final_rel_err", "final_log10_abs_err"
            ])

        self.step_per_point = config.get("step_per_point", 2000)
        self.shots = config.get("shots", 2048)

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

        ham = xyz_ham(Jxyz, self.model.hypernet.num_wire)
        optimizer = torch.optim.Adam([phi], lr=lr)

        losses = []
        relerrs = []

        pbar = tqdm(range(steps), desc="adapt")
        for _ in pbar:
            optimizer.zero_grad()
            expval = self.model.hypernet.qnode_analytic(
                phi, ham, self.model.hypernet.ini_st,
                self.model.hypernet.num_wire, self.model.hypernet.gates_per_block
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
        """Run Phase 2."""
        ini_st = self.model.hypernet.ini_st

        for idx, batch in enumerate(self.dataloader):
            x = batch["x"][0].to(self.device)
            y = batch["y"][0].to(self.device)

            if torch.all(x == 0):
                print(f"[{idx}] INVALID x (all zeros) -> skip")
                continue

            print(f"\n=== adapt idx={idx}  x={x.cpu().numpy()},  y={float(y.item())} ===")

            phi, loss_traj, relerr_traj = self.adapt_one_point(
                Jxyz=x,
                y_gs=y,
                lr=float(self.config.get("learning_rate", 1e-3)),
                steps=self.step_per_point,
            )

            with torch.no_grad():
                ham = xyz_ham(x, self.model.hypernet.num_wire)
                qnode = self.model.hypernet._get_shot_qnode(self.shots)
                E = qnode(phi, ham, ini_st, self.model.hypernet.num_wire, self.model.hypernet.gates_per_block)
                eps = 1e-12
                final_rel_err = (E - y) / (y + eps)
                log10_abs_err = torch.log10(torch.clamp(torch.abs(E - y), min=1e-12))

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, float(x[0]), float(x[1]), float(x[2]),
                    float(y.item()), float(E.item()), float(final_rel_err.item()),
                    float(log10_abs_err.item())
                ])

            torch.save(phi.cpu(), os.path.join(self.out_dir, f"phi_{idx}.pt"))
            np.savetxt(os.path.join(self.out_dir, f"loss_traj_{idx}.csv"), np.array(loss_traj), delimiter=",")
            np.savetxt(os.path.join(self.out_dir, f"relerr_traj_{idx}.csv"), np.array(relerr_traj), delimiter=",")

        print("Done.")


class RandInitRunner(BaseRunner):
    """Random-init fine-tuning: start from random parameters instead of the HyperNet."""

    def __init__(self, model: XYZModel, dataset: XYZDatasetTest, config: Dict[str, Any]) -> None:
        super().__init__(model, dataset)
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model.hypernet.to(self.device)
        self.model.hypernet.eval()

        ckpt_path = config.get("checkpoint_path", None)
        if ckpt_path:
            print(f"[RandInitRunner] checkpoint_path={ckpt_path} is ignored (random-init mode).")

        seed = config.get("seed", 2025)
        torch.manual_seed(seed)
        num_samples = config.get("num", 16)
        subset, _ = random_split(
            dataset,
            [num_samples, len(dataset) - num_samples],
            generator=torch.Generator().manual_seed(seed),
        )
        self.dataloader = DataLoader(subset, shuffle=False)

        base_dir = os.path.join(os.getcwd(), "results", "xyz_continuous", f"qubit{self.config.get('qubit', 20)}")
        subdir = config.get("output_dir", "randinit_grad_check")
        self.out_dir = os.path.join(base_dir, subdir, f"seed{seed}")
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(self.out_dir, "train_log.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "idx", "Jx", "Jy", "Jz", "label_gs", "final_E",
                "final_rel_err", "final_log10_abs_err"
            ])

        self.step_per_point = config.get("step_per_point", 2000)
        self.shots = config.get("shots", 2048)

    def adapt_one_point(
        self,
        Jxyz: torch.Tensor,
        y_gs: torch.Tensor,
        lr: float,
        steps: int,
    ) -> tuple[torch.Tensor, list[float], list[float]]:
        """Adapt to a single data point (initial phi generated randomly)."""
        Jxyz = Jxyz.to(self.device, dtype=torch.float32)
        y_gs = y_gs.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            theta_shape = self.model.hypernet.net(Jxyz.unsqueeze(0)).squeeze(0).shape
            theta0 = torch.randn(theta_shape, device=self.device, dtype=torch.float32)

        phi = theta0.clone().detach().requires_grad_(True)

        ham = xyz_ham(Jxyz, self.model.hypernet.num_wire)
        optimizer = torch.optim.Adam([phi], lr=lr)

        losses = []
        relerrs = []

        pbar = tqdm(range(steps), desc="adapt_rand")
        for _ in pbar:
            optimizer.zero_grad()
            expval = self.model.hypernet.qnode_analytic(
                phi, ham, self.model.hypernet.ini_st,
                self.model.hypernet.num_wire, self.model.hypernet.gates_per_block
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
        """Run Phase 2 based on random initialization."""
        ini_st = self.model.hypernet.ini_st

        for idx, batch in enumerate(self.dataloader):
            x = batch["x"][0].to(self.device)
            y = batch["y"][0].to(self.device)

            if torch.all(x == 0):
                print(f"[{idx}] INVALID x (all zeros) -> skip")
                continue

            print(f"\n=== [RandInit] adapt idx={idx}  x={x.cpu().numpy()},  y={float(y.item())} ===")

            phi, loss_traj, relerr_traj = self.adapt_one_point(
                Jxyz=x,
                y_gs=y,
                lr=float(self.config.get("learning_rate", 1e-3)),
                steps=self.step_per_point,
            )

            with torch.no_grad():
                ham = xyz_ham(x, self.model.hypernet.num_wire)
                qnode = self.model.hypernet._get_shot_qnode(self.shots)
                E = qnode(phi, ham, ini_st, self.model.hypernet.num_wire, self.model.hypernet.gates_per_block)
                eps = 1e-12
                final_rel_err = (E - y) / (y + eps)
                log10_abs_err = torch.log10(torch.clamp(torch.abs(E - y), min=1e-12))

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, float(x[0]), float(x[1]), float(x[2]),
                    float(y.item()), float(E.item()), float(final_rel_err.item()),
                    float(log10_abs_err.item())
                ])

            torch.save(phi.cpu(), os.path.join(self.out_dir, f"phi_{idx}.pt"))
            np.savetxt(os.path.join(self.out_dir, f"loss_traj_{idx}.csv"), np.array(loss_traj), delimiter=",")
            np.savetxt(os.path.join(self.out_dir, f"relerr_traj_{idx}.csv"), np.array(relerr_traj), delimiter=",")

        print("Done [RandInit].")


def build_runner(config: Dict[str, Any]) -> BaseRunner:
    """Return the appropriate runner based on the phase in the config."""
    model = XYZModel(config)

    phase = config.get("phase", "phase1")
    if phase == "phase1":
        dataset = XYZDataset(config)
        return Phase1Runner(model, dataset, config)
    elif phase == "phase2":
        dataset = XYZDatasetTest(config)
        return Phase2Runner(model, dataset, config)
    elif phase == "randinit":
        dataset = XYZDatasetTest(config)
        return RandInitRunner(model, dataset, config)
    else:
        raise ValueError(f"Unknown phase: {phase}")
