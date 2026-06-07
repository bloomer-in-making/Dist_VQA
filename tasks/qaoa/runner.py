from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..common import BaseRunner
from .dataset import QAOADataset, QAOADatasetTest, collate_graph
from .model import QAOAModel, build_qaoa_qnode


class Phase1Runner(BaseRunner):
    """Phase 1: initial training of the QAOA HyperNet."""

    def __init__(self, model: QAOAModel, dataset: QAOADataset, config: Dict[str, Any]) -> None:
        super().__init__(model, dataset)
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model.hypernet = self.model.hypernet.to(self.device)

        seed = config.get("seed", 2025)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_ratio = config.get("train_ratio", 0.8)
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )

        batch_size = config.get("batch_size", 16)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_graph,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_graph,
        )

        lr = float(config.get("learning_rate", 1e-3))
        self.optimizer = torch.optim.Adam(
            self.model.hypernet.parameters(), lr=lr, weight_decay=1e-4
        )

        self.num_epoch = config.get("epoch", 30)

        self.model_path = self._build_model_path()
        os.makedirs(self.model_path, exist_ok=True)

        self.best_valE = -float("inf")
        self.best_path = os.path.join(self.model_path, "best.pt")

        self.train_csv = os.path.join(self.model_path, "train_energy.csv")
        self.train_perf_csv = os.path.join(self.model_path, "train_perf_log10_abs_err.csv")
        self.val_csv = os.path.join(self.model_path, "val_energy.csv")
        self.val_perf_csv = os.path.join(self.model_path, "val_perf_log10_abs_err.csv")

        if not os.path.exists(self.train_csv):
            with open(self.train_csv, "w") as f:
                f.write("epoch,mean_energy\n")
            with open(self.train_perf_csv, "w") as f:
                f.write("epoch,mean_log10_abs_err\n")
            with open(self.val_csv, "w") as f:
                f.write("epoch,mean_energy\n")
            with open(self.val_perf_csv, "w") as f:
                f.write("epoch,mean_log10_abs_err\n")

    def _build_model_path(self) -> str:
        """Build the model save path."""
        path = os.getcwd()
        n = self.config.get("n", 10)
        p = self.config.get("p", 4)
        seed = self.config.get("seed", 2025)
        return os.path.join(
            path,
            "results", "qaoa_grad",
            f"train_hypernet_n{n}_p{p}_MLP",
            f"seed{seed}",
        )

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train one epoch."""
        self.model.hypernet.train()
        train_E_list = []
        train_perf_list = []

        with tqdm(self.train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for batch in tepoch:
                opt_true = batch["opt"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model.forward(batch)
                E_batch = outputs["pred"]

                loss = self.model.loss(outputs, batch)
                loss.backward()
                self.optimizer.step()

                train_E_list.append(E_batch.detach().cpu().numpy())

                err = torch.abs(E_batch - opt_true)
                log10_err = torch.log10(err + 1e-9)
                log10_err = log10_err[~torch.isinf(log10_err)]
                if log10_err.numel() > 0:
                    train_perf_list.append(log10_err.mean().item())

                tepoch.set_postfix(loss=loss.item())

        mean_train_E = np.concatenate(train_E_list).mean()
        mean_train_pf = np.mean(train_perf_list) if train_perf_list else float("nan")
        return mean_train_E, mean_train_pf

    def validate(self) -> tuple[float, float]:
        """Validation."""
        self.model.hypernet.eval()
        val_E_list = []
        val_perf_list = []

        with tqdm(self.val_loader, unit="batch") as tepoch:
            for batch in tepoch:
                opt_true = batch["opt"]

                with torch.no_grad():
                    outputs = self.model.forward(batch)
                    E_batch = outputs["pred"]

                val_E_list.append(E_batch.cpu().numpy())

                err = torch.abs(E_batch.cpu() - opt_true)
                log10_err = torch.log10(err + 1e-9)
                log10_err = log10_err[~torch.isinf(log10_err)]
                if log10_err.numel() > 0:
                    val_perf_list.append(log10_err.mean().item())

                tepoch.set_postfix()

        mean_val_E = np.concatenate(val_E_list).mean()
        mean_val_pf = np.mean(val_perf_list) if val_perf_list else float("nan")
        return mean_val_E, mean_val_pf

    def run(self) -> None:
        """Run Phase 1."""
        for epoch in range(1, self.num_epoch + 1):
            print(f"\n[{epoch:02d}] Train")
            mean_train_E, mean_train_pf = self.train_epoch(epoch)

            print(f"[{epoch:02d}] Valid")
            mean_val_E, mean_val_pf = self.validate()

            with open(self.train_csv, "a") as f:
                f.write(f"{epoch},{mean_train_E:.9f}\n")
            with open(self.train_perf_csv, "a") as f:
                f.write(f"{epoch},{mean_train_pf:.9f}\n")
            with open(self.val_csv, "a") as f:
                f.write(f"{epoch},{mean_val_E:.9f}\n")
            with open(self.val_perf_csv, "a") as f:
                f.write(f"{epoch},{mean_val_pf:.9f}\n")

            print(
                f"[{epoch:02d}] trainE={mean_train_E:.6f}, "
                f"train log10|err|={mean_train_pf:.3f} | "
                f"valE={mean_val_E:.6f}, val log10|err|={mean_val_pf:.3f}"
            )

            if mean_val_E > self.best_valE:
                self.best_valE = mean_val_E
                ckpt = {
                    "state_dict": self.model.hypernet.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "meta": {
                        "in_dim": self.dataset.x_dim,
                        "p": self.config.get("p", 4),
                        "hidden_dim": self.config.get("model", {}).get("hidden_size", 256),
                        "n": self.config.get("n", 10),
                        "epoch": epoch,
                        "best_valE": self.best_valE,
                    },
                }
                torch.save(ckpt, self.best_path)
                print(f"[SAVE][BEST] ep={epoch} valE={self.best_valE:.6f} -> {self.best_path}")

        print("Done.")


class Phase2Runner(BaseRunner):
    """Phase 2: per-graph optimization (fine-tuning)."""

    def __init__(self, model: QAOAModel, dataset: QAOADatasetTest, config: Dict[str, Any]) -> None:
        super().__init__(model, dataset)
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model.hypernet.to(self.device)
        self.model.hypernet.eval()

        ckpt_path = config.get("checkpoint_path")
        if not ckpt_path:
            raise ValueError("checkpoint_path is required to run Phase 2.")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.hypernet.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Loaded checkpoint: epoch={ckpt['meta']['epoch']}, valE={ckpt['meta']['best_valE']}")

        seed = config.get("seed", 2025)
        num_samples = config.get("num", 16)
        if num_samples > len(dataset):
            num_samples = len(dataset)
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
            :num_samples
        ]
        self.subset = torch.utils.data.Subset(dataset, indices.tolist())

        model_path = os.path.dirname(ckpt_path)
        subdir = config.get("output_dir", "pergrap_runs")
        self.out_dir = os.path.join(model_path, subdir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.step_per_point = config.get("step_per_point", 2000)
        self.lr = float(config.get("learning_rate", 1e-2))

        self.summary_path = os.path.join(self.out_dir, "summary.csv")
        with open(self.summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["idx", "opt_energy", "final_energy", "best_energy", "final_rel_err", "final_log10_abs_err"]
            )

    def optimize_one_graph(self, idx: int, item: Dict[str, Any]) -> tuple:
        """Optimize a single graph."""
        Hc = item["Hc"]
        x = item["x"].unsqueeze(0).to(self.device)
        opt_true = float(item["opt"])

        n = int(max(Hc.wires)) + 1
        p = self.config.get("p", 4)
        qaoa = build_qaoa_qnode(n=n, p=p, dev_name="lightning.gpu")

        with torch.no_grad():
            g0, b0 = self.model.hypernet(x)
        gammas = torch.nn.Parameter(g0.squeeze(0).clone().detach())
        betas = torch.nn.Parameter(b0.squeeze(0).clone().detach())

        optimizer = torch.optim.Adam([gammas, betas], lr=self.lr)

        energy_traj = []
        relerr_traj = []
        log10_err_traj = []
        best_energy = -float("inf")

        pbar = tqdm(range(self.step_per_point), desc=f"Graph {idx}")
        for step in pbar:
            optimizer.zero_grad()
            E = qaoa(gammas, betas, Hc)
            loss = -E
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                E_val = float(E.item())
                energy_traj.append(E_val)
                rel_err = (opt_true - E_val) / (abs(opt_true) + 1e-12)
                log10_err = math.log10(abs(E_val - opt_true) + 1e-12)
                relerr_traj.append(rel_err)
                log10_err_traj.append(log10_err)

                if E_val > best_energy:
                    best_energy = E_val

            pbar.set_postfix(E=f"{E_val:.6f}", best=f"{best_energy:.6f}")

        return gammas.detach(), betas.detach(), energy_traj, relerr_traj, log10_err_traj

    def run(self) -> None:
        """Run Phase 2."""
        for idx in range(len(self.subset)):
            item = self.subset[idx]
            opt_true = float(item["opt"])

            print(f"\n=== Graph {idx} / {len(self.subset)}, ground_truth={opt_true} ===")
            gammas, betas, energy_traj, relerr_traj, log10_err_traj = self.optimize_one_graph(idx, item)

            final_energy = energy_traj[-1]
            best_energy = max(energy_traj)
            final_rel_err = relerr_traj[-1]
            final_log10_err = log10_err_traj[-1]

            with open(self.summary_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [idx, opt_true, final_energy, best_energy, final_rel_err, final_log10_err]
                )

            np.savetxt(os.path.join(self.out_dir, f"energy_traj_{idx}.csv"), np.array(energy_traj), delimiter=",")
            np.savetxt(os.path.join(self.out_dir, f"relerr_traj_{idx}.csv"), np.array(relerr_traj), delimiter=",")
            np.savetxt(os.path.join(self.out_dir, f"log10_abs_err_traj_{idx}.csv"), np.array(log10_err_traj), delimiter=",")

        print("Done.")


class RandInitRunner(BaseRunner):
    """Random-init fine-tuning: start from random parameters instead of the HyperNet."""

    def __init__(self, model: QAOAModel, dataset: QAOADatasetTest, config: Dict[str, Any]) -> None:
        super().__init__(model, dataset)
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model.hypernet.to(self.device)
        self.model.hypernet.eval()

        ckpt_path = config.get("checkpoint_path", None)
        if ckpt_path:
            print(f"[RandInitRunner] checkpoint_path={ckpt_path} is ignored (random-init mode).")

        seed = config.get("seed", 2025)
        num_samples = config.get("num", 16)
        if num_samples > len(dataset):
            num_samples = len(dataset)
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
            :num_samples
        ]
        self.subset = torch.utils.data.Subset(dataset, indices.tolist())

        base_dir = os.path.join(
            os.getcwd(), "results", "qaoa",
            f"n{self.config.get('n', 14)}_p{self.config.get('p', 16)}", f"seed{seed}",
        )
        subdir = config.get("output_dir", "pergrap_runs")
        self.out_dir = os.path.join(base_dir, subdir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.step_per_point = config.get("step_per_point", 2000)
        self.lr = float(config.get("learning_rate", 1e-2))

        self.summary_path = os.path.join(self.out_dir, "summary.csv")
        with open(self.summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["idx", "opt_energy", "final_energy", "best_energy", "final_rel_err", "final_log10_abs_err"]
            )

    def optimize_one_graph(self, idx: int, item: Dict[str, Any]) -> tuple:
        """Optimize a single graph (random initial parameters)."""
        Hc = item["Hc"]
        opt_true = float(item["opt"])

        n = int(max(Hc.wires)) + 1
        p = self.config.get("p", 4)
        qaoa = build_qaoa_qnode(n=n, p=p, dev_name="lightning.gpu")

        with torch.no_grad():
            g0 = torch.randn(p, device=self.device, dtype=torch.float32)
            b0 = torch.randn(p, device=self.device, dtype=torch.float32)
        gammas = torch.nn.Parameter(g0.clone().detach())
        betas = torch.nn.Parameter(b0.clone().detach())

        optimizer = torch.optim.Adam([gammas, betas], lr=self.lr)

        energy_traj = []
        relerr_traj = []
        log10_err_traj = []
        best_energy = -float("inf")

        pbar = tqdm(range(self.step_per_point), desc=f"Graph {idx}")
        for step in pbar:
            optimizer.zero_grad()
            E = qaoa(gammas, betas, Hc)
            loss = -E
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                E_val = float(E.item())
                energy_traj.append(E_val)
                rel_err = (opt_true - E_val) / (abs(opt_true) + 1e-12)
                log10_err = math.log10(abs(E_val - opt_true) + 1e-12)
                relerr_traj.append(rel_err)
                log10_err_traj.append(log10_err)

                if E_val > best_energy:
                    best_energy = E_val

            pbar.set_postfix(E=f"{E_val:.6f}", best=f"{best_energy:.6f}")

        return gammas.detach(), betas.detach(), energy_traj, relerr_traj, log10_err_traj

    def run(self) -> None:
        """Run Phase 2 based on random initialization."""
        for idx in range(len(self.subset)):
            item = self.subset[idx]
            opt_true = float(item["opt"])

            print(f"\n=== Graph {idx} / {len(self.subset)}, ground_truth={opt_true} ===")
            gammas, betas, energy_traj, relerr_traj, log10_err_traj = self.optimize_one_graph(idx, item)

            final_energy = energy_traj[-1]
            best_energy = max(energy_traj)
            final_rel_err = relerr_traj[-1]
            final_log10_err = log10_err_traj[-1]

            with open(self.summary_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [idx, opt_true, final_energy, best_energy, final_rel_err, final_log10_err]
                )

            np.savetxt(os.path.join(self.out_dir, f"energy_traj_{idx}.csv"), np.array(energy_traj), delimiter=",")
            np.savetxt(os.path.join(self.out_dir, f"relerr_traj_{idx}.csv"), np.array(relerr_traj), delimiter=",")
            np.savetxt(os.path.join(self.out_dir, f"log10_abs_err_traj_{idx}.csv"), np.array(log10_err_traj), delimiter=",")

        print("Done.")


def build_runner(config: Dict[str, Any]) -> BaseRunner:
    """Return the appropriate runner based on the phase in the config."""
    model = QAOAModel(config)

    phase = config.get("phase", "phase1")
    if phase == "phase1":
        dataset = QAOADataset(config)
        return Phase1Runner(model, dataset, config)
    elif phase == "phase2":
        dataset = QAOADatasetTest(config)
        return Phase2Runner(model, dataset, config)
    elif phase == "randinit":
        dataset = QAOADatasetTest(config)
        return RandInitRunner(model, dataset, config)
    else:
        raise ValueError(f"Unknown phase: {phase}")
