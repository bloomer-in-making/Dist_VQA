from __future__ import annotations

import os
import csv
import math
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..common import BaseRunner
from .dataset import QAOADataset, QAOADatasetTest, collate_graph
from .model import QAOAVoPModel, build_qaoa_qnode


class PretrainQAOAVoPRunner(BaseRunner):
    def __init__(self, model: QAOAVoPModel, dataset: QAOADataset, config: Dict[str, Any]) -> None:
        super().__init__(model, dataset)
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
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
            train_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_graph,
        )
        self.val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_graph,
        )

        lr = float(config.get("learning_rate", 1e-3))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.num_epoch = config.get("epoch", 30)

        self.n = config.get("n", 14)
        self.p = config.get("p", 16)
        self.beta = config.get("beta", 0.05)
        self.K = config.get("K", 5)
        self.prior_sigma0 = config.get("prior_sigma0", 0.1)
        self.use_proto_prior = config.get("use_proto_prior", True)

        self.out_dir = config.get(
            "out_dir",
            f"./results/qaoa_vop_grad/n{self.n}_p{self.p}/beta_{self.beta}_K_{self.K}/seed{seed}",
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.best_loss = -float("inf")
        self.best_path = os.path.join(self.out_dir, "best.pt")

        self.train_loss_epoch = []
        self.train_perf_epoch = []
        self.val_loss_epoch = []
        self.val_perf_epoch = []

        self.criterion = torch.nn.L1Loss()

    def _extract_pred(self, outputs):
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

        for batch in tqdm(self.train_loader, desc=f"[QAOA-VoP] Epoch {epoch}"):
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
                y = batch["opt"].to(pred.device).to(pred.dtype)
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
            for batch in tqdm(self.val_loader, desc="[QAOA-VoP] Valid"):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)

                outputs = self.model.forward(batch, sample=False)
                loss_energy = self.model.loss_energy_only(outputs, batch)
                loss_list.append(loss_energy.item())

                pred = self._extract_pred(outputs)
                if pred is not None:
                    y = batch["opt"].to(pred.device).to(pred.dtype)
                    tmp = torch.abs(y - pred).log10()
                    tmp = tmp[~torch.isinf(tmp)]
                    if tmp.numel() > 0:
                        perf_list.append(tmp.mean().item())

        return float(np.mean(loss_list)), float(np.mean(perf_list))

    def run(self) -> None:
        for epoch in range(1, self.num_epoch + 1):
            train_loss, train_perf = self.train_epoch(epoch)
            self.train_loss_epoch.append(train_loss)
            self.train_perf_epoch.append(train_perf)

            val_loss, val_perf = self.validate()
            self.val_loss_epoch.append(val_loss)
            self.val_perf_epoch.append(val_perf)

            print(f"[QAOA-VoP] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss > self.best_loss:
                self.best_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.hypernet.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "rng_state": torch.get_rng_state(),
                    },
                    self.best_path,
                )
                print(f"[QAOA-VoP BEST] epoch={epoch}, val_loss={val_loss:.6f} -> {self.best_path}")

        np.savetxt(os.path.join(self.out_dir, "train_epoch_loss.csv"), np.array(self.train_loss_epoch), delimiter=",")
        np.savetxt(os.path.join(self.out_dir, "train_epoch_perf.csv"), np.array(self.train_perf_epoch), delimiter=",")
        np.savetxt(os.path.join(self.out_dir, "val_epoch_loss.csv"), np.array(self.val_loss_epoch), delimiter=",")
        np.savetxt(os.path.join(self.out_dir, "val_epoch_perf.csv"), np.array(self.val_perf_epoch), delimiter=",")


class AdaptQAOAVoPRunner(BaseRunner):
    """Per-task adaptation based on the VoP HyperNet."""

    def __init__(self, model: QAOAVoPModel, dataset: QAOADatasetTest, config: Dict[str, Any]) -> None:
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

        num_samples = config.get("num", 16)
        if num_samples > len(dataset):
            num_samples = len(dataset)
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
            :num_samples
        ]
        self.subset = torch.utils.data.Subset(dataset, indices.tolist())

        model_path = os.path.dirname(ckpt_path)
        subdir = config.get("output_dir", "pergraph_runs")
        self.out_dir = os.path.join(model_path, subdir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.n = config.get("n", 14)
        self.p = config.get("p", 16)
        self.step_per_point = config.get("step_per_point", 2000)
        self.lr = float(config.get("learning_rate", 1e-2))

        self.summary_path = os.path.join(self.out_dir, "summary.csv")
        with open(self.summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["idx", "opt_energy", "final_energy", "best_energy", "final_rel_err", "final_log10_abs_err"]
            )

    def adapt_one_point(self, idx: int, item: Dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
        """Optimize a single graph."""
        Hc = item["Hc"]
        x = item["x"].unsqueeze(0).to(self.device)
        opt_true = float(item["opt"])

        n = int(max(Hc.wires)) + 1
        p = self.config.get("p", 4)
        qaoa = build_qaoa_qnode(n=n, p=p, dev_name="lightning.gpu")

        with torch.no_grad():
            theta0 = self.model.hypernet.net(x)[0]
        thetas = torch.nn.Parameter(theta0.clone().detach())

        optimizer = torch.optim.Adam([thetas], lr=self.lr)

        energy_traj = []
        relerr_traj = []
        log10_err_traj = []
        best_energy = -float("inf")

        pbar = tqdm(range(self.step_per_point), desc=f"Graph {idx}")
        for step in pbar:
            optimizer.zero_grad()
            E = qaoa(thetas, Hc)
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

        return energy_traj, relerr_traj, log10_err_traj

    def run(self) -> None:
        """Run Phase 2."""
        for idx in range(len(self.subset)):
            item = self.subset[idx]
            opt_true = float(item["opt"])

            print(f"\n=== Graph {idx} / {len(self.subset)}, ground_truth={opt_true} ===")
            energy_traj, relerr_traj, log10_err_traj = self.adapt_one_point(idx, item)

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
    model = QAOAVoPModel(config)

    phase = config.get("phase", "phase1")
    if phase == "phase1":
        dataset = QAOADataset(config)
        return PretrainQAOAVoPRunner(model, dataset, config)
    elif phase == "phase2":
        dataset = QAOADatasetTest(config)
        return AdaptQAOAVoPRunner(model, dataset, config)
    else:
        raise ValueError(f"Unknown phase: {phase}")
