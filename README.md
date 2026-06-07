# Learning Task-Conditioned Distributional Initializers for Variational Quantum Algorithms

Reference implementation of a two-phase pipeline for variational quantum algorithms (VQAs). In Phase 1, a task-conditioned hypernetwork is meta-trained to predict circuit initializations from task descriptors. In Phase 2, the predicted mean parameters are used as deterministic starting points for per-task fine-tuning.

Two problem families are supported:

- **XYZ** — ground-state energy estimation for XYZ-type spin Hamiltonians.
- **QAOA-MaxCut** — MaxCut via single-/multi-layer QAOA on random weighted graphs.

Each family supports both a deterministic point-estimate hypernetwork and a distributional variant that predicts a task-conditioned mean and exploration scale. The distributional variant corresponds to the proposed method and uses a KNN-prototype KL regularizer during Phase 1.

## Requirements

```bash
pip install -r requirements.txt
```

The quantum circuits run on PennyLane's `lightning.gpu` backend, so a
CUDA-capable GPU and a matching CUDA toolkit are required.

## Repository layout

```
main.py                 # entry point: config loading, phase dispatch, task launch
configs/                # YAML configs (one per task, with phase1/phase2 blocks)
  xyz.yaml  xyz_vop.yaml  qaoa.yaml  qaoa_vop.yaml
tasks/
  common/               # shared dataset / model / runner interfaces
  xyz/      xyz_vop/     # XYZ task: dataset, model, runner(s)
  qaoa/     qaoa_vop/    # QAOA task: dataset, model, runner(s)
dataset/
  generate_xyz_dataset.py   # regenerates the XYZ Hamiltonian CSVs
  xyz_ham/                  # generated XYZ data lands here (data_dir default)
run.sh                  # convenience wrapper (Phase 1 -> auto-locate ckpt -> Phase 2)
```

## Datasets

Trained checkpoints are **not** included. QAOA generates its random graphs on
the fly, so it needs no data files. The XYZ task reads Hamiltonian CSVs from the
`data_dir` given in the config (default `dataset/xyz_ham`); generate them with:

```bash
# 20-qubit data used in the paper (3000 train / 100 test) -> dataset/xyz_ham/
python dataset/generate_xyz_dataset.py --qubit 20

# smaller / faster system
python dataset/generate_xyz_dataset.py --qubit 10
```

For each sample the script draws couplings `(Jx, Jy, Jz) ~ U(0, 2]` and computes
the ground-state energy `E0` of the periodic XYZ chain (the same Hamiltonian as
`tasks/xyz/model.py`) with a sparse Lanczos solver. With the default
`--seed 42` this reproduces the released dataset bit-for-bit. Output is written
in the header-less `[4, N]` layout (`rows = Jx, Jy, Jz, E0`) that the loader
expects, so no post-processing/transpose step is needed. See `--help` for the
full set of options (`--num-train`, `--num-test`, `--low`/`--high`, `--out-dir`).

## Usage

The entry point selects a task and a phase, and supports dotted config
overrides via `--set`.

```bash
# Phase 1: train the hypernetwork
python main.py --task xyz --config configs/xyz.yaml --phase phase1

# Phase 2: per-task fine-tuning from a Phase-1 checkpoint
python main.py --task xyz --config configs/xyz.yaml --phase phase2 \
    --set phases.phase2.config.checkpoint_path=<path/to/best.pt>

# Random-initialization baseline (no hypernetwork)
python main.py --task xyz --config configs/xyz.yaml --phase randinit
```

The distributional initializer variants are implemented under the `_vop` task names:

```bash
python main.py --task xyz_vop  --config configs/xyz_vop.yaml  --phase phase1
python main.py --task qaoa_vop --config configs/qaoa_vop.yaml --phase phase1
```

Common overrides:

```bash
--set phases.phase1.config.seed=2026          # change seed
--set phases.phase1.config.epoch=50           # change number of epochs
CUDA_VISIBLE_DEVICES=0 python main.py ...      # pin GPU
```

`run.sh` automates the Phase 1 -> Phase 2 hand-off (it locates the Phase-1
`best.pt` and passes it to Phase 2):

```bash
./run.sh xyz all 2025      # task, phase, seed
./run.sh qaoa all 2025
```

## Configuration

Each `configs/*.yaml` defines a `phases` block with `phase1`, `phase2`, and
(where applicable) `randinit` sections. Any value inside a phase config can be
overridden on the command line with `--set <dotted.key>=<value>`.

During Phase 2, sampling and KL regularization are disabled; only the learned mean initialization is used.