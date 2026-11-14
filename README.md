# Learning Task-Conditioned Distributional Initializers for Variational Quantum Algorithms

Implementation of a two-phase approach to initialization for variational quantum algorithms (VQAs).

In Phase 1, a task-conditioned hypernetwork is meta-trained to predict circuit initialization parameters from task descriptors. In Phase 2, the predicted mean parameters are used as deterministic starting points for per-task circuit optimization.

The repository includes implementations for two problem families:

- **XYZ** — Ground-state energy estimation for XYZ spin Hamiltonians.
- **QAOA-MaxCut** — MaxCut optimization using QAOA on weighted random graphs.

For each family, the repository includes a point-trained initializer and a distribution-trained variant. The distributional method predicts a task-conditioned mean and exploration scale and uses KNN-prototype KL regularization during meta-training.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The implementations use PennyLane's `lightning.gpu` backend for quantum circuit simulation. Running this backend requires a compatible NVIDIA GPU and CUDA environment.

## Repository Layout

```text
main.py                   # Configuration loading and experiment entry point
configs/                  # Configuration files for each task
  xyz.yaml
  xyz_vop.yaml
  qaoa.yaml
  qaoa_vop.yaml
tasks/
  common/                 # Shared components
  xyz/                    # XYZ point-trained implementation
  xyz_vop/                # XYZ distribution-trained implementation
  qaoa/                   # QAOA point-trained implementation
  qaoa_vop/               # QAOA distribution-trained implementation
dataset/
  generate_xyz_dataset.py # XYZ dataset generation
  xyz_ham/                # Default XYZ dataset directory
run.sh                    # Script for running experiment phases
THIRD_PARTY_LICENSES/
  Q-MAML-LICENSE        # original Q-MAML MIT license and copyright notice
```

## Datasets

Trained checkpoints are not included in this repository.

For QAOA-MaxCut, graphs are generated during experiment execution according to the corresponding configuration.

For XYZ, Hamiltonian datasets are loaded from the directory specified by `data_dir` (default: `dataset/xyz_ham`).

The dataset generation script can be used as follows:

```bash
# Generate data for a 20-qubit XYZ system
python dataset/generate_xyz_dataset.py --qubit 20

# Generate data for a 10-qubit XYZ system
python dataset/generate_xyz_dataset.py --qubit 10
```

The script generates XYZ coupling parameters and estimates the corresponding ground-state energies using a sparse eigensolver.

The generated datasets are intended for use with the XYZ implementations in this repository. Dataset size, random seed, coupling ranges, and output location can be configured through the script's command-line arguments.

For available options, run:

```bash
python dataset/generate_xyz_dataset.py --help
```

## Usage

Experiments are configured through YAML files and launched using `main.py`.

### Phase 1: Meta-training

```bash
python main.py --task xyz --config configs/xyz.yaml --phase phase1
```

### Phase 2: Circuit Optimization

Phase 2 uses a checkpoint obtained from Phase 1.

```bash
python main.py --task xyz --config configs/xyz.yaml --phase phase2 \
  --set phases.phase2.config.checkpoint_path=<path/to/best.pt>
```

### Random Initialization Baseline

```bash
python main.py --task xyz --config configs/xyz.yaml --phase randinit
```

### Distributional Initializers

Distribution-trained variants use the `_vop` task names:

```bash
python main.py --task xyz_vop --config configs/xyz_vop.yaml --phase phase1

python main.py --task qaoa_vop --config configs/qaoa_vop.yaml --phase phase1
```


### Configuration Overrides

Selected configuration values can be overridden from the command line.

For example:

```bash
# Override configuration values
python main.py --task xyz --config configs/xyz.yaml --phase phase1 \
  --set phases.phase1.config.seed=2026

# Select GPU
CUDA_VISIBLE_DEVICES=0 python main.py --task xyz --config configs/xyz.yaml --phase phase1
```


### Running Both Phases

The repository also includes `run.sh`, which supports running Phase 1 followed by Phase 2.

```bash
./run.sh xyz all 2025
./run.sh qaoa all 2025
```

## Configuration

Each task has a YAML configuration file in `configs/`.

The configuration files define settings for Phase 1, Phase 2, and, where applicable, random-initialization experiments.

Configuration values can be adjusted directly in the YAML files or overridden using the `--set` argument.

## Acknowledgments and Code Attribution

The XYZ implementation builds on the publicly available code associated with **Q-MAML** by Lee et al. (AAAI 2025). We extend it with task-conditioned distributional initialization, learned exploration scales, and KNN-prototype KL regularization. Some experimental settings differ from the original work.

- **Paper:** [Q-MAML: Quantum Model-Agnostic Meta-Learning for Variational Quantum Algorithms](https://doi.org/10.1609/aaai.v39i17.33995)
- **Original code:** [JayC1208/Q-MAML](https://github.com/JayC1208/Q-MAML)

The original Q-MAML code is distributed under the MIT License. Its copyright and license notice are included in [`THIRD_PARTY_LICENSES/Q-MAML-LICENSE`](THIRD_PARTY_LICENSES/Q-MAML-LICENSE).
