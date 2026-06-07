#!/usr/bin/env python3
"""Generate the XYZ-Hamiltonian dataset used by the ``xyz`` / ``xyz_vop`` tasks.

For each sample we draw three coupling coefficients ``(Jx, Jy, Jz)`` and compute
the ground-state energy ``E0`` of the periodic XYZ spin chain

    H = sum_w  Jx X_w X_{w+1} + Jy Y_w Y_{w+1} + Jz Z_w Z_{w+1}      (w+1 mod n)

This is exactly the Hamiltonian built in ``tasks/xyz/model.py`` (``xyz_ham``),
so the descriptor ``x = (Jx, Jy, Jz)`` and target ``y = E0`` produced here line
up with what the hypernetwork is trained to predict.

The ground state is obtained with a sparse Lanczos solver (``scipy.eigsh``,
``which='SA'``), which scales to ~20 qubits on a single machine. The sparse
Hamiltonian is assembled from Kronecker blocks only (no dense ``2^n x 2^n``
matrix multiplies), so construction is cheap relative to the eigensolve.

Output format matches ``tasks/xyz/dataset.py`` exactly: a header-less CSV with
**features in rows, samples in columns** (shape ``[4, N]``), where rows are
``[Jx, Jy, Jz, E0]``. Files are named ``{qubit}_data_continuous_3000.csv``
(train) and ``{qubit}_data_continuous_3000_test.csv`` (test) regardless of the
sample count, to match the loader's hard-coded filenames.

Example
-------
    # 10-qubit dataset (fast), default 3000 train / 100 test
    python dataset/generate_xyz_dataset.py --qubit 10

    # 20-qubit dataset matching the released data
    python dataset/generate_xyz_dataset.py --qubit 20 \
        --num-train 3000 --num-test 100 --out-dir dataset/xyz_ham
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

# Single-qubit Pauli operators as sparse matrices.
_I = sp.identity(2, format="csr", dtype=complex)
_X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_Y = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
_Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))


def _edge_term(pauli: sp.csr_matrix, a: int, b: int, n: int) -> sp.csr_matrix:
    """Two-site operator ``P_a P_b`` on an ``n``-qubit chain (Kronecker blocks).

    For an adjacent edge ``b == a + 1`` this is a single Kronecker product with
    identity padding; the periodic wrap edge ``(n-1, 0)`` is handled as a
    special case. Either way no dense matrix multiply is performed.
    """
    if b == a + 1:  # adjacent sites
        pp = sp.kron(pauli, pauli, format="csr")
        left = sp.identity(2**a, format="csr", dtype=complex)
        right = sp.identity(2 ** (n - a - 2), format="csr", dtype=complex)
        return sp.kron(sp.kron(left, pp, format="csr"), right, format="csr")
    if a == 0 and b == n - 1:  # periodic wrap: P on first and last qubit
        mid = sp.identity(2 ** (n - 2), format="csr", dtype=complex)
        return sp.kron(sp.kron(pauli, mid, format="csr"), pauli, format="csr")
    raise ValueError(f"unsupported edge ({a}, {b}) for n={n}")


def xyz_hamiltonian(delta, n: int) -> sp.csr_matrix:
    """Sparse periodic XYZ-chain Hamiltonian for couplings ``delta=(Jx,Jy,Jz)``."""
    jx, jy, jz = float(delta[0]), float(delta[1]), float(delta[2])
    h = sp.csr_matrix((2**n, 2**n), dtype=complex)
    for w in range(n):
        a, b = sorted((w, (w + 1) % n))
        h = h + jx * _edge_term(_X, a, b, n)
        h = h + jy * _edge_term(_Y, a, b, n)
        h = h + jz * _edge_term(_Z, a, b, n)
    return h.tocsr()


def ground_state_energy(delta, n: int) -> float:
    """Lowest eigenvalue of the XYZ Hamiltonian for the given couplings."""
    h = xyz_hamiltonian(delta, n)
    if n <= 4:  # eigsh needs k < dim-1; just go dense for tiny systems
        return float(np.linalg.eigvalsh(h.toarray()).min())
    vals = eigsh(h, k=1, which="SA", return_eigenvectors=False)
    return float(np.real(vals[0]))


def generate_split(n: int, num_samples: int, low: float, high: float, seed: int) -> np.ndarray:
    """Generate one split, returned as a ``[4, num_samples]`` array (rows = Jx,Jy,Jz,E0)."""
    rng = np.random.default_rng(seed)
    deltas = rng.uniform(low, high, size=(num_samples, 3))
    energies = np.empty(num_samples, dtype=np.float64)
    for i in tqdm(range(num_samples), desc=f"n={n} ({num_samples} samples)"):
        energies[i] = ground_state_energy(deltas[i], n)
    # Transpose to features-in-rows layout expected by tasks/xyz/dataset.py.
    return np.vstack([deltas.T, energies[None, :]])


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qubit", type=int, default=10, help="number of spins / qubits (must be even)")
    p.add_argument("--num-train", type=int, default=3000, help="number of training samples")
    p.add_argument("--num-test", type=int, default=100, help="number of test samples")
    p.add_argument("--low", type=float, default=0.0, help="lower bound for uniform coupling sampling")
    p.add_argument("--high", type=float, default=2.0, help="upper bound for uniform coupling sampling")
    p.add_argument("--seed", type=int, default=42, help="base RNG seed (test split uses seed+1)")
    p.add_argument("--out-dir", type=str, default="dataset/xyz_ham", help="output directory")
    return p


def main() -> None:
    opt = get_parser().parse_args()
    if opt.qubit % 2 != 0:
        raise ValueError("qubit count must be even (Neel initial state requires an even chain)")
    os.makedirs(opt.out_dir, exist_ok=True)

    splits = [
        ("train", opt.num_train, opt.seed, f"{opt.qubit}_data_continuous_3000.csv"),
        ("test", opt.num_test, opt.seed + 1, f"{opt.qubit}_data_continuous_3000_test.csv"),
    ]
    for name, count, seed, fname in splits:
        data = generate_split(opt.qubit, count, opt.low, opt.high, seed)
        path = os.path.join(opt.out_dir, fname)
        # Header-less, comma-delimited, [4, N]: matches np.loadtxt in the loader.
        np.savetxt(path, data, delimiter=",")
        print(f"[{name}] saved {data.shape[1]} samples -> {path}")


if __name__ == "__main__":
    main()
