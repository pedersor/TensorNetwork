from tensornetwork import FiniteMPS
from tensornetwork.matrixproductstates.dmrg import FiniteDMRG, BaseDMRG
from tensornetwork.backends import backend_factory
from tensornetwork.matrixproductstates.mpo import FiniteXXZ, FiniteTFI
import pytest
import numpy as np
import copy
import sys


def get_XXZ_Hamiltonian(N, Jx, Jy, Jz):
    Sx = {}
    Sy = {}
    Sz = {}
    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, 0.5], [-0.5, 0]])
    sz = np.diag([-0.5, 0.5])
    for n in range(N):
        Sx[n] = np.kron(np.kron(np.eye(2 ** n), sx), np.eye(2 ** (N - 1 - n)))
        Sy[n] = np.kron(np.kron(np.eye(2 ** n), sy), np.eye(2 ** (N - 1 - n)))
        Sz[n] = np.kron(np.kron(np.eye(2 ** n), sz), np.eye(2 ** (N - 1 - n)))
    H = np.zeros((2 ** N, 2 ** N))
    for n in range(N - 1):
        H += Jx * Sx[n] @ Sx[n + 1] - Jy * Sy[n] @ Sy[n + 1] + Jz * Sz[n] @ Sz[
            n +
            1]
    return H


backend = 'numpy'
dtype = np.float64

N = 7

# H = get_XXZ_Hamiltonian(N, 1, 1, 1)
# eta, _ = np.linalg.eigh(H)

mpo = FiniteXXZ(
    Jz=np.ones(N - 1),
    Jxy=np.ones(N - 1),
    Bz=np.zeros(N),
    dtype=dtype,
    backend=backend)

D = 32
mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)
mps1tensors = mps.tensors
mps1 = FiniteMPS(mps1tensors, center_position=0, backend=backend)

dmrg = FiniteDMRG(mps, mpo)
energy = dmrg.run_two_site(num_sweeps=6, num_krylov_vecs=10, verbose=1)
