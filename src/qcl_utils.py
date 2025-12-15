import numpy as np
from functools import reduce
from qiskit.circuit.library import UnitaryGate


# Basic gates
I_mat = np.eye(2, dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)


# Function to create a full-size gate matrix.
def make_fullgate(list_SiteAndOperator, nqubit):
    """
    Take ``list_SiteAndOperator = [[i_0, O_0], [i_1, O_1], ...]`` and insert
    identity operators for qubits not listed.
    Constructs the ``(2**nqubit, 2**nqubit)`` matrix corresponding to
    ``I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...``.
    """
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = []  # Collect single-qubit gates and combine with np.kron
    cnt = 0
    for i in range(nqubit):
        if i in list_Site:
            list_SingleGates.append( list_SiteAndOperator[cnt][1] )
            cnt += 1
        else:  # Insert identity on sites without specified gates
            list_SingleGates.append(I_mat)

    return reduce(np.kron, list_SingleGates)


def create_time_evol_gate(nqubit, time_step=0.77):
    """Create a random-field, random-coupling Ising Hamiltonian and return a Qiskit ``UnitaryGate``.

    :param time_step: Evolution time under the random Hamiltonian.
    :return: ``UnitaryGate`` representing the time evolution.
    """
    ham = np.zeros((2**nqubit,2**nqubit), dtype = complex)
    for i in range(nqubit):  # i runs 0 to nqubit-1
        Jx = -1. + 2.*np.random.rand()  # Random value in [-1, 1]
        ham += Jx * make_fullgate( [ [i, X_mat] ], nqubit)
        for j in range(i+1, nqubit):
            J_ij = -1. + 2.*np.random.rand()
            ham += J_ij * make_fullgate ([ [i, Z_mat], [j, Z_mat]], nqubit)

    # Diagonalize the Hamiltonian to create the time-evolution operator. H*P = P*D <-> H = P*D*P^dagger
    diag, eigen_vecs = np.linalg.eigh(ham)
    time_evol_op = np.dot(np.dot(eigen_vecs, np.diag(np.exp(-1j*time_step*diag))), eigen_vecs.T.conj())  # e^-iHT

    # Convert the matrix into a Qiskit gate
    return UnitaryGate(time_evol_op)


def min_max_scaling(x, axis=0):
    """Scale each feature of ``x`` to [-1, 1]."""
    min_val = x.min(axis=axis, keepdims=True)
    max_val = x.max(axis=axis, keepdims=True)
    scaled = (x - min_val) / (max_val - min_val)
    return 2.0 * scaled - 1.0


def softmax(x):
    """softmax function
    :param x: ndarray
    """
    exp_x = np.exp(x)
    y = exp_x / np.sum(np.exp(x))
    return y
