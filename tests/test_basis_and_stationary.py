"""Minimal tests for basis-matrix and stationary-eigenvector helpers."""

import numpy as np

from hetmacro.forward import stationary_eigenvector
from hetmacro.interpolation import cubic_basis_matrix, linear_basis_matrix, tensor_basis_matrix


def test_cubic_basis_shape_and_partition_of_unity():
    bp = np.linspace(0.0, 10.0, 51)
    x = np.linspace(0.0, 10.0, 101)
    phi = cubic_basis_matrix(bp, x).toarray()

    # For cubic with n breakpoints and open knots: n + 2 basis functions.
    assert phi.shape == (x.size, bp.size + 2)
    assert np.allclose(phi.sum(axis=1), 1.0, atol=1e-10)


def test_linear_basis_tent_structure():
    bp = np.array([0.0, 0.5, 1.0, 2.0])
    x = np.array([0.2, 1.8])
    phi = linear_basis_matrix(bp, x).toarray()

    assert phi.shape == (2, bp.size)
    assert np.allclose(phi.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(np.sum(phi > 0.0, axis=1) <= 2)
    assert np.allclose(phi[0], np.array([0.6, 0.4, 0.0, 0.0]), atol=1e-12)
    assert np.allclose(phi[1], np.array([0.0, 0.0, 0.2, 0.8]), atol=1e-12)


def test_tensor_basis_shape_and_row_sums():
    bp_a = np.linspace(0.0, 1.0, 5)
    bp_z = np.linspace(-1.0, 1.0, 4)
    x_a = np.array([0.0, 0.25, 0.75])
    x_z = np.array([-1.0, 0.0, 1.0])

    phi_a = linear_basis_matrix(bp_a, x_a)
    phi_z = linear_basis_matrix(bp_z, x_z)
    phi_az = tensor_basis_matrix(phi_a, phi_z).toarray()

    assert phi_az.shape == (x_a.size, bp_a.size * bp_z.size)
    assert np.allclose(phi_az.sum(axis=1), 1.0, atol=1e-12)


def test_stationary_eigenvector_matches_two_state_chain():
    q = np.array([[0.9, 0.1], [0.3, 0.7]])
    n = stationary_eigenvector(q)

    assert np.allclose(n.sum(), 1.0, atol=1e-12)
    assert np.all(n >= 0.0)
    assert np.allclose(n @ q, n, atol=1e-10)
    assert np.allclose(n, np.array([0.75, 0.25]), atol=1e-10)

