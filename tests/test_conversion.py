"""Rotation conversion tests for Quaternion class.

Tests for the conversions to and from rotation matrices, axis-angle, and rotation vectors.
"""

import jax
import jax.numpy as jnp
import pytest

from fastquat.quaternion import Quaternion


# from_rotation_matrix
@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_matrix_identity(do_jit):
    """Test from_rotation_matrix with identity matrix."""

    def func(rot):
        return Quaternion.from_rotation_matrix(rot)

    if do_jit:
        func = jax.jit(func)

    identity_matrix = jnp.eye(3)
    q = func(identity_matrix)

    expected = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_matrix_90deg_z(do_jit):
    """Test from_rotation_matrix with 90° rotation around z-axis."""

    def func(rot):
        return Quaternion.from_rotation_matrix(rot)

    if do_jit:
        func = jax.jit(func)

    rot_z_90 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q = func(rot_z_90)

    angle = jnp.pi / 2
    expected = jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])
    assert jnp.allclose(q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_matrix_batch(do_jit):
    """Test from_rotation_matrix with batch of matrices."""

    def func(rot_batch):
        return Quaternion.from_rotation_matrix(rot_batch)

    if do_jit:
        func = jax.jit(func)

    rot_batch = jnp.tile(jnp.eye(3), (3, 1, 1))
    q_batch = func(rot_batch)

    assert q_batch.shape == (3,)
    expected_batch = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    assert jnp.allclose(q_batch.wxyz, expected_batch, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_matrix_wrong_shape(do_jit):
    """Test from_rotation_matrix with wrong shape raises ValueError."""

    def func(rot):
        return Quaternion.from_rotation_matrix(rot)

    if do_jit:
        func = jax.jit(func)

    wrong_matrix = jnp.ones((2, 2))
    with pytest.raises(ValueError):
        func(wrong_matrix)


# to_rotation_matrix
@pytest.mark.parametrize('do_jit', [False, True])
def test_to_rotation_matrix_identity(do_jit):
    """Test to_rotation_matrix with identity quaternion."""

    def func(q):
        return q.to_rotation_matrix()

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0)
    R = func(q)

    assert jnp.allclose(R, jnp.eye(3), atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_to_rotation_matrix_90deg_z(do_jit):
    """Test to_rotation_matrix with 90° rotation around z-axis."""

    def func(q):
        return q.to_rotation_matrix()

    if do_jit:
        func = jax.jit(func)

    angle = jnp.pi / 2
    q = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))
    R = func(q)

    expected = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert jnp.allclose(R, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_to_rotation_matrix_is_orthogonal(do_jit):
    """Test that rotation matrix is orthogonal."""

    def func(q):
        return q.to_rotation_matrix()

    if do_jit:
        func = jax.jit(func)

    key = jax.random.PRNGKey(42)
    q = Quaternion.random(key)
    R = func(q)

    # Check orthogonality: R @ R.T = I
    assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)
    # Check determinant = 1
    assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_to_rotation_matrix_batch(do_jit):
    """Test to_rotation_matrix with batch of quaternions."""

    def func(q_batch):
        return q_batch.to_rotation_matrix()

    if do_jit:
        func = jax.jit(func)

    key = jax.random.PRNGKey(42)
    q_batch = Quaternion.random(key, (3,))
    R_batch = func(q_batch)

    assert R_batch.shape == (3, 3, 3)

    # All should be orthogonal
    for i in range(3):
        R = R_batch[i]
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)


# Rotation matrix round trip
@pytest.mark.parametrize('do_jit', [False, True])
def test_matrix_quaternion_roundtrip(do_jit):
    """Test round-trip conversion: quaternion -> matrix -> quaternion."""

    def func(q):
        R = q.to_rotation_matrix()
        return Quaternion.from_rotation_matrix(R)

    if do_jit:
        func = jax.jit(func)

    q_original = Quaternion(0.7071, 0.7071, 0.0, 0.0).normalize()
    q_recovered = func(q_original)

    # Should be the same (up to sign ambiguity)
    assert jnp.allclose(q_recovered.wxyz, q_original.wxyz, atol=1e-4) or jnp.allclose(
        q_recovered.wxyz, -q_original.wxyz, atol=1e-4
    )


# from_axis_angle
@pytest.mark.parametrize(
    'axis, angle, expected',
    [
        ([0.0, 0.0, 1.0], 0.0, [1.0, 0.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], jnp.pi, [0.0, 1.0, 0.0, 0.0]),
        ([0.0, 2.0, 0.0], jnp.pi / 2, [jnp.sqrt(0.5), 0.0, jnp.sqrt(0.5), 0.0]),
        ([0.0, 0.0, 1.0], -jnp.pi / 2, [jnp.sqrt(0.5), 0.0, 0.0, -jnp.sqrt(0.5)]),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_from_axis_angle(axis, angle, expected, do_jit):
    """Test from_axis_angle against known quaternions (axis need not be normalized)."""
    func = Quaternion.from_axis_angle
    if do_jit:
        func = jax.jit(func)

    q = func(jnp.array(axis), jnp.array(angle))
    assert jnp.allclose(q.wxyz, jnp.array(expected), atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_axis_angle_right_hand_rule(do_jit):
    """A positive rotation about z maps x to y."""

    def func(angle):
        return Quaternion.from_axis_angle(jnp.array([0.0, 0.0, 1.0]), angle).rotate_vector(
            jnp.array([1.0, 0.0, 0.0])
        )

    if do_jit:
        func = jax.jit(func)

    assert jnp.allclose(func(jnp.pi / 2), jnp.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_from_axis_angle_consistency_with_matrix():
    """The rotation matrix of from_axis_angle is Rodrigues' formula."""
    axis = jnp.array([1.0, -2.0, 0.5])
    angle = 0.7
    n = axis / jnp.linalg.norm(axis)
    k = jnp.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    expected = jnp.eye(3) + jnp.sin(angle) * k + (1 - jnp.cos(angle)) * k @ k

    q = Quaternion.from_axis_angle(axis, angle)
    assert jnp.allclose(q.to_rotation_matrix(), expected, atol=1e-6)


def test_from_axis_angle_broadcast():
    """Axes and angles broadcast against each other."""
    axes = jnp.eye(3)  # (3, 3)
    angles = jnp.array([[0.1], [0.2]])  # (2, 1)
    q = Quaternion.from_axis_angle(axes, angles)
    assert q.shape == (2, 3)
    assert jnp.allclose(abs(q), 1.0, atol=1e-6)
    expected = Quaternion.from_axis_angle(axes[2], angles[1, 0])
    assert jnp.allclose(q[1, 2].wxyz, expected.wxyz, atol=1e-6)


def test_from_axis_angle_grad_at_zero():
    """The derivative with respect to the angle is finite and exact at zero."""

    def func(angle):
        return Quaternion.from_axis_angle(jnp.array([0.0, 1.0, 0.0]), angle).wxyz

    jac = jax.jacfwd(func)(0.0)
    assert jnp.allclose(jac, jnp.array([0.0, 0.0, 0.5, 0.0]))


def test_from_axis_angle_wrong_shape():
    with pytest.raises(ValueError, match='Axis must have shape'):
        Quaternion.from_axis_angle(jnp.array([1.0, 0.0]), 0.1)


# from_rotation_vector
@pytest.mark.parametrize(
    'rotvec, expected',
    [
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ([jnp.pi, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
        ([0.0, jnp.pi / 2, 0.0], [jnp.sqrt(0.5), 0.0, jnp.sqrt(0.5), 0.0]),
        ([0.0, 0.0, -jnp.pi / 2], [jnp.sqrt(0.5), 0.0, 0.0, -jnp.sqrt(0.5)]),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_vector(rotvec, expected, do_jit):
    """Test from_rotation_vector against known quaternions."""
    func = Quaternion.from_rotation_vector
    if do_jit:
        func = jax.jit(func)

    q = func(jnp.array(rotvec))
    assert jnp.allclose(q.wxyz, jnp.array(expected), atol=1e-6)


def test_from_rotation_vector_consistency_with_axis_angle():
    """A rotation vector is the axis scaled by the angle."""
    axes = jnp.array([[1.0, -2.0, 0.5], [0.0, 0.3, 0.4], [-1.0, 1.0, 1.0]])
    angles = jnp.array([0.7, -2.5, 4.0])
    unit_axes = axes / jnp.linalg.norm(axes, axis=-1, keepdims=True)

    q = Quaternion.from_rotation_vector(angles[:, None] * unit_axes)
    expected = Quaternion.from_axis_angle(axes, angles)
    assert q.shape == (3,)
    assert jnp.allclose(q.wxyz, expected.wxyz, atol=1e-6)


def test_from_rotation_vector_integer_input():
    """Integer rotation vectors are promoted to floating point."""
    q = Quaternion.from_rotation_vector(jnp.array([0, 0, 0]))
    assert jnp.issubdtype(q.dtype, jnp.floating)
    assert jnp.allclose(q.wxyz, jnp.array([1.0, 0.0, 0.0, 0.0]))


def test_from_rotation_vector_grad_at_zero():
    """The derivative is finite and exact at the zero vector."""

    def func(rotvec):
        return Quaternion.from_rotation_vector(rotvec).wxyz

    jac = jax.jacfwd(func)(jnp.zeros(3))
    expected = jnp.concatenate([jnp.zeros((1, 3)), 0.5 * jnp.eye(3)])
    assert jnp.allclose(jac, expected)
    jac = jax.jacrev(func)(jnp.zeros(3))
    assert jnp.allclose(jac, expected)


def test_from_rotation_vector_wrong_shape():
    with pytest.raises(ValueError, match='Rotation vector must have shape'):
        Quaternion.from_rotation_vector(jnp.array([1.0, 0.0]))


# to_rotation_vector
@pytest.mark.parametrize(
    'wxyz, expected',
    [
        ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 1.0, 0.0, 0.0], [jnp.pi, 0.0, 0.0]),
        ([jnp.sqrt(0.5), 0.0, jnp.sqrt(0.5), 0.0], [0.0, jnp.pi / 2, 0.0]),
        ([-jnp.sqrt(0.5), 0.0, 0.0, jnp.sqrt(0.5)], [0.0, 0.0, -jnp.pi / 2]),
        ([2.0, 0.0, 0.0, 2.0], [0.0, 0.0, jnp.pi / 2]),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_to_rotation_vector(wxyz, expected, do_jit):
    """Test to_rotation_vector against known rotation vectors (q and -q agree, |q| is ignored)."""

    def func(q):
        return q.to_rotation_vector()

    if do_jit:
        func = jax.jit(func)

    rotvec = func(Quaternion.from_array(jnp.array(wxyz)))
    assert jnp.allclose(rotvec, jnp.array(expected), atol=1e-6)


def test_to_rotation_vector_roundtrip():
    """from_rotation_vector and to_rotation_vector are inverse for angles in [0, π)."""
    key = jax.random.key(0)
    q = Quaternion.random(key, (100,))
    rotvec = q.to_rotation_vector()
    assert rotvec.shape == (100, 3)
    angle = jnp.linalg.norm(rotvec, axis=-1)
    assert jnp.all(angle <= jnp.pi + 1e-6)
    q2 = Quaternion.from_rotation_vector(rotvec)
    assert jnp.allclose(q2.to_rotation_matrix(), q.to_rotation_matrix(), atol=1e-5)
    # Near π, rounding can flip the sign of w and return the opposite vector
    away_from_pi = (angle < jnp.pi - 1e-3)[:, None]
    assert jnp.allclose(
        jnp.where(away_from_pi, q2.to_rotation_vector(), 0),
        jnp.where(away_from_pi, rotvec, 0),
        atol=1e-5,
    )


def test_to_rotation_vector_at_pi():
    """At angle π, q and -q may give opposite vectors, but of norm π and for the same rotation."""
    q = Quaternion(0.0, 0.0, 1.0, 0.0)
    rotvec, rotvec_neg = q.to_rotation_vector(), (-q).to_rotation_vector()
    assert jnp.allclose(jnp.linalg.norm(rotvec), jnp.pi)
    assert jnp.allclose(jnp.linalg.norm(rotvec_neg), jnp.pi)
    assert jnp.allclose(
        Quaternion.from_rotation_vector(rotvec).to_rotation_matrix(),
        Quaternion.from_rotation_vector(rotvec_neg).to_rotation_matrix(),
        atol=1e-6,
    )


def test_to_rotation_vector_grad_at_identity():
    """The derivative is finite and exact at the identity."""

    def func(wxyz):
        return Quaternion.from_array(wxyz).to_rotation_vector()

    wxyz = jnp.array([1.0, 0.0, 0.0, 0.0])
    expected = jnp.concatenate([jnp.zeros((3, 1)), 2 * jnp.eye(3)], axis=1)
    assert jnp.allclose(jax.jacfwd(func)(wxyz), expected)
    assert jnp.allclose(jax.jacrev(func)(wxyz), expected)
