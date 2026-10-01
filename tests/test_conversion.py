"""Rotation conversion tests for Quaternion class.

Tests for the conversions to and from rotation matrices, axis-angle, rotation vectors, Euler
angles, and jax.scipy Rotation.
"""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.spatial.transform import Rotation

from fastquat.quaternion import Quaternion

EXTRINSIC_SEQS = [
    ''.join(axes)
    for axes in itertools.product('xyz', repeat=3)
    if axes[0] != axes[1] and axes[1] != axes[2]
]
SEQS = EXTRINSIC_SEQS + [seq.upper() for seq in EXTRINSIC_SEQS]


def angle_diff(a, b):
    """Absolute difference of angles, modulo 2π."""
    return jnp.abs((a - b + jnp.pi) % (2 * jnp.pi) - jnp.pi)


def is_proper(seq: str) -> bool:
    return seq[0] == seq[2]


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


@pytest.mark.parametrize(
    'axis', [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -2.0, 0.5]]
)
@pytest.mark.parametrize('angle', [jnp.pi, 0.999 * jnp.pi, 0.75 * jnp.pi, -0.6 * jnp.pi])
@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_matrix_large_angles(axis, angle, do_jit):
    """Rotations by angles up to π, where the trace of the matrix is -1, are recovered."""
    func = Quaternion.from_rotation_matrix
    if do_jit:
        func = jax.jit(func)

    q = Quaternion.from_axis_angle(jnp.array(axis), angle)
    result = func(q.to_rotation_matrix())
    assert jnp.allclose(abs(result), 1.0, atol=1e-6)
    # q and -q are the same rotation
    sign = jnp.sign(jnp.sum(result.wxyz * q.wxyz))
    assert jnp.allclose(sign * result.wxyz, q.wxyz, atol=1e-5)


def test_from_rotation_matrix_roundtrip_random():
    """Random rotations cover the four cases of the conversion."""
    q = Quaternion.random(jax.random.key(0), (1000,))
    result = Quaternion.from_rotation_matrix(q.to_rotation_matrix())
    assert jnp.allclose(abs(result), 1.0, atol=1e-6)
    sign = jnp.sign(jnp.sum(result.wxyz * q.wxyz, axis=-1))[:, None]
    assert jnp.allclose(sign * result.wxyz, q.wxyz, atol=1e-5)
    assert jnp.all(result.w >= 0)


def test_from_rotation_matrix_grad_at_pi():
    """The gradient is finite for rotations by π."""

    def func(rot):
        return Quaternion.from_rotation_matrix(rot).wxyz

    rot = jnp.diag(jnp.array([1.0, -1.0, -1.0]))  # rotation by π about x
    jac = jax.jacfwd(func)(rot)
    assert jnp.all(jnp.isfinite(jac))


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


# from_euler
@pytest.mark.parametrize('seq', SEQS)
@pytest.mark.parametrize('do_jit', [False, True])
def test_from_euler(seq, do_jit):
    """from_euler gives the same rotations as scipy."""
    func = partial(Quaternion.from_euler, seq)
    if do_jit:
        func = jax.jit(func)

    angles = jax.random.uniform(jax.random.key(0), (50, 3), minval=-jnp.pi, maxval=jnp.pi)
    q = func(angles)
    assert q.shape == (50,)
    assert jnp.allclose(abs(q), 1.0, atol=1e-6)
    expected = Rotation.from_euler(seq, angles).as_matrix()
    assert jnp.allclose(q.to_rotation_matrix(), expected, atol=1e-5)


@pytest.mark.parametrize('seq', ['x', 'Y', 'zx', 'XZ'])
def test_from_euler_short_sequences(seq):
    """Sequences of 1 or 2 axes are supported."""
    angles = jnp.array([0.3, -1.2])[: len(seq)]
    q = Quaternion.from_euler(seq, angles)
    expected = Rotation.from_euler(seq, angles).as_matrix()
    assert jnp.allclose(q.to_rotation_matrix(), expected, atol=1e-6)


def test_from_euler_single_axis():
    """A single axis rotation is the corresponding axis-angle rotation."""
    q = Quaternion.from_euler('z', jnp.array([jnp.pi / 2]))
    expected = Quaternion.from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi / 2)
    assert jnp.allclose(q.wxyz, expected.wxyz, atol=1e-6)


def test_from_euler_intrinsic_is_reversed_extrinsic():
    """An intrinsic sequence is the reversed extrinsic sequence with the angles reversed."""
    angles = jnp.array([0.3, -1.2, 2.0])
    q_intrinsic = Quaternion.from_euler('XYZ', angles)
    q_extrinsic = Quaternion.from_euler('zyx', angles[::-1])
    assert jnp.allclose(q_intrinsic.wxyz, q_extrinsic.wxyz, atol=1e-6)


def test_from_euler_degrees():
    angles = jnp.array([30.0, -45.0, 120.0])
    q_deg = Quaternion.from_euler('zyx', angles, degrees=True)
    q_rad = Quaternion.from_euler('zyx', jnp.deg2rad(angles))
    assert jnp.allclose(q_deg.wxyz, q_rad.wxyz, atol=1e-6)


def test_from_euler_integer_input():
    """Integer angles are promoted to floating point."""
    q = Quaternion.from_euler('xyz', jnp.array([0, 0, 0]))
    assert jnp.issubdtype(q.dtype, jnp.floating)
    assert jnp.allclose(q.wxyz, jnp.array([1.0, 0.0, 0.0, 0.0]))


def test_from_euler_wrong_shape():
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\) for sequence 'xyz'"):
        Quaternion.from_euler('xyz', jnp.zeros(2))


# to_euler
@pytest.mark.parametrize('seq', SEQS)
@pytest.mark.parametrize('do_jit', [False, True])
def test_to_euler(seq, do_jit):
    """to_euler gives the same angles as scipy, in the documented ranges."""

    def func(q):
        return q.to_euler(seq)

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.random(jax.random.key(0), (100,))
    angles = func(q)
    assert angles.shape == (100, 3)
    assert jnp.all(angle_diff(angles, q.to_scipy_rotation().as_euler(seq)) < 1e-4)
    assert jnp.all(jnp.abs(angles[:, [0, 2]]) <= jnp.pi + 1e-6)
    if is_proper(seq):
        assert jnp.all((angles[:, 1] >= 0) & (angles[:, 1] <= jnp.pi + 1e-6))
    else:
        assert jnp.all(jnp.abs(angles[:, 1]) <= jnp.pi / 2 + 1e-6)


@pytest.mark.parametrize('seq', SEQS)
def test_to_euler_roundtrip(seq):
    """from_euler(to_euler(q)) is the same rotation as q."""
    q = Quaternion.random(jax.random.key(1), (100,))
    q2 = Quaternion.from_euler(seq, q.to_euler(seq))
    assert jnp.allclose(q2.to_rotation_matrix(), q.to_rotation_matrix(), atol=1e-5)


@pytest.mark.parametrize('seq', SEQS)
@pytest.mark.parametrize('at_lower_bound', [False, True])
def test_to_euler_gimbal_lock(enable_x64: None, seq, at_lower_bound):
    """In gimbal lock, the third angle is 0 and the rotation is preserved, as in scipy."""
    if is_proper(seq):
        middle = 0.0 if at_lower_bound else jnp.pi
    else:
        middle = -jnp.pi / 2 if at_lower_bound else jnp.pi / 2
    angles = jnp.array([[0.4, middle, -0.3], [-2.0, middle, 1.5]])
    q = Quaternion.from_euler(seq, angles)
    result = q.to_euler(seq)
    assert jnp.all(result[:, 2] == 0)
    assert jnp.all(angle_diff(result, Rotation.from_euler(seq, angles).as_euler(seq)) < 1e-9)
    q2 = Quaternion.from_euler(seq, result)
    assert jnp.allclose(q2.to_rotation_matrix(), q.to_rotation_matrix(), atol=1e-9)


def test_to_euler_degrees():
    q = Quaternion.random(jax.random.key(2), (10,))
    assert jnp.allclose(q.to_euler('ZYX', degrees=True), jnp.rad2deg(q.to_euler('ZYX')))


def test_to_euler_non_unit_and_negated():
    """The angles depend neither on the norm nor on the sign of the quaternion."""
    q = Quaternion.random(jax.random.key(3), (10,))
    expected = q.to_euler('xyz')
    assert jnp.allclose((3.0 * q).to_euler('xyz'), expected, atol=1e-5)
    assert jnp.allclose((-q).to_euler('xyz'), expected, atol=1e-5)


def test_to_euler_wrong_length():
    with pytest.raises(ValueError, match='Expected 3 axes'):
        Quaternion.random(jax.random.key(0)).to_euler('xy')


# Sequence validation
@pytest.mark.parametrize('seq', ['', 'xyzx', 'xyZ', 'xwz', 'xxy', 'XYY'])
def test_invalid_sequence(seq):
    with pytest.raises(ValueError):
        Quaternion.from_euler(seq, jnp.zeros(len(seq)))
    with pytest.raises(ValueError):
        Quaternion.random(jax.random.key(0)).to_euler(seq)


# from_scipy_rotation, to_scipy_rotation
def test_scipy_rotation_component_order():
    """Rotation quaternions are scalar-last (xyzw), Quaternion ones are scalar-first (wxyz)."""
    q = Quaternion(0.1, 0.2, 0.3, 0.4).normalize()
    rotation = q.to_scipy_rotation()
    assert isinstance(rotation, Rotation)
    assert jnp.allclose(rotation.as_quat(), jnp.roll(q.wxyz, -1), atol=1e-6)
    assert jnp.allclose(Quaternion.from_scipy_rotation(rotation).wxyz, q.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_scipy_rotation_roundtrip(do_jit):
    """The conversions are inverse, keep the batch shape, and agree on the rotation."""

    def func(q):
        return Quaternion.from_scipy_rotation(q.to_scipy_rotation())

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.random(jax.random.key(0), (4, 5))
    assert jnp.allclose(func(q).wxyz, q.wxyz, atol=1e-6)
    assert jnp.allclose(q.to_scipy_rotation().as_matrix(), q.to_rotation_matrix(), atol=1e-5)


def test_to_scipy_rotation_normalizes():
    q = Quaternion(2.0, 0.0, 0.0, 0.0)
    assert jnp.allclose(q.to_scipy_rotation().as_quat(), jnp.array([0.0, 0.0, 0.0, 1.0]))


def test_from_scipy_rotation_numpy_scipy():
    """A NumPy scipy Rotation is accepted too."""
    scipy_transform = pytest.importorskip('scipy.spatial.transform')
    rotation = scipy_transform.Rotation.from_euler('z', 90, degrees=True)
    q = Quaternion.from_scipy_rotation(rotation)
    expected = Quaternion.from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi / 2)
    assert jnp.allclose(q.wxyz, expected.wxyz, atol=1e-6)
