"""Rotation operations tests for Quaternion class.

Tests for from_rotation_matrix, to_rotation_matrix, rotate_vector, and slerp.
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


# Round-trip consistency
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


# rotate_vector
@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_identity(do_jit):
    """Test rotate_vector with identity quaternion."""

    def func(q, v):
        return q.rotate_vector(v)

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0)
    v = jnp.array([1.0, 2.0, 3.0])
    v_rotated = func(q, v)

    assert jnp.allclose(v_rotated, v, atol=1e-6)


@pytest.mark.parametrize(
    'axis,input_vec,expected_vec',
    [
        ('z', [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  # 90° around z: x -> y
        ('x', [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),  # 90° around x: y -> z
        ('y', [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),  # 90° around y: z -> x
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_90deg(axis, input_vec, expected_vec, do_jit):
    """Test rotate_vector with 90° rotations around each axis."""

    def func(q, v):
        return q.rotate_vector(v)

    if do_jit:
        func = jax.jit(func)

    angle = jnp.pi / 2
    if axis == 'x':
        q = Quaternion(jnp.cos(angle / 2), jnp.sin(angle / 2), 0.0, 0.0)
    elif axis == 'y':
        q = Quaternion(jnp.cos(angle / 2), 0.0, jnp.sin(angle / 2), 0.0)
    else:  # z
        q = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))

    v = jnp.array(input_vec)
    v_rotated = func(q, v)

    assert jnp.allclose(v_rotated, jnp.array(expected_vec), atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_preserves_length(do_jit):
    """Test rotate_vector preserves vector length."""

    def func(q, v):
        return q.rotate_vector(v)

    if do_jit:
        func = jax.jit(func)

    key = jax.random.PRNGKey(42)
    q = Quaternion.random(key)
    v = jax.random.normal(jax.random.split(key)[0], (3,))

    v_rotated = func(q, v)

    original_length = jnp.linalg.norm(v)
    rotated_length = jnp.linalg.norm(v_rotated)
    assert jnp.allclose(original_length, rotated_length, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_zero(do_jit):
    """Test rotation of zero vector."""

    def func(q, v):
        return q.rotate_vector(v)

    if do_jit:
        func = jax.jit(func)

    key = jax.random.PRNGKey(42)
    q = Quaternion.random(key)
    v_zero = jnp.zeros(3)

    v_rotated = func(q, v_zero)
    assert jnp.allclose(v_rotated, v_zero, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_batch_vectors(do_jit):
    """Test rotate_vector with batch of vectors."""

    def func(q, v_batch):
        return q.rotate_vector(v_batch)

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0)
    v_batch = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    v_rotated_batch = func(q, v_batch)
    assert jnp.allclose(v_rotated_batch, v_batch, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_batch_quaternions(do_jit):
    """Test rotate_vector with batch of quaternions."""

    def func(q_batch, v):
        return q_batch.rotate_vector(v)

    if do_jit:
        func = jax.jit(func)

    q_batch = Quaternion.from_array(jnp.tile(jnp.array([1, 0, 0, 0]), (3, 1)))
    v = jnp.array([1.0, 2.0, 3.0])

    v_rotated_batch = func(q_batch, v)

    expected_batch = jnp.tile(v, (3, 1))
    assert jnp.allclose(v_rotated_batch, expected_batch, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_multiple_vectors(do_jit):
    """Test rotating multiple vectors with same quaternion."""

    def func(q, v_batch):
        return q.rotate_vector(v_batch)

    if do_jit:
        func = jax.jit(func)

    # 90° rotation around z
    angle = jnp.pi / 2
    q = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))

    v_batch = jnp.array(
        [
            [1.0, 0.0, 0.0],  # Should become [0, 1, 0]
            [0.0, 1.0, 0.0],  # Should become [-1, 0, 0]
            [0.0, 0.0, 1.0],  # Should remain [0, 0, 1]
        ]
    )

    v_rotated_batch = func(q, v_batch)
    expected = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    assert jnp.allclose(v_rotated_batch, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_consistency_with_matrix(do_jit):
    """Test consistency between rotate_vector and to_rotation_matrix."""

    def func(q, v):
        v_rot1 = q.rotate_vector(v)
        R = q.to_rotation_matrix()
        v_rot2 = R @ v
        return v_rot1, v_rot2

    if do_jit:
        func = jax.jit(func)

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    q = Quaternion.random(key1)
    v = jax.random.normal(key2, (3,))

    v_rot1, v_rot2 = func(q, v)
    assert jnp.allclose(v_rot1, v_rot2, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_inverse(do_jit):
    """Test that inverse quaternion performs inverse rotation."""

    def func(q, v):
        v_rotated = q.rotate_vector(v)
        return (1 / q).rotate_vector(v_rotated)

    if do_jit:
        func = jax.jit(func)

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    q = Quaternion.random(key1)
    v = jax.random.normal(key2, (3,))

    v_recovered = func(q, v)
    assert jnp.allclose(v_recovered, v, atol=1e-6)


# SLERP tests
@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_endpoints(do_jit):
    """Test SLERP at endpoints t=0 and t=1."""

    def func(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        func = jax.jit(func)

    q1 = Quaternion(1.0)
    q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0).normalize()

    # Test t=0 (should return q1)
    result_0 = func(q1, q2, 0.0)
    assert jnp.allclose(result_0.wxyz, q1.wxyz, atol=1e-6)

    # Test t=1 (should return q2, up to sign)
    result_1 = func(q1, q2, 1.0)
    assert jnp.allclose(result_1.wxyz, q2.wxyz, atol=1e-6) or jnp.allclose(
        result_1.wxyz, -q2.wxyz, atol=1e-6
    )


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_midpoint(do_jit):
    """Test SLERP at midpoint t=0.5."""

    def func(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        func = jax.jit(func)

    q1 = Quaternion(1.0)
    q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0).normalize()

    result = func(q1, q2, 0.5)

    # At t=0.5, should be 45° rotation around x
    angle_45 = jnp.pi / 4
    expected = Quaternion(jnp.cos(angle_45 / 2), jnp.sin(angle_45 / 2), 0.0, 0.0)

    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-4) or jnp.allclose(
        result.wxyz, -expected.wxyz, atol=1e-4
    )


@pytest.mark.parametrize('t', [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_preserves_normalization(t, do_jit):
    """Test that SLERP result is always normalized."""

    def func(q1, q2):
        return q1.slerp(q2, t)

    if do_jit:
        func = jax.jit(func)

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    q1 = Quaternion.random(key1)
    q2 = Quaternion.random(key2)

    result = func(q1, q2)
    assert jnp.allclose(abs(result), 1.0, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_shortest_path(do_jit):
    """Test that SLERP takes the shortest path."""

    def func(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        func = jax.jit(func)

    # Test with quaternions that have negative dot product
    q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
    q2 = Quaternion(-0.7071, 0.7071, 0.0, 0.0)

    result = func(q1, q2, 0.5)

    # Should be normalized
    assert abs(result) > 0.9

    # The dot product between q1 and the result should be positive
    dot = jnp.sum(q1.wxyz * result.wxyz)
    assert dot > 0


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_identical_quaternions(do_jit):
    """Test SLERP with identical quaternions."""

    def func(q, t):
        return q.slerp(q, t)

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.random(jax.random.PRNGKey(42))
    result = func(q, 0.5)

    assert jnp.allclose(result.wxyz, q.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_batch(do_jit):
    """Test SLERP with batch of quaternions."""

    def func(q1_batch, q2_batch, t):
        return q1_batch.slerp(q2_batch, t)

    if do_jit:
        func = jax.jit(func)

    q1_batch = Quaternion.from_array(
        jnp.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    )

    q2_batch = Quaternion.from_array(
        jnp.array(
            [[0.7071, 0.7071, 0.0, 0.0], [0.7071, 0.0, 0.7071, 0.0], [0.7071, 0.0, 0.0, 0.7071]]
        )
    )

    result_batch = func(q1_batch, q2_batch, 0.5)

    assert result_batch.shape == (3,)
    assert jnp.allclose(abs(result_batch), 1.0, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_array_t(do_jit):
    """Test SLERP with array of t values."""

    def func(q1, q2, t_array):
        return q1.slerp(q2, t_array)

    if do_jit:
        func = jax.jit(func)

    q1 = Quaternion(1.0)
    q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0).normalize()

    t_array = jnp.array([0.0, 0.5, 1.0])
    result = func(q1, q2, t_array)

    assert result.shape == (3,)

    # Check endpoints
    assert jnp.allclose(result.wxyz[0], q1.wxyz, atol=1e-6)
    assert jnp.allclose(result.wxyz[2], q2.wxyz, atol=1e-6) or jnp.allclose(
        result.wxyz[2], -q2.wxyz, atol=1e-6
    )


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_close_quaternions(do_jit):
    """Test SLERP with very close quaternions (should use linear interpolation)."""

    def func(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        func = jax.jit(func)

    # Very close quaternions (dot product > 0.9995)
    q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
    q2 = Quaternion(0.9999, 0.001, 0.0, 0.0).normalize()

    result = func(q1, q2, 0.5)

    # Should still be normalized and reasonable
    assert jnp.allclose(abs(result), 1.0, atol=1e-6)

    # Result should be between the two quaternions
    dot1 = jnp.sum(q1.wxyz * result.wxyz)
    dot2 = jnp.sum(q2.wxyz * result.wxyz)
    assert dot1 > 0.9
    assert dot2 > 0.9
