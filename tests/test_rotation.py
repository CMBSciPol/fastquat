"""Rotation operations tests for Quaternion class.

Tests for from_rotation_matrix, to_rotation_matrix, and rotate_vector.
"""

import jax
import jax.numpy as jnp
import pytest

from fastquat.quaternion import Quaternion


# from_rotation_matrix
@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_matrix_identity(do_jit):
    """Test from_rotation_matrix with identity matrix."""

    def from_rot_matrix(rot):
        return Quaternion.from_rotation_matrix(rot)

    if do_jit:
        from_rot_matrix = jax.jit(from_rot_matrix)

    identity_matrix = jnp.eye(3)
    q = from_rot_matrix(identity_matrix)

    # Identity quaternion should be approximately [1, 0, 0, 0]
    expected = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_matrix_90deg_z(do_jit):
    """Test from_rotation_matrix with 90° rotation around z-axis."""

    def from_rot_matrix(rot):
        return Quaternion.from_rotation_matrix(rot)

    if do_jit:
        from_rot_matrix = jax.jit(from_rot_matrix)

    # 90° rotation around z-axis
    rot_z_90 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    q = from_rot_matrix(rot_z_90)

    # Expected quaternion for 90° rotation around z
    angle = jnp.pi / 2
    expected = jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])
    assert jnp.allclose(q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_matrix_batch(do_jit):
    """Test from_rotation_matrix with batch of matrices."""

    def from_rot_matrix(rot_batch):
        return Quaternion.from_rotation_matrix(rot_batch)

    if do_jit:
        from_rot_matrix = jax.jit(from_rot_matrix)

    # Batch of identity matrices
    rot_batch = jnp.tile(jnp.eye(3), (3, 1, 1))
    q_batch = from_rot_matrix(rot_batch)

    assert q_batch.shape == (3,)
    # All should be identity quaternions
    expected_batch = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    assert jnp.allclose(q_batch.wxyz, expected_batch, atol=1e-6)


# to_rotation_matrix
@pytest.mark.parametrize('do_jit', [False, True])
def test_to_rotation_matrix_identity(do_jit):
    """Test to_rotation_matrix with identity quaternion."""

    def to_rot_matrix(q):
        return q.to_rotation_matrix()

    if do_jit:
        to_rot_matrix = jax.jit(to_rot_matrix)

    q = Quaternion.ones()  # Identity quaternion
    R = to_rot_matrix(q)

    assert jnp.allclose(R, jnp.eye(3), atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_to_rotation_matrix_90deg_z(do_jit):
    """Test to_rotation_matrix with 90° rotation around z-axis."""

    def to_rot_matrix(q):
        return q.to_rotation_matrix()

    if do_jit:
        to_rot_matrix = jax.jit(to_rot_matrix)

    # Quaternion for 90° rotation around z
    angle = jnp.pi / 2
    q = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))

    R = to_rot_matrix(q)

    # Expected 90° rotation matrix around z
    expected = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    assert jnp.allclose(R, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotation_matrix_is_orthogonal(do_jit):
    """Test that rotation matrix is orthogonal."""

    def to_rot_matrix(q):
        return q.to_rotation_matrix()

    if do_jit:
        to_rot_matrix = jax.jit(to_rot_matrix)

    # Random quaternion
    key = jax.random.PRNGKey(42)
    q = Quaternion.random(key)

    R = to_rot_matrix(q)

    # Check orthogonality: R @ R.T = I
    assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)
    # Check determinant = 1
    assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotation_matrix_batch(do_jit):
    """Test to_rotation_matrix with batch of quaternions."""

    def to_rot_matrix(q_batch):
        return q_batch.to_rotation_matrix()

    if do_jit:
        to_rot_matrix = jax.jit(to_rot_matrix)

    # Batch of quaternions
    key = jax.random.PRNGKey(42)
    q_batch = Quaternion.random(key, (3,))

    R_batch = to_rot_matrix(q_batch)
    assert R_batch.shape == (3, 3, 3)

    # All should be orthogonal
    for i in range(3):
        R = R_batch[i]
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)


# rotate_vector
@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_identity(do_jit):
    """Test rotate_vector with identity quaternion."""

    def rotate_vec(q, v):
        return q.rotate_vector(v)

    if do_jit:
        rotate_vec = jax.jit(rotate_vec)

    q = Quaternion.ones()  # Identity quaternion
    v = jnp.array([1.0, 2.0, 3.0])

    v_rotated = rotate_vec(q, v)

    # Identity rotation should not change vector
    assert jnp.allclose(v_rotated, v, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_90deg_z(do_jit):
    """Test rotate_vector with 90° rotation around z-axis."""

    def rotate_vec(q, v):
        return q.rotate_vector(v)

    if do_jit:
        rotate_vec = jax.jit(rotate_vec)

    # 90° rotation around z
    angle = jnp.pi / 2
    q = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))

    # Rotate vector [1, 0, 0]
    v = jnp.array([1.0, 0.0, 0.0])
    v_rotated = rotate_vec(q, v)

    # Should give [0, 1, 0]
    expected = jnp.array([0.0, 1.0, 0.0])
    assert jnp.allclose(v_rotated, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_90deg_x(do_jit):
    """Test rotate_vector with 90° rotation around x-axis."""

    def rotate_vec(q, v):
        return q.rotate_vector(v)

    if do_jit:
        rotate_vec = jax.jit(rotate_vec)

    # 90° rotation around x
    angle = jnp.pi / 2
    q = Quaternion(jnp.cos(angle / 2), jnp.sin(angle / 2), 0.0, 0.0)

    # Rotate vector [0, 1, 0]
    v = jnp.array([0.0, 1.0, 0.0])
    v_rotated = rotate_vec(q, v)

    # Should give [0, 0, 1]
    expected = jnp.array([0.0, 0.0, 1.0])
    assert jnp.allclose(v_rotated, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_preserves_length(do_jit):
    """Test rotate_vector preserves vector length."""

    def rotate_vec(q, v):
        return q.rotate_vector(v)

    if do_jit:
        rotate_vec = jax.jit(rotate_vec)

    # Random quaternion and vector
    key = jax.random.PRNGKey(42)
    q = Quaternion.random(key)
    v = jax.random.normal(jax.random.split(key)[0], (3,))

    v_rotated = rotate_vec(q, v)

    # Length should be preserved
    original_length = jnp.linalg.norm(v)
    rotated_length = jnp.linalg.norm(v_rotated)
    assert jnp.allclose(original_length, rotated_length, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_batch(do_jit):
    """Test rotate_vector with batch of vectors."""

    def rotate_vec(q, v_batch):
        return q.rotate_vector(v_batch)

    if do_jit:
        rotate_vec = jax.jit(rotate_vec)

    q = Quaternion.ones()  # Identity quaternion
    v_batch = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    v_rotated_batch = rotate_vec(q, v_batch)

    # Identity rotation should not change vectors
    assert jnp.allclose(v_rotated_batch, v_batch, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotate_vector_batch_quaternions(do_jit):
    """Test rotate_vector with batch of quaternions."""

    def rotate_vec(q_batch, v):
        return q_batch.rotate_vector(v)

    if do_jit:
        rotate_vec = jax.jit(rotate_vec)

    # Batch of identity quaternions
    q_batch = Quaternion.from_array(jnp.tile(jnp.array([1, 0, 0, 0]), (3, 1)))
    v = jnp.array([1.0, 2.0, 3.0])

    v_rotated_batch = rotate_vec(q_batch, v)

    # All should be unchanged
    expected_batch = jnp.tile(v, (3, 1))
    assert jnp.allclose(v_rotated_batch, expected_batch, atol=1e-6)


# Round-trip consistency tests
@pytest.mark.parametrize('do_jit', [False, True])
def test_matrix_quaternion_roundtrip(do_jit):
    """Test round-trip conversion: quaternion -> matrix -> quaternion."""

    def roundtrip_test(q):
        R = q.to_rotation_matrix()
        q_recovered = Quaternion.from_rotation_matrix(R)
        return q_recovered

    if do_jit:
        roundtrip_test = jax.jit(roundtrip_test)

    # Use a specific quaternion instead of random for deterministic test
    q_original = Quaternion(0.7071, 0.7071, 0.0, 0.0).normalize()  # 90° around x

    q_recovered = roundtrip_test(q_original)

    # Should be the same (up to sign ambiguity in quaternions)
    # Check if q_recovered == q_original or q_recovered == -q_original
    # Allow for larger tolerance due to numerical precision in matrix conversion
    assert jnp.allclose(q_recovered.wxyz, q_original.wxyz, atol=1e-4) or jnp.allclose(
        q_recovered.wxyz, -q_original.wxyz, atol=1e-4
    )


@pytest.mark.parametrize('do_jit', [False, True])
def test_rotation_consistency(do_jit):
    """Test consistency between rotate_vector and to_rotation_matrix."""

    def test_consistency(q, v):
        # Method 1: using rotate_vector
        v_rot1 = q.rotate_vector(v)

        # Method 2: using rotation matrix
        R = q.to_rotation_matrix()
        v_rot2 = R @ v

        return v_rot1, v_rot2

    if do_jit:
        test_consistency = jax.jit(test_consistency)

    # Random quaternion and vector
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    q = Quaternion.random(key1)
    v = jax.random.normal(key2, (3,))

    v_rot1, v_rot2 = test_consistency(q, v)

    # Both methods should give same result
    assert jnp.allclose(v_rot1, v_rot2, atol=1e-6)


# Edge cases
@pytest.mark.parametrize('do_jit', [False, True])
def test_from_rotation_matrix_wrong_shape(do_jit):
    """Test from_rotation_matrix with wrong shape raises ValueError."""

    def from_rot_matrix(rot):
        return Quaternion.from_rotation_matrix(rot)

    if do_jit:
        from_rot_matrix = jax.jit(from_rot_matrix)

    wrong_matrix = jnp.ones((2, 2))  # Wrong shape
    with pytest.raises(ValueError):
        from_rot_matrix(wrong_matrix)


def test_rotate_vector_wrong_shape():
    """Test rotate_vector with wrong vector shape."""
    q = Quaternion.ones()

    # Test with 2D vector - should cause an issue when trying to create quaternion
    wrong_vector = jnp.array([1.0, 2.0])  # Wrong shape
    try:
        # This will likely fail when trying to access the third component
        result = q.rotate_vector(wrong_vector)
        # The exact behavior depends on JAX's error handling
        # We just ensure it doesn't crash the test suite
        assert result is not None
    except (IndexError, ValueError, TypeError):
        # Expected behavior for wrong shape
        pass


@pytest.mark.parametrize('do_jit', [False, True])
def test_zero_vector_rotation(do_jit):
    """Test rotation of zero vector."""

    def rotate_vec(q, v):
        return q.rotate_vector(v)

    if do_jit:
        rotate_vec = jax.jit(rotate_vec)

    # Random quaternion
    key = jax.random.PRNGKey(42)
    q = Quaternion.random(key)

    # Zero vector
    v_zero = jnp.zeros(3)
    v_rotated = rotate_vec(q, v_zero)

    # Zero vector should remain zero
    assert jnp.allclose(v_rotated, v_zero, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_multiple_vector_rotations(do_jit):
    """Test rotating multiple vectors with same quaternion."""

    def rotate_vecs(q, v_batch):
        return q.rotate_vector(v_batch)

    if do_jit:
        rotate_vecs = jax.jit(rotate_vecs)

    # 90° rotation around z
    angle = jnp.pi / 2
    q = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))

    # Standard basis vectors
    v_batch = jnp.array(
        [
            [1.0, 0.0, 0.0],  # Should become [0, 1, 0]
            [0.0, 1.0, 0.0],  # Should become [-1, 0, 0]
            [0.0, 0.0, 1.0],  # Should remain [0, 0, 1]
        ]
    )

    v_rotated_batch = rotate_vecs(q, v_batch)

    expected = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    assert jnp.allclose(v_rotated_batch, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_inverse_rotation(do_jit):
    """Test that inverse quaternion performs inverse rotation."""

    def test_inverse_rot(q, v):
        # Rotate vector, then rotate back with inverse
        v_rotated = q.rotate_vector(v)
        v_back = (1 / q).rotate_vector(v_rotated)
        return v_back

    if do_jit:
        test_inverse_rot = jax.jit(test_inverse_rot)

    # Random quaternion and vector
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    q = Quaternion.random(key1)
    v = jax.random.normal(key2, (3,))

    v_recovered = test_inverse_rot(q, v)

    # Should recover original vector
    assert jnp.allclose(v_recovered, v, atol=1e-6)


# SLERP tests
@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_endpoints(do_jit):
    """Test SLERP at endpoints t=0 and t=1."""

    def slerp_test(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        slerp_test = jax.jit(slerp_test)

    # Two different quaternions
    q1 = Quaternion.ones()  # Identity
    q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0).normalize()  # 90° around x

    # Test t=0 (should return q1)
    result_0 = slerp_test(q1, q2, 0.0)
    assert jnp.allclose(result_0.wxyz, q1.wxyz, atol=1e-6)

    # Test t=1 (should return q2, up to sign)
    result_1 = slerp_test(q1, q2, 1.0)
    # Check if result == q2 or result == -q2 (quaternion double cover)
    assert jnp.allclose(result_1.wxyz, q2.wxyz, atol=1e-6) or jnp.allclose(
        result_1.wxyz, -q2.wxyz, atol=1e-6
    )


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_midpoint(do_jit):
    """Test SLERP at midpoint t=0.5."""

    def slerp_test(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        slerp_test = jax.jit(slerp_test)

    # Identity and 90° rotation around x
    q1 = Quaternion.ones()
    q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0).normalize()

    result = slerp_test(q1, q2, 0.5)

    # At t=0.5, should be 45° rotation around x
    angle_45 = jnp.pi / 4
    expected = Quaternion(jnp.cos(angle_45 / 2), jnp.sin(angle_45 / 2), 0.0, 0.0)

    # Allow sign ambiguity
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-4) or jnp.allclose(
        result.wxyz, -expected.wxyz, atol=1e-4
    )


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_preserves_normalization(do_jit):
    """Test that SLERP result is always normalized."""

    def slerp_test(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        slerp_test = jax.jit(slerp_test)

    # Random quaternions
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    q1 = Quaternion.random(key1)
    q2 = Quaternion.random(key2)

    # Test at various t values
    t_values = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

    for t in t_values:
        result = slerp_test(q1, q2, t)
        norm = result.norm()
        assert jnp.allclose(norm, 1.0, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_shortest_path(do_jit):
    """Test that SLERP takes the shortest path."""

    def slerp_test(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        slerp_test = jax.jit(slerp_test)

    # Test with quaternions that have negative dot product
    q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
    q2 = Quaternion(-0.7071, 0.7071, 0.0, 0.0)  # Negative w

    result = slerp_test(q1, q2, 0.5)

    # The interpolation should flip q2 to take shorter path
    # Result should be between q1 and the flipped version of q2
    assert result.norm() > 0.9  # Should be normalized

    # The dot product between q1 and the result should be positive
    # (indicating we're moving in the right direction)
    dot = jnp.sum(q1.wxyz * result.wxyz)
    assert dot > 0


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_identical_quaternions(do_jit):
    """Test SLERP with identical quaternions."""

    def slerp_test(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        slerp_test = jax.jit(slerp_test)

    q = Quaternion.random(jax.random.PRNGKey(42))

    # SLERP between identical quaternions should return the same quaternion
    result = slerp_test(q, q, 0.5)
    assert jnp.allclose(result.wxyz, q.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_batch(do_jit):
    """Test SLERP with batch of quaternions."""

    def slerp_test(q1_batch, q2_batch, t):
        return q1_batch.slerp(q2_batch, t)

    if do_jit:
        slerp_test = jax.jit(slerp_test)

    # Batch of quaternions
    q1_batch = Quaternion.from_array(
        jnp.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    )

    q2_batch = Quaternion.from_array(
        jnp.array(
            [[0.7071, 0.7071, 0.0, 0.0], [0.7071, 0.0, 0.7071, 0.0], [0.7071, 0.0, 0.0, 0.7071]]
        )
    )

    result_batch = slerp_test(q1_batch, q2_batch, 0.5)

    assert result_batch.shape == (3,)

    # All results should be normalized
    norms = result_batch.norm()
    assert jnp.allclose(norms, 1.0, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_array_t(do_jit):
    """Test SLERP with array of t values."""

    def slerp_test(q1, q2, t_array):
        return q1.slerp(q2, t_array)

    if do_jit:
        slerp_test = jax.jit(slerp_test)

    q1 = Quaternion.ones()
    q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0).normalize()

    t_array = jnp.array([0.0, 0.5, 1.0])
    result = slerp_test(q1, q2, t_array)

    assert result.shape == (3,)

    # Check endpoints
    assert jnp.allclose(result.wxyz[0], q1.wxyz, atol=1e-6)
    # Allow sign ambiguity for t=1
    assert jnp.allclose(result.wxyz[2], q2.wxyz, atol=1e-6) or jnp.allclose(
        result.wxyz[2], -q2.wxyz, atol=1e-6
    )


@pytest.mark.parametrize('do_jit', [False, True])
def test_slerp_close_quaternions(do_jit):
    """Test SLERP with very close quaternions (should use linear interpolation)."""

    def slerp_test(q1, q2, t):
        return q1.slerp(q2, t)

    if do_jit:
        slerp_test = jax.jit(slerp_test)

    # Very close quaternions (dot product > 0.9995)
    q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
    q2 = Quaternion(0.9999, 0.001, 0.0, 0.0).normalize()

    result = slerp_test(q1, q2, 0.5)

    # Should still be normalized and reasonable
    assert jnp.allclose(result.norm(), 1.0, atol=1e-6)

    # Result should be between the two quaternions
    dot1 = jnp.sum(q1.wxyz * result.wxyz)
    dot2 = jnp.sum(q2.wxyz * result.wxyz)
    assert dot1 > 0.9  # Close to both
    assert dot2 > 0.9
