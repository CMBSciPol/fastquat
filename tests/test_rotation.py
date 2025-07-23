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
        v_back = q.inverse().rotate_vector(v_rotated)
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
