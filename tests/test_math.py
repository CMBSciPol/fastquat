"""Mathematical operations tests for Quaternion class.

Tests for addition, subtraction, multiplication, negation, conjugate, norm, normalize, inverse.
"""

import jax
import jax.numpy as jnp
import pytest

from fastquat.quaternion import Quaternion


# Addition
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_addition(do_jit):
    """Test quaternion addition."""

    def add_quaternions(q1, q2):
        return q1 + q2

    if do_jit:
        add_quaternions = jax.jit(add_quaternions)

    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, 1.5, 2.5, 3.5)
    result = add_quaternions(q1, q2)
    expected = jnp.array([1.5, 3.5, 5.5, 7.5])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_scalar_addition(do_jit):
    """Test quaternion addition with scalar."""

    def add_scalar(q, scalar):
        return q + scalar

    if do_jit:
        add_scalar = jax.jit(add_scalar)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    scalar = 2.5
    result = add_scalar(q, scalar)
    expected = jnp.array([3.5, 2.0, 3.0, 4.0])  # Only w component affected
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_scalar_quaternion_addition(do_jit):
    """Test scalar addition with quaternion (right addition)."""

    def add_scalar(scalar, q):
        return scalar + q

    if do_jit:
        add_scalar = jax.jit(add_scalar)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    scalar = 2.5
    result = add_scalar(scalar, q)
    expected = jnp.array([3.5, 2.0, 3.0, 4.0])  # Only w component affected
    assert jnp.allclose(result.wxyz, expected)


# Subtraction
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_subtraction(do_jit):
    """Test quaternion subtraction."""

    def sub_quaternions(q1, q2):
        return q1 - q2

    if do_jit:
        sub_quaternions = jax.jit(sub_quaternions)

    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, 1.5, 2.5, 3.5)
    result = sub_quaternions(q1, q2)
    expected = jnp.array([0.5, 0.5, 0.5, 0.5])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_scalar_subtraction(do_jit):
    """Test quaternion subtraction with scalar."""

    def sub_scalar(q, scalar):
        return q - scalar

    if do_jit:
        sub_scalar = jax.jit(sub_scalar)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    scalar = 2.5
    result = sub_scalar(q, scalar)
    expected = jnp.array([-1.5, 2.0, 3.0, 4.0])  # Only w component affected
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_scalar_quaternion_subtraction(do_jit):
    """Test scalar subtraction with quaternion (right subtraction)."""

    def sub_scalar(scalar, q):
        return scalar - q

    if do_jit:
        sub_scalar = jax.jit(sub_scalar)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    scalar = 2.5
    result = sub_scalar(scalar, q)
    expected = jnp.array([1.5, -2.0, -3.0, -4.0])  # w = scalar - w, vector negated
    assert jnp.allclose(result.wxyz, expected)


# Multiplication
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_multiplication(do_jit):
    """Test quaternion multiplication."""

    def mul_quaternions(q1, q2):
        return q1 * q2

    if do_jit:
        mul_quaternions = jax.jit(mul_quaternions)

    # Identity quaternion
    q1 = Quaternion.ones()
    q2 = Quaternion(0.0, 1.0, 0.0, 0.0)

    result = mul_quaternions(q1, q2)
    expected = jnp.array([0.0, 1.0, 0.0, 0.0])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_unit_multiplication(do_jit):
    """Test quaternion unit multiplication: i * j = k."""

    def mul_quaternions(qi, qj):
        return qi * qj

    if do_jit:
        mul_quaternions = jax.jit(mul_quaternions)

    qi = Quaternion(0.0, 1.0, 0.0, 0.0)  # i
    qj = Quaternion(0.0, 0.0, 1.0, 0.0)  # j
    qk = mul_quaternions(qi, qj)
    expected_k = jnp.array([0.0, 0.0, 0.0, 1.0])  # k
    assert jnp.allclose(qk.wxyz, expected_k)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_scalar_multiplication(do_jit):
    """Test quaternion multiplication by scalar."""

    def mul_scalar(q, scalar):
        return q * scalar

    if do_jit:
        mul_scalar = jax.jit(mul_scalar)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    scalar = 2.5
    result = mul_scalar(q, scalar)
    expected = jnp.array([2.5, 5.0, 7.5, 10.0])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_scalar_quaternion_multiplication(do_jit):
    """Test scalar multiplication with quaternion (right multiplication)."""

    def mul_scalar(scalar, q):
        return scalar * q

    if do_jit:
        mul_scalar = jax.jit(mul_scalar)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    scalar = 2.5
    result = mul_scalar(scalar, q)
    expected = jnp.array([2.5, 5.0, 7.5, 10.0])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_array_multiplication(do_jit):
    """Test quaternion multiplication by array."""

    def mul_array(q, arr):
        return q * arr

    if do_jit:
        mul_array = jax.jit(mul_array)

    q = Quaternion.from_array(jnp.ones((2, 4)))
    arr = jnp.array([2.0, 3.0])
    result = mul_array(q, arr)
    expected = jnp.array([[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]])
    assert jnp.allclose(result.wxyz, expected)


# Division
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_division(do_jit):
    """Test quaternion division."""

    def div_quaternions(q1, q2):
        return q1 / q2

    if do_jit:
        div_quaternions = jax.jit(div_quaternions)

    # Identity quaternion
    q1 = Quaternion.ones()
    q2 = Quaternion(0.0, 1.0, 0.0, 0.0)

    result = div_quaternions(q1, q2)
    # q1 / q2 should equal q1 * (1 / q2)
    expected = q1 * (1 / q2)
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_scalar_division(do_jit):
    """Test quaternion division by scalar."""

    def div_scalar(q, scalar):
        return q / scalar

    if do_jit:
        div_scalar = jax.jit(div_scalar)

    q = Quaternion(2.0, 4.0, 6.0, 8.0)
    scalar = 2.0
    result = div_scalar(q, scalar)
    expected = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_array_division(do_jit):
    """Test quaternion division by array."""

    def div_array(q, arr):
        return q / arr

    if do_jit:
        div_array = jax.jit(div_array)

    q = Quaternion.from_array(jnp.array([[2.0, 4.0, 6.0, 8.0], [3.0, 6.0, 9.0, 12.0]]))
    arr = jnp.array([2.0, 3.0])
    result = div_array(q, arr)
    expected = jnp.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_division_identity_property(do_jit):
    """Test that q / q approximates identity quaternion."""

    def div_self(q):
        return q / q

    if do_jit:
        div_self = jax.jit(div_self)

    # Test with normalized quaternion
    q_norm = Quaternion(1.0, 2.0, 3.0, 4.0).normalize()
    result_norm = div_self(q_norm)
    expected_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(result_norm.wxyz, expected_identity, atol=1e-6)

    # Test with non-normalized quaternion
    q = Quaternion(2.0, 4.0, 6.0, 8.0)
    result = div_self(q)
    assert jnp.allclose(result.wxyz, expected_identity, atol=1e-6)


@pytest.mark.parametrize('scalar', [1, 1.0])
@pytest.mark.parametrize('do_jit', [False, True])
def test_one_quaternion_division(scalar: int | float, do_jit: bool):
    """Test scalar division by quaternion."""

    def div_scalar(scalar, q):
        return scalar / q

    if do_jit:
        div_scalar = jax.jit(div_scalar)

    q = Quaternion(1.0, 0.0, 0.0, 0.0)  # Real quaternion
    result = div_scalar(scalar, q)
    # scalar / q should equal scalar * (1 / q)
    expected = scalar * q._inverse()
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_scalar_quaternion_division(do_jit):
    """Test scalar division by quaternion."""

    def div_scalar(scalar, q):
        return scalar / q

    if do_jit:
        div_scalar = jax.jit(div_scalar)

    q = Quaternion(1.0, 0.0, 0.0, 0.0)  # Real quaternion
    scalar = 2.0
    result = div_scalar(scalar, q)
    # scalar / q should equal scalar * (1 / q)
    expected = scalar * q._inverse()
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_array_quaternion_division(do_jit):
    """Test array division by quaternion."""

    def div_array(arr, q):
        return arr / q

    if do_jit:
        div_array = jax.jit(div_array)

    # Test with real quaternion for simplicity
    q = Quaternion(2.0, 0.0, 0.0, 0.0)
    arr = jnp.array([4.0, 6.0])
    result = div_array(arr, q)

    # arr / q should equal arr * (1 / q)
    expected = arr * (1 / q)
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_division_multiplication_inverse(do_jit):
    """Test that (q1 / q2) * q2 ≈ q1."""

    def div_mul_inverse(q1, q2):
        return (q1 / q2) * q2

    if do_jit:
        div_mul_inverse = jax.jit(div_mul_inverse)

    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, 1.5, 2.5, 3.5)

    result = div_mul_inverse(q1, q2)
    assert jnp.allclose(result.wxyz, q1.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_division_by_identity(do_jit):
    """Test division by identity quaternion."""

    def div_identity(q):
        identity = Quaternion.ones()
        return q / identity

    if do_jit:
        div_identity = jax.jit(div_identity)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = div_identity(q)

    # q / identity should equal q
    assert jnp.allclose(result.wxyz, q.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_division_associativity_property(do_jit):
    """Test division associativity: (q1 / q2) / q3 = q1 / (q2 * q3)."""

    def div_associativity(q1, q2, q3):
        left = (q1 / q2) / q3
        right = q1 / (q3 * q2)
        return left, right

    if do_jit:
        div_associativity = jax.jit(div_associativity)

    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, 1.0, 1.5, 2.0)
    q3 = Quaternion(2.0, 1.0, 0.5, 1.5)

    left, right = div_associativity(q1, q2, q3)
    assert jnp.allclose(left.wxyz, right.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_division_batch_operations(do_jit):
    """Test division with batch of quaternions."""

    def batch_div(q1_batch, q2_batch):
        return q1_batch / q2_batch

    if do_jit:
        batch_div = jax.jit(batch_div)

    # Create batches of quaternions
    q1_batch = Quaternion.from_array(
        jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    )
    q2_batch = Quaternion.from_array(
        jnp.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]])
    )

    result = batch_div(q1_batch, q2_batch)

    # Check each result individually
    for i in range(3):
        q1 = Quaternion.from_array(q1_batch.wxyz[i])
        q2 = Quaternion.from_array(q2_batch.wxyz[i])
        expected = q1 / q2
        assert jnp.allclose(result.wxyz[i], expected.wxyz, atol=1e-6)


# Negation
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_negation(do_jit):
    """Test quaternion negation."""

    def negate_quaternion(q):
        return -q

    if do_jit:
        negate_quaternion = jax.jit(negate_quaternion)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = negate_quaternion(q)
    expected = jnp.array([-1.0, -2.0, -3.0, -4.0])
    assert jnp.allclose(result.wxyz, expected)


# Conjugate
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_conjugate(do_jit):
    """Test quaternion conjugate."""

    def conjugate_quaternion(q):
        return q.conj()

    if do_jit:
        conjugate_quaternion = jax.jit(conjugate_quaternion)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    conj_q = conjugate_quaternion(q)
    expected = jnp.array([1.0, -2.0, -3.0, -4.0])
    assert jnp.allclose(conj_q.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_conjugate_alias(do_jit):
    """Test quaternion conjugate alias."""

    def conjugate_quaternion(q):
        return q.conjugate()

    if do_jit:
        conjugate_quaternion = jax.jit(conjugate_quaternion)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    conj_q = conjugate_quaternion(q)
    expected = jnp.array([1.0, -2.0, -3.0, -4.0])
    assert jnp.allclose(conj_q.wxyz, expected)


# Norm
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_norm(do_jit):
    """Test quaternion norm calculation."""

    def get_norm(q):
        return q.norm()

    if do_jit:
        get_norm = jax.jit(get_norm)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    norm = get_norm(q)
    expected = jnp.sqrt(1 + 4 + 9 + 16)  # sqrt(30)
    assert jnp.allclose(norm, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_unit_quaternion_norm(do_jit):
    """Test unit quaternion norm is 1."""

    def get_norm(q):
        return q.norm()

    if do_jit:
        get_norm = jax.jit(get_norm)

    q = Quaternion.ones()  # Identity quaternion
    norm = get_norm(q)
    assert jnp.allclose(norm, 1.0)


# Normalize
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_normalize(do_jit):
    """Test quaternion normalization."""

    def normalize_quaternion(q):
        return q.normalize()

    if do_jit:
        normalize_quaternion = jax.jit(normalize_quaternion)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    q_norm = normalize_quaternion(q)
    assert jnp.allclose(q_norm.norm(), 1.0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_normalize_preserves_direction(do_jit):
    """Test normalization preserves quaternion direction."""

    def normalize_quaternion(q):
        return q.normalize()

    if do_jit:
        normalize_quaternion = jax.jit(normalize_quaternion)

    q = Quaternion(2.0, 4.0, 6.0, 8.0)  # 2 * (1, 2, 3, 4)
    q_norm = normalize_quaternion(q)

    # Direction should be same as (1, 2, 3, 4) normalized
    expected_direction = jnp.array([1.0, 2.0, 3.0, 4.0]) / jnp.sqrt(30)
    assert jnp.allclose(q_norm.wxyz, expected_direction)


@pytest.mark.parametrize('do_jit', [False, True])
def test_normalize_zero_quaternion(do_jit):
    """Test normalization of zero quaternion returns identity."""

    def normalize_quaternion(q):
        return q.normalize()

    if do_jit:
        normalize_quaternion = jax.jit(normalize_quaternion)

    # Zero quaternion
    q_zero = Quaternion(0.0, 0.0, 0.0, 0.0)
    q_norm = normalize_quaternion(q_zero)

    # Should return identity quaternion
    expected_identity = jnp.array([0.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(q_norm.wxyz, expected_identity)


@pytest.mark.parametrize('do_jit', [False, True])
def test_tensor_normalize_with_zero(do_jit):
    """Test normalization of batch containing zero quaternions."""

    def normalize_tensor(q):
        return q.normalize()

    if do_jit:
        normalize_tensor = jax.jit(normalize_tensor)

    # Batch with mix of normal and zero quaternions
    q_tensor = Quaternion.from_array(
        jnp.array(
            [
                [1.0, 2.0, 3.0, 4.0],  # Normal quaternion
                [0.0, 0.0, 0.0, 0.0],  # Zero quaternion
                [2.0, 0.0, 0.0, 0.0],  # Real quaternion
            ]
        )
    )

    q_norm_tensor = normalize_tensor(q_tensor)

    # Check individual results
    # First quaternion should be normalized normally
    expected_first = jnp.array([1.0, 2.0, 3.0, 4.0]) / jnp.sqrt(30)
    assert jnp.allclose(q_norm_tensor.wxyz[0], expected_first)

    # Second quaternion (zero) should become zero
    expected_zero = jnp.array([0.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(q_norm_tensor.wxyz[1], expected_zero)

    # Third quaternion should be normalized normally
    expected_third = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(q_norm_tensor.wxyz[2], expected_third)


# Inverse
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_inverse(do_jit):
    """Test quaternion inverse."""

    def get_inverse(q):
        return 1 / q

    if do_jit:
        get_inverse = jax.jit(get_inverse)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    q_inv = get_inverse(q)

    # Check q * q_inv = identity (approximately)
    identity = q * q_inv
    expected_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(identity.wxyz, expected_identity, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_unit_quaternion_inverse(do_jit):
    """Test unit quaternion inverse equals conjugate."""

    def get_inverse_and_conj(q):
        return 1 / q, q.conj()

    if do_jit:
        get_inverse_and_conj = jax.jit(get_inverse_and_conj)

    # Normalized quaternion
    q = Quaternion(1.0, 2.0, 3.0, 4.0).normalize()
    q_inv, q_conj = get_inverse_and_conj(q)

    assert jnp.allclose(q_inv.wxyz, q_conj.wxyz, atol=1e-6)


# Batch operations
@pytest.mark.parametrize('do_jit', [False, True])
def test_tensor_addition(do_jit):
    """Test batch quaternion addition."""

    def batch_add(q1_batch, q2_batch):
        return q1_batch + q2_batch

    if do_jit:
        batch_add = jax.jit(batch_add)

    q1_batch = Quaternion.from_array(jnp.ones((3, 4)))
    q2_batch = Quaternion.from_array(2 * jnp.ones((3, 4)))
    result = batch_add(q1_batch, q2_batch)
    expected = 3 * jnp.ones((3, 4))
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_tensor_multiplication(do_jit):
    """Test batch quaternion multiplication."""

    def tensor_mul(q1, q2):
        return q1 * q2

    if do_jit:
        tensor_mul = jax.jit(tensor_mul)

    # Create identity quaternions batch
    q1_batch = Quaternion.from_array(jnp.tile(jnp.array([1, 0, 0, 0]), (3, 1)))
    q2_batch = Quaternion.from_array(jnp.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    result = tensor_mul(q1_batch, q2_batch)
    assert jnp.allclose(result.wxyz, q2_batch.wxyz)


@pytest.mark.parametrize('do_jit', [False, True])
def test_tensor_normalization(do_jit):
    """Test batch quaternion normalization."""

    def tensor_normalize(q_batch):
        return q_batch.normalize()

    if do_jit:
        tensor_normalize = jax.jit(tensor_normalize)

    # Create random batch
    key = jax.random.PRNGKey(42)
    q_batch = Quaternion.random(key, (5, 3))
    q_norm = tensor_normalize(q_batch)

    # All should have norm 1
    norms = q_norm.norm()
    assert jnp.allclose(norms, 1.0)


# Edge cases
@pytest.mark.parametrize('do_jit', [False, True])
def test_zero_quaternion_operations(do_jit):
    """Test operations with zero quaternion."""

    def zero_ops(q_zero, q_normal):
        return q_zero + q_normal, q_zero * q_normal, -q_zero

    if do_jit:
        zero_ops = jax.jit(zero_ops)

    q_zero = Quaternion(0.0, 0.0, 0.0, 0.0)
    q_normal = Quaternion(1.0, 2.0, 3.0, 4.0)

    add_result, mul_result, neg_result = zero_ops(q_zero, q_normal)

    # Addition: 0 + q = q
    assert jnp.allclose(add_result.wxyz, q_normal.wxyz)
    # Multiplication: 0 * q = 0
    assert jnp.allclose(mul_result.wxyz, jnp.zeros(4))
    # Negation: -0 = 0
    assert jnp.allclose(neg_result.wxyz, jnp.zeros(4))


@pytest.mark.parametrize('do_jit', [False, True])
def test_broadcasting_operations(do_jit):
    """Test operations with broadcasting."""

    def broadcast_ops(q_scalar, q_batch):
        return q_scalar + q_batch, q_scalar * q_batch

    if do_jit:
        broadcast_ops = jax.jit(broadcast_ops)

    q_scalar = Quaternion.ones()
    q_batch = Quaternion.from_array(jnp.ones((3, 4)))  # [1, 1, 1, 1] x 3

    add_result, mul_result = broadcast_ops(q_scalar, q_batch)

    # Should broadcast correctly
    assert add_result.shape == (3,)
    assert mul_result.shape == (3,)

    expected_add = jnp.array([[2.0, 1.0, 1.0, 1.0]] * 3)
    expected_mul = jnp.array([[1.0, 1.0, 1.0, 1.0]] * 3)

    assert jnp.allclose(add_result.wxyz, expected_add)
    assert jnp.allclose(mul_result.wxyz, expected_mul)


@pytest.mark.parametrize('do_jit', [False, True])
def test_complex_mathematical_expression(do_jit):
    """Test complex mathematical expression."""

    def complex_expr(q1, q2, scalar):
        # (q1 * q2).normalize() + scalar * q1.conj() - q2
        product = q1 * q2
        normalized = product.normalize()
        scaled_conj = scalar * q1.conj()
        return normalized + scaled_conj - q2

    if do_jit:
        complex_expr = jax.jit(complex_expr)

    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, 1.5, 2.5, 3.5)
    scalar = 1.5

    result = complex_expr(q1, q2, scalar)

    # Verify the computation step by step
    product = q1 * q2
    normalized = product.normalize()
    scaled_conj = scalar * q1.conj()
    expected = normalized + scaled_conj - q2

    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


# Logarithm and Exponential operations
@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_log_exp_inverse(do_jit):
    """Test that exp(log(q)) = q for general quaternions."""

    def log_exp_roundtrip(q):
        return q.log().exp()

    if do_jit:
        log_exp_roundtrip = jax.jit(log_exp_roundtrip)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = log_exp_roundtrip(q)
    assert jnp.allclose(result.wxyz, q.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_log_real(do_jit):
    """Test logarithm of real quaternion."""

    def log_real(q):
        return q.log()

    if do_jit:
        log_real = jax.jit(log_real)

    # Real quaternion (no vector part)
    q_real = Quaternion(2.0, 0.0, 0.0, 0.0)
    log_q = log_real(q_real)

    # log(2) should be (log(2), 0, 0, 0)
    expected = jnp.array([jnp.log(2.0), 0.0, 0.0, 0.0])
    assert jnp.allclose(log_q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_log_unit(do_jit):
    """Test logarithm of unit quaternion."""

    def log_unit(q):
        return q.log()

    if do_jit:
        log_unit = jax.jit(log_unit)

    # Unit quaternion representing 90° rotation around z
    angle = jnp.pi / 2
    q_unit = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))
    log_q = log_unit(q_unit)

    # log should be (0, 0, 0, π/2) for 90° rotation around z
    expected = jnp.array([0.0, 0.0, 0.0, angle / 2])
    assert jnp.allclose(log_q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_exp_real(do_jit):
    """Test exponential of real quaternion."""

    def exp_real(q):
        return q.exp()

    if do_jit:
        exp_real = jax.jit(exp_real)

    # Real quaternion
    q_real = Quaternion(1.5, 0.0, 0.0, 0.0)
    exp_q = exp_real(q_real)

    # exp(1.5) should be (exp(1.5), 0, 0, 0)
    expected = jnp.array([jnp.exp(1.5), 0.0, 0.0, 0.0])
    assert jnp.allclose(exp_q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_exp_pure_imaginary(do_jit):
    """Test exponential of pure imaginary quaternion."""

    def exp_pure(q):
        return q.exp()

    if do_jit:
        exp_pure = jax.jit(exp_pure)

    # Pure imaginary quaternion (0, 0, 0, π/2)
    angle = jnp.pi / 2
    q_pure = Quaternion(0.0, 0.0, 0.0, angle / 2)
    exp_q = exp_pure(q_pure)

    # Should give unit quaternion for 90° rotation around z
    expected = jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])
    assert jnp.allclose(exp_q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_log_exp_property(do_jit):
    """Test exp(log(q)) = q property for various quaternions."""

    def exp_log_property(q):
        return q.log().exp()

    if do_jit:
        exp_log_property = jax.jit(exp_log_property)

    # Use small rotation quaternions (more stable for log/exp)
    angle = 0.1  # Small angle
    q1 = Quaternion(jnp.cos(angle / 2), jnp.sin(angle / 2), 0.0, 0.0)
    q2 = Quaternion(jnp.cos(angle / 2), 0.0, jnp.sin(angle / 2), 0.0)

    result1 = exp_log_property(q1)
    result2 = exp_log_property(q2)

    # exp(log(q)) should equal q
    assert jnp.allclose(result1.wxyz, q1.wxyz, atol=1e-5)
    assert jnp.allclose(result2.wxyz, q2.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_log_exp_batch(do_jit):
    """Test log and exp with batch of quaternions."""

    def log_exp_batch(q_batch):
        return q_batch.log().exp()

    if do_jit:
        log_exp_batch = jax.jit(log_exp_batch)

    # Batch of normalized quaternions for better numerical stability
    q_batch = Quaternion.from_array(
        jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # Identity
                [0.707, 0.707, 0.0, 0.0],  # 90° around x
                [0.5, 0.5, 0.5, 0.5],  # Equal components, normalized
            ]
        )
    )
    q_batch = q_batch.normalize()  # Ensure normalization

    result = log_exp_batch(q_batch)
    # Use looser tolerance for numerical stability
    assert jnp.allclose(result.wxyz, q_batch.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_quaternion_log_zero_handling(do_jit):
    """Test log handles zero quaternion gracefully."""

    def log_zero(q):
        return q.log()

    if do_jit:
        log_zero = jax.jit(log_zero)

    # Zero quaternion - should not crash
    q_zero = Quaternion(0.0, 0.0, 0.0, 0.0)
    log_q = log_zero(q_zero)

    # Result should be finite (not NaN or inf)
    assert jnp.all(jnp.isfinite(log_q.wxyz))


# Power operations
@pytest.mark.parametrize('do_jit', [False, True])
def test_power_special_case_neg_two(do_jit):
    """Test quaternion power with exponent -2."""

    def power_neg_two(q):
        return q**-2

    if do_jit:
        power_neg_two = jax.jit(power_neg_two)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = power_neg_two(q)

    # Should equal (q^-1)^2
    expected = (1 / q) * (1 / q)
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_special_case_neg_one(do_jit):
    """Test quaternion power with exponent -1."""

    def power_neg_one(q):
        return q**-1

    if do_jit:
        power_neg_one = jax.jit(power_neg_one)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = power_neg_one(q)

    # Should equal 1 / q
    expected = 1 / q
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_special_case_zero(do_jit):
    """Test quaternion power with exponent 0."""

    def power_zero(q):
        return q**0

    if do_jit:
        power_zero = jax.jit(power_zero)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = power_zero(q)

    # Should equal identity quaternion
    expected = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(result.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_special_case_one(do_jit):
    """Test quaternion power with exponent 1."""

    def power_one(q):
        return q**1

    if do_jit:
        power_one = jax.jit(power_one)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = power_one(q)

    # Should equal q itself
    assert jnp.allclose(result.wxyz, q.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_special_case_two(do_jit):
    """Test quaternion power with exponent 2."""

    def power_two(q):
        return q**2

    if do_jit:
        power_two = jax.jit(power_two)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = power_two(q)

    # Should equal q * q
    expected = q * q
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_integer_positive(do_jit):
    """Test quaternion power with positive integer exponents."""

    def power_integer(q, n):
        return q**n

    if do_jit:
        power_integer = jax.jit(power_integer)

    q = Quaternion(0.5, 0.5, 0.5, 0.5).normalize()  # Unit quaternion

    # Test q^3
    result_3 = power_integer(q, 3)
    expected_3 = q * q * q
    assert jnp.allclose(result_3.wxyz, expected_3.wxyz, atol=1e-5)

    # Test q^4
    result_4 = power_integer(q, 4)
    expected_4 = q * q * q * q
    assert jnp.allclose(result_4.wxyz, expected_4.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_integer_negative(do_jit):
    """Test quaternion power with negative integer exponents."""

    def power_integer(q, n):
        return q**n

    if do_jit:
        power_integer = jax.jit(power_integer)

    q = Quaternion(1.0, 1.0, 1.0, 1.0).normalize()  # Unit quaternion

    # Test q^-3
    result_neg3 = power_integer(q, -3)
    q_inv = 1 / q
    expected_neg3 = q_inv * q_inv * q_inv
    assert jnp.allclose(result_neg3.wxyz, expected_neg3.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_fractional(do_jit):
    """Test quaternion power with fractional exponents."""

    def power_fractional(q, n):
        return q**n

    if do_jit:
        power_fractional = jax.jit(power_fractional)

    # Use a rotation quaternion (90° around z-axis)
    angle = jnp.pi / 2
    q = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))

    # Test q^0.5 (should be 45° rotation around z)
    result_half = power_fractional(q, 0.5)
    expected_angle = angle / 2
    expected_half = Quaternion(jnp.cos(expected_angle / 2), 0.0, 0.0, jnp.sin(expected_angle / 2))
    assert jnp.allclose(result_half.wxyz, expected_half.wxyz, atol=1e-5)

    # Verify (q^0.5)^2 ≈ q
    double_half = result_half * result_half
    assert jnp.allclose(double_half.wxyz, q.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_properties(do_jit):
    """Test mathematical properties of quaternion powers."""

    def power_test(q, a, b):
        return q**a, q**b, q ** (a * b), (q**a) ** b

    if do_jit:
        power_test = jax.jit(power_test)

    q = Quaternion(0.6, 0.8, 0.0, 0.0).normalize()  # Unit quaternion
    a, b = 2.0, 1.5

    q_a, q_b, q_ab, q_a_b = power_test(q, a, b)

    # Test (q^a)^b ≈ q^(a*b) for unit quaternions
    assert jnp.allclose(q_a_b.wxyz, q_ab.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_unit_quaternion(do_jit):
    """Test power operations preserve unit quaternion properties."""

    def power_unit(q, n):
        return q**n

    if do_jit:
        power_unit = jax.jit(power_unit)

    # Unit quaternion
    q = Quaternion(0.5, 0.5, 0.5, 0.5).normalize()

    # Powers of unit quaternions should remain unit quaternions (except for n=0)
    for n in [0.5, 1.5, 2.0, -0.5, -1.5]:
        result = power_unit(q, n)
        if n != 0:  # q^0 = identity, which has norm 1 but isn't the same quaternion
            assert jnp.allclose(result.norm(), 1.0, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_batch(do_jit):
    """Test quaternion power with batch of quaternions."""

    def power_batch(q_batch, n):
        return q_batch**n

    if do_jit:
        power_batch = jax.jit(power_batch)

    # Batch of quaternions
    q_batch = Quaternion.from_array(
        jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    )

    # Test with exponent 2
    result = power_batch(q_batch, 2)
    expected = q_batch * q_batch
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_array_exponent(do_jit):
    """Test quaternion power with array of exponents."""

    def power_array_exp(q, n_array):
        return q**n_array

    if do_jit:
        power_array_exp = jax.jit(power_array_exp)

    q = Quaternion(1.0, 1.0, 0.0, 0.0).normalize()
    n_array = jnp.array([0.0, 1.0, 2.0])

    result = power_array_exp(q, n_array)

    # Check each result
    assert jnp.allclose(result.wxyz[0], jnp.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)  # q^0
    assert jnp.allclose(result.wxyz[1], q.wxyz, atol=1e-6)  # q^1
    assert jnp.allclose(result.wxyz[2], (q * q).wxyz, atol=1e-6)  # q^2


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_edge_cases(do_jit):
    """Test quaternion power edge cases."""

    def power_edge(q, n):
        return q**n

    if do_jit:
        power_edge = jax.jit(power_edge)

    # Test with identity quaternion
    identity = Quaternion.ones()
    result = power_edge(identity, 3.5)
    assert jnp.allclose(result.wxyz, identity.wxyz, atol=1e-6)

    # Test with real quaternion (no vector part)
    real_q = Quaternion(2.0, 0.0, 0.0, 0.0)
    result = power_edge(real_q, 2.0)
    expected = Quaternion(4.0, 0.0, 0.0, 0.0)  # 2^2 = 4
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)

    # Test with pure imaginary quaternion
    pure_q = Quaternion(0.0, 1.0, 0.0, 0.0)
    result = power_edge(pure_q, 2.0)
    expected = Quaternion(-1.0, 0.0, 0.0, 0.0)  # i^2 = -1
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_consistency(do_jit):
    """Test consistency between special cases and general formula."""

    def power_test(q, n):
        return q**n

    if do_jit:
        power_test = jax.jit(power_test)

    q = Quaternion(0.8, 0.6, 0.0, 0.0).normalize()

    # Test that special cases give same results as would be computed generally
    # We use slightly offset values to avoid exact special case detection
    result_near_2 = power_test(q, 2.000001)
    result_exact_2 = power_test(q, 2.0)

    # Should be very close
    assert jnp.allclose(result_near_2.wxyz, result_exact_2.wxyz, atol=1e-4)


@pytest.mark.parametrize('do_jit', [False, True])
def test_power_zero_quaternion(do_jit):
    """Test power operations with zero quaternion."""

    def power_zero_q(q, n):
        return q**n

    if do_jit:
        power_zero_q = jax.jit(power_zero_q)

    zero_q = Quaternion(0.0, 0.0, 0.0, 0.0)

    # 0^0 should give 1 (by convention)
    result_0_0 = power_zero_q(zero_q, 0.0)
    expected_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(result_0_0.wxyz, expected_identity, atol=1e-6)

    # 0^n for n > 0 should give 0
    result_0_2 = power_zero_q(zero_q, 2.0)
    assert jnp.allclose(result_0_2.wxyz, zero_q.wxyz, atol=1e-6)
