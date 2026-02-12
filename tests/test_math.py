"""Mathematical operations tests for Quaternion class.

Tests for addition, subtraction, multiplication, negation, conjugate, norm, normalize, inverse.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fastquat.quaternion import Quaternion


# Addition
@pytest.mark.parametrize('do_jit', [False, True])
def test_add_quaternion(do_jit):
    """Test quaternion addition."""

    def func(q1_, q2_):
        return q1_ + q2_

    if do_jit:
        func = jax.jit(func)

    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, 1.5, 2.5, 3.5)
    result = func(q1, q2)
    expected = jnp.array([1.5, 3.5, 5.5, 7.5])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('func', [lambda q, s: q + s, lambda q, s: s + q])
@pytest.mark.parametrize(
    'scalar', [True, 1, 1.0, np.bool_(True), np.float16(1.0), jnp.array(1.0), np.array(1.0)]
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_add_radd_real_scalar(func, scalar, do_jit):
    """Test quaternion addition with real scalar."""
    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0, dtype=jnp.float64)
    result = func(q, scalar)
    expected = jnp.array([2.0, 2.0, 3.0, 4.0])  # Only w component affected
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('func', [lambda q, a: q + a, lambda q, a: a + q])
@pytest.mark.parametrize('array', [jnp.array([1, 2]), np.array([1, 2])])
@pytest.mark.parametrize('do_jit', [False, True])
def test_add_radd_real_array(func, array, do_jit):
    """Test quaternion multiplication by real array."""

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.from_array([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = func(q, array)
    expected = jnp.array([[2, 2, 3, 4], [7, 6, 7, 8]])
    assert jnp.allclose(result.wxyz, expected)


# Subtraction
@pytest.mark.parametrize('do_jit', [False, True])
def test_sub_quaternion(do_jit):
    """Test quaternion subtraction."""

    def func(q1, q2):
        return q1 - q2

    if do_jit:
        func = jax.jit(func)

    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, 1.5, 2.5, 3.5)
    result = func(q1, q2)
    expected = jnp.array([0.5, 0.5, 0.5, 0.5])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize(
    'scalar', [True, 1, 1.0, np.bool_(True), np.float16(1.0), jnp.array(1.0), np.array(1.0)]
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_sub_real_scalar(scalar, do_jit):
    """Test quaternion subtraction with scalar."""

    def func(q_, scalar_):
        return q_ - scalar_

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = func(q, scalar)
    expected = jnp.array([0.0, 2.0, 3.0, 4.0])  # Only w component affected
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('array', [jnp.array([1, 2]), np.array([1, 2])])
@pytest.mark.parametrize('do_jit', [False, True])
def test_sub_real_array(array, do_jit):
    """Test quaternion multiplication by real array."""

    def func(q_, array_):
        return q_ - array_

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.from_array([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = func(q, array)
    expected = jnp.array([[0, 2, 3, 4], [3, 6, 7, 8]])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize(
    'scalar', [True, 1, 1.0, np.bool_(True), np.float16(1.0), jnp.array(1.0), np.array(1.0)]
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_rsub_real_scalar(scalar, do_jit):
    """Test quaternion subtraction with scalar."""

    def func(q_, scalar_):
        return scalar_ - q_

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = func(q, scalar)
    expected = jnp.array([0.0, -2.0, -3.0, -4.0])  # Only w component affected
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('array', [jnp.array([1, 2]), np.array([1, 2])])
@pytest.mark.parametrize('do_jit', [False, True])
def test_rsub_real_array(array, do_jit):
    """Test quaternion multiplication by real array."""

    def func(q_, array_):
        return array_ - q_

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.from_array([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = func(q, array)
    expected = jnp.array([[0, -2, -3, -4], [-3, -6, -7, -8]])
    assert jnp.allclose(result.wxyz, expected)


# Multiplication
@pytest.mark.parametrize('do_jit', [False, True])
def test_mul_quaternion(do_jit):
    """Test quaternion multiplication."""

    def func(q1, q2):
        return q1 * q2

    if do_jit:
        func = jax.jit(func)

    qi = Quaternion(x=1.0)
    qj = Quaternion(y=1.0)
    qk = Quaternion(z=1.0)

    for q1, q2, expected_q in [(qi, qj, qk), (qj, qk, qi), (qk, qi, qj)]:
        actual_q = func(q1, q2)
        assert jnp.allclose(actual_q.wxyz, expected_q.wxyz)


@pytest.mark.parametrize('func', [lambda q, s: q * s, lambda q, s: s * q])
@pytest.mark.parametrize(
    'scalar', [True, 2, 2.0, np.bool_(True), np.float16(2.0), jnp.array(2), np.array(2.0)]
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_mul_rmul_real_scalar(func, scalar, do_jit):
    """Test quaternion multiplication by scalar."""

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = func(q, scalar)
    expected = q.wxyz * scalar
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('func', [lambda q, a: q * a, lambda q, a: a * q])
@pytest.mark.parametrize('array', [jnp.array([1, 2]), np.array([1, 2])])
@pytest.mark.parametrize('do_jit', [False, True])
def test_mul_rmul_real_array(func, array, do_jit):
    """Test quaternion multiplication by real array."""

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.from_array([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = func(q, array)
    expected = jnp.array([[1, 2, 3, 4], [10, 12, 14, 16]])
    assert jnp.allclose(result.wxyz, expected)


# Division
@pytest.mark.parametrize('do_jit', [False, True])
def test_truediv_quaternion(do_jit):
    """Test quaternion division."""

    def func(q1, q2):
        return q1 / q2

    if do_jit:
        func = jax.jit(func)

    # Identity quaternion
    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    qi = Quaternion(x=1.0)
    qj = Quaternion(y=1.0)
    qk = Quaternion(z=1.0)

    for q2, q2_inv in [(1, 1), (qi, -qi), (qj, -qj), (qk, -qk)]:
        result = func(q1, q2)
        expected = q1 * q2_inv
        assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize(
    'scalar', [True, 2, 2.0, np.bool_(True), np.float16(2.0), jnp.array(2), np.array(2.0)]
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_truediv_real_scalar(scalar, do_jit):
    """Test quaternion division by real scalar."""

    def func(q, scalar):
        return q / scalar

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(2.0, 4.0, 6.0, 8.0)
    result = func(q, scalar)
    expected = q.wxyz / scalar
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('array', [jnp.array([1, 2]), np.array([1, 2])])
@pytest.mark.parametrize('do_jit', [False, True])
def test_truediv_real_array(array, do_jit):
    """Test quaternion division by real array."""

    def func(q, arr):
        return q / arr

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.from_array([[2, 4, 6, 8], [10, 12, 14, 16]])
    result = func(q, array)
    expected = jnp.array([[2, 4, 6, 8], [5, 6, 7, 8]])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize(
    'scalar', [True, 2, 2.0, np.bool_(True), np.float16(2.0), jnp.array(2), np.array(2.0)]
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_rtruediv_real_scalar(scalar, do_jit):
    """Test real scalar division by quaternion."""

    def func(scalar, q):
        return scalar / q

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = func(scalar, q)
    expected = scalar * q._inverse()
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('array', [jnp.array([1, 2]), np.array([1, 2])])
@pytest.mark.parametrize('do_jit', [False, True])
def test_rtruediv_real_array(array, do_jit):
    """Test real array division by quaternion."""

    def func(arr, q):
        return arr / q

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.from_array([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = func(array, q)
    expected = jnp.array(
        [[1 / 30, -2 / 30, -3 / 30, -4 / 30], [10 / 174, -12 / 174, -14 / 174, -16 / 174]]
    )
    assert jnp.allclose(result.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_division_by_self(do_jit):
    """Test that q / q approximates identity quaternion."""

    def func(q):
        return q / q

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(2.0, 4.0, 6.0, 8.0)
    result = func(q)

    expected_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(result.wxyz, expected_identity, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_division_by_identity(do_jit):
    """Test division by identity quaternion."""

    def func(q):
        identity = Quaternion(1)
        return q / identity

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = func(q)

    # q / identity should equal q
    assert jnp.allclose(result.wxyz, q.wxyz)


@pytest.mark.parametrize('do_jit', [False, True])
def test_division_associativity_property(do_jit):
    """Test division associativity: (q1 / q2) / q3 = q1 / (q2 * q3)."""

    def func(q1, q2, q3):
        left = (q1 / q2) / q3
        right = q1 / (q3 * q2)
        return left, right

    if do_jit:
        func = jax.jit(func)

    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, 1.0, 1.5, 2.0)
    q3 = Quaternion(2.0, 1.0, 0.5, 1.5)

    left, right = func(q1, q2, q3)
    assert jnp.allclose(left.wxyz, right.wxyz, atol=1e-6)


# Broadcasting
@pytest.mark.parametrize(
    'op,expected',
    [
        (lambda a, b: a + b, jnp.array([[2.0, 1.0, 1.0, 1.0]] * 3)),  # add
        (lambda a, b: b + a, jnp.array([[2.0, 1.0, 1.0, 1.0]] * 3)),  # radd
        (lambda a, b: a - b, jnp.array([[0.0, -1.0, -1.0, -1.0]] * 3)),  # sub
        (lambda a, b: b - a, jnp.array([[0.0, 1.0, 1.0, 1.0]] * 3)),  # rsub
        (lambda a, b: a * b, jnp.array([[1.0, 1.0, 1.0, 1.0]] * 3)),  # mul
        (lambda a, b: b * a, jnp.array([[1.0, 1.0, 1.0, 1.0]] * 3)),  # rmul
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_broadcasting_operations(op, expected, do_jit):
    """Test operations with broadcasting."""
    if do_jit:
        op = jax.jit(op)

    q_scalar = Quaternion(1.0)
    q_batch = Quaternion.from_array(jnp.ones((3, 4)))  # [1, 1, 1, 1] x 3

    result = op(q_scalar, q_batch)

    assert result.shape == (3,)
    assert jnp.allclose(result.wxyz, expected)


# Negation
@pytest.mark.parametrize('do_jit', [False, True])
def test_pos(do_jit):
    """Test quaternion negation."""

    def func(q):
        return +q

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = func(q)
    expected = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(result.wxyz, expected)


# Negation
@pytest.mark.parametrize('do_jit', [False, True])
def test_neg(do_jit):
    """Test quaternion negation."""

    def func(q):
        return -q

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = func(q)
    expected = jnp.array([-1.0, -2.0, -3.0, -4.0])
    assert jnp.allclose(result.wxyz, expected)


@pytest.mark.parametrize('func', [lambda q: q.conj(), lambda q: q.conjugate()])
@pytest.mark.parametrize('do_jit', [False, True])
def test_conj_conjugate(func, do_jit):
    """Test quaternion conjugate."""

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    conj_q = func(q)
    expected = jnp.array([1.0, -2.0, -3.0, -4.0])
    assert jnp.allclose(conj_q.wxyz, expected)


# Norm
@pytest.mark.parametrize('do_jit', [False, True])
def test_norm(do_jit):
    """Test quaternion norm calculation."""

    def func(q):
        return q.norm()

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    norm = func(q)
    expected = jnp.sqrt(1 + 4 + 9 + 16)  # sqrt(30)
    assert jnp.allclose(norm, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_norm_unit(do_jit):
    """Test unit quaternion norm is 1."""

    def func(q):
        return q.norm()

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0)  # Identity quaternion
    norm = func(q)
    assert jnp.allclose(norm, 1.0)


# Normalize
@pytest.mark.parametrize('do_jit', [False, True])
def test_normalize(do_jit):
    """Test quaternion normalization."""

    def func(q):
        return q.normalize()

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    q_norm = func(q)
    assert jnp.allclose(q_norm.norm(), 1.0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_normalize_preserves_direction(do_jit):
    """Test normalization preserves quaternion direction."""

    def func(q):
        return q.normalize()

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(2.0, 4.0, 6.0, 8.0)  # 2 * (1, 2, 3, 4)
    q_norm = func(q)

    # Direction should be same as (1, 2, 3, 4) normalized
    expected_direction = jnp.array([1.0, 2.0, 3.0, 4.0]) / jnp.sqrt(30)
    assert jnp.allclose(q_norm.wxyz, expected_direction)


@pytest.mark.parametrize('do_jit', [False, True])
def test_normalize_zero_quaternion(do_jit):
    """Test normalization of zero quaternion returns identity."""

    def func(q):
        return q.normalize()

    if do_jit:
        func = jax.jit(func)

    # Zero quaternion
    q_zero = Quaternion(0.0)
    q_norm = func(q_zero)

    # Should return identity quaternion
    expected_identity = jnp.array([0.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(q_norm.wxyz, expected_identity)


@pytest.mark.parametrize('do_jit', [False, True])
def test_normalize_tensor_with_zero(do_jit):
    """Test normalization of batch containing zero quaternions."""

    def func(q):
        return q.normalize()

    if do_jit:
        func = jax.jit(func)

    # Batch with mix of normal and zero quaternions
    q_tensor = Quaternion.from_array(
        [
            [1.0, 2.0, 3.0, 4.0],  # Normal quaternion
            [0.0, 0.0, 0.0, 0.0],  # Zero quaternion
            [2.0, 0.0, 0.0, 0.0],  # Real quaternion
        ]
    )

    q_norm_tensor = func(q_tensor)

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
def test_inverse(do_jit):
    """Test quaternion inverse."""

    def func(q):
        return 1 / q

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    q_inv = func(q)

    # Check q * q_inv = identity (approximately)
    identity = q * q_inv
    expected = Quaternion(1)
    assert jnp.allclose(identity.wxyz, expected.wxyz, atol=1e-6)


# Edge cases
@pytest.mark.parametrize('do_jit', [False, True])
def test_zero_quaternion_operations(do_jit):
    """Test operations with zero quaternion."""

    def func(q_zero, q_normal):
        return q_zero + q_normal, q_zero * q_normal, -q_zero

    if do_jit:
        func = jax.jit(func)

    q_zero = Quaternion(0.0)
    q_normal = Quaternion(1.0, 2.0, 3.0, 4.0)

    add_result, mul_result, neg_result = func(q_zero, q_normal)

    # Addition: 0 + q = q
    assert jnp.allclose(add_result.wxyz, q_normal.wxyz)
    # Multiplication: 0 * q = 0
    assert jnp.allclose(mul_result.wxyz, jnp.zeros(4))
    # Negation: -0 = 0
    assert jnp.allclose(neg_result.wxyz, jnp.zeros(4))
