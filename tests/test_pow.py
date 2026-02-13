import jax
import jax.numpy as jnp
import pytest

from fastquat import Quaternion


# Logarithm and Exponential operations
@pytest.mark.parametrize('do_jit', [False, True])
def test_log_exp_inverse(do_jit):
    """Test that exp(log(q)) = q for general quaternions."""

    def func(q):
        return q.log().exp()

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = func(q)
    assert jnp.allclose(result.wxyz, q.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_log_real(do_jit):
    """Test logarithm of real quaternion."""

    def func(q):
        return q.log()

    if do_jit:
        func = jax.jit(func)

    q_real = Quaternion(2.0)
    log_q = func(q_real)

    # log(2) should be (log(2), 0, 0, 0)
    expected = jnp.array([jnp.log(2.0), 0.0, 0.0, 0.0])
    assert jnp.allclose(log_q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_log_unit(do_jit):
    """Test logarithm of unit quaternion."""

    def func(q):
        return q.log()

    if do_jit:
        func = jax.jit(func)

    # Unit quaternion representing 90° rotation around z
    angle = jnp.pi / 2
    q_unit = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))
    log_q = func(q_unit)

    # log should be (0, 0, 0, π/4) for 90° rotation around z
    expected = jnp.array([0.0, 0.0, 0.0, angle / 2])
    assert jnp.allclose(log_q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_log_zero(do_jit):
    """Test log handles zero quaternion correctly."""

    def func(q):
        return q.log()

    if do_jit:
        func = jax.jit(func)

    q_zero = Quaternion(0.0)
    log_q = func(q_zero)

    # log(0) should be (-inf, 0, 0, 0)
    assert jnp.isneginf(log_q.w)
    assert jnp.allclose(log_q.vector, 0.0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_log_exp_batch(do_jit):
    """Test log and exp with batch of quaternions."""

    def func(q_batch):
        return q_batch.log().exp()

    if do_jit:
        func = jax.jit(func)

    # Batch of normalized quaternions for better numerical stability
    q_batch = Quaternion.from_array(
        jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # Identity
                [0.707, 0.707, 0.0, 0.0],  # 90° around x
                [0.5, 0.5, 0.5, 0.5],  # Equal components, normalized
            ]
        )
    ).normalize()

    result = func(q_batch)
    assert jnp.allclose(result.wxyz, q_batch.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_exp_real(do_jit):
    """Test exponential of real quaternion."""

    def func(q):
        return q.exp()

    if do_jit:
        func = jax.jit(func)

    q_real = Quaternion(1.5)
    exp_q = func(q_real)

    # exp(1.5) should be (exp(1.5), 0, 0, 0)
    expected = jnp.array([jnp.exp(1.5), 0.0, 0.0, 0.0])
    assert jnp.allclose(exp_q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_exp_zero(do_jit):
    """Test exponential of zero quaternion."""

    def func(q):
        return q.exp()

    if do_jit:
        func = jax.jit(func)

    q_zero = Quaternion(0.0)
    exp_q = func(q_zero)

    # exp(0) should be (1, 0, 0, 0)
    expected = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(exp_q.wxyz, expected, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_exp_pure_imaginary(do_jit):
    """Test exponential of pure imaginary quaternion."""

    def func(q):
        return q.exp()

    if do_jit:
        func = jax.jit(func)

    # Pure imaginary quaternion (0, 0, 0, π/4)
    angle = jnp.pi / 2
    q_pure = Quaternion(0.0, 0.0, 0.0, angle / 2)
    exp_q = func(q_pure)

    # Should give unit quaternion for rotation around z
    expected = jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])
    assert jnp.allclose(exp_q.wxyz, expected, atol=1e-6)


# Power operations
@pytest.mark.parametrize(
    'exponent,expected_func',
    [
        (-2, lambda q: (1 / q) * (1 / q)),
        (-1, lambda q: 1 / q),
        (0, lambda q: Quaternion.ones(q.shape, q.dtype)),
        (1, lambda q: q),
        (2, lambda q: q * q),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_special_cases(exponent, expected_func, do_jit):
    """Test quaternion power with special case exponents."""

    def func(q):
        return q**exponent

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = func(q)
    expected = expected_func(q)
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_integer_positive(do_jit):
    """Test quaternion power with positive integer exponents."""

    def func(q, n):
        return q**n

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(0.5, 0.5, 0.5, 0.5).normalize()

    # Test q^3
    result_3 = func(q, 3)
    expected_3 = q * q * q
    assert jnp.allclose(result_3.wxyz, expected_3.wxyz, atol=1e-5)

    # Test q^4
    result_4 = func(q, 4)
    expected_4 = q * q * q * q
    assert jnp.allclose(result_4.wxyz, expected_4.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_integer_negative(do_jit):
    """Test quaternion power with negative integer exponents."""

    def func(q, n):
        return q**n

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 1.0, 1.0, 1.0).normalize()

    # Test q^-3
    result_neg3 = func(q, -3)
    q_inv = 1 / q
    expected_neg3 = q_inv * q_inv * q_inv
    assert jnp.allclose(result_neg3.wxyz, expected_neg3.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_fractional(do_jit):
    """Test quaternion power with fractional exponents."""

    def func(q, n):
        return q**n

    if do_jit:
        func = jax.jit(func)

    # Use a rotation quaternion (90° around z-axis)
    angle = jnp.pi / 2
    q = Quaternion(jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2))

    # Test q^0.5 (should be 45° rotation around z)
    result_half = func(q, 0.5)
    expected_angle = angle / 2
    expected_half = Quaternion(jnp.cos(expected_angle / 2), 0.0, 0.0, jnp.sin(expected_angle / 2))
    assert jnp.allclose(result_half.wxyz, expected_half.wxyz, atol=1e-5)

    # Verify (q^0.5)^2 ≈ q
    double_half = result_half * result_half
    assert jnp.allclose(double_half.wxyz, q.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_properties(do_jit):
    """Test mathematical properties of quaternion powers: (q^a)^b = q^(a*b)."""

    def func(q, a, b):
        return q ** (a * b), (q**a) ** b

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(0.6, 0.8, 0.0, 0.0).normalize()
    a, b = 2.0, 1.5

    q_ab, q_a_b = func(q, a, b)
    assert jnp.allclose(q_a_b.wxyz, q_ab.wxyz, atol=1e-5)


@pytest.mark.parametrize('exponent', [0.5, 1.5, 2.0, -0.5, -1.5])
@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_preserves_unit_norm(exponent, do_jit):
    """Test power operations preserve unit quaternion properties."""

    def func(q):
        return q**exponent

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(0.5, 0.5, 0.5, 0.5).normalize()
    result = func(q)
    assert jnp.allclose(abs(result), 1.0, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_batch(do_jit):
    """Test quaternion power with batch of quaternions."""

    def func(q_batch, n):
        return q_batch**n

    if do_jit:
        func = jax.jit(func)

    q_batch = Quaternion.from_array(
        jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    )

    result = func(q_batch, 2)
    expected = q_batch * q_batch
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-6)


@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_array_exponent(do_jit):
    """Test quaternion power with array of exponents."""

    def func(q, n_array):
        return q**n_array

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(1.0, 1.0, 0.0, 0.0).normalize()
    n_array = jnp.array([0.0, 1.0, 2.0])

    result = func(q, n_array)

    assert jnp.allclose(result.wxyz[0], jnp.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)  # q^0
    assert jnp.allclose(result.wxyz[1], q.wxyz, atol=1e-6)  # q^1
    assert jnp.allclose(result.wxyz[2], (q * q).wxyz, atol=1e-6)  # q^2


@pytest.mark.parametrize(
    'q,exponent,expected',
    [
        (Quaternion(1.0), 3.5, Quaternion(1.0)),  # identity^n = identity
        (Quaternion(2.0), 2.0, Quaternion(4.0)),  # real: 2^2 = 4
        (Quaternion(x=1.0), 2.0, Quaternion(-1.0)),  # i^2 = -1
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_edge_cases(q, exponent, expected, do_jit):
    """Test quaternion power edge cases."""

    def func(q_, n):
        return q_**n

    if do_jit:
        func = jax.jit(func)

    result = func(q, exponent)
    assert jnp.allclose(result.wxyz, expected.wxyz, atol=1e-5)


@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_consistency(do_jit):
    """Test consistency between special cases and general formula."""

    def func(q, n):
        return q**n

    if do_jit:
        func = jax.jit(func)

    q = Quaternion(0.8, 0.6, 0.0, 0.0).normalize()

    # Test that special cases give same results as would be computed generally
    result_near_2 = func(q, 2.000001)
    result_exact_2 = func(q, 2.0)

    assert jnp.allclose(result_near_2.wxyz, result_exact_2.wxyz, atol=1e-4)


@pytest.mark.parametrize('cast_type', [int, float, jnp.array])
@pytest.mark.parametrize('do_jit', [False, True])
def test_pow_zero_quaternion(cast_type, do_jit):
    """Test power operations with zero quaternion."""

    def func(q, n):
        return q**n

    if do_jit:
        func = jax.jit(func)

    zero_q = Quaternion(0.0)

    # 0^0 should give 1 (by convention)
    result_0_0 = func(zero_q, cast_type(0.0))
    expected_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(result_0_0.wxyz, expected_identity)

    # 0^n for n > 0 should give 0
    result_0_2 = func(zero_q, cast_type(2.0))
    assert jnp.allclose(result_0_2.wxyz, zero_q.wxyz)
