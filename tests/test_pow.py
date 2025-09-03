import jax
import jax.numpy as jnp
import pytest

from fastquat import Quaternion


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
def test_quaternion_exp_zero(do_jit):
    """Test exponential of real quaternion."""

    def exp_zero(q):
        return q.exp()

    if do_jit:
        exp_zero = jax.jit(exp_zero)

    # Zero quaternion
    q_real = Quaternion(0.0, 0.0, 0.0, 0.0)
    exp_q = exp_zero(q_real)

    # exp(0) should be (1, 0, 0, 0)
    expected = jnp.array([1.0, 0.0, 0.0, 0.0])
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
    """Test log handles zero quaternion correctly."""

    def log_zero(q):
        return q.log()

    if do_jit:
        log_zero = jax.jit(log_zero)

    # Zero quaternion - should not crash
    q_zero = Quaternion(0.0, 0.0, 0.0, 0.0)
    log_q = log_zero(q_zero)

    assert jnp.allclose(log_q.vector, 0.0)
    assert jnp.isneginf(log_q.w)


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


@pytest.mark.parametrize('cast_type', [int, float, jnp.array])
@pytest.mark.parametrize('do_jit', [False, True])
def test_power_zero_quaternion(do_jit: bool, cast_type: type) -> None:
    """Test power operations with zero quaternion."""

    def power_zero_q(q, n):
        return q**n

    if do_jit:
        power_zero_q = jax.jit(power_zero_q)

    zero_q = Quaternion(0.0, 0.0, 0.0, 0.0)

    # 0^0 should give 1 (by convention)
    result_0_0 = power_zero_q(zero_q, cast_type(0.0))
    expected_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(result_0_0.wxyz, expected_identity)

    # 0^n for n > 0 should give 0
    result_0_2 = power_zero_q(zero_q, cast_type(2.0))
    assert jnp.allclose(result_0_2.wxyz, zero_q.wxyz)
