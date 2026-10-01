"""Comparison tests for Quaternion class.

Tests for element-wise __eq__ and __ne__.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fastquat.quaternion import Quaternion


@pytest.mark.parametrize('do_jit', [False, True])
def test_eq_scalar(do_jit: bool):
    """Test equality of 0-d quaternions."""

    def eq(q, p):
        return q == p, q != p

    if do_jit:
        eq = jax.jit(eq)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    is_eq, is_ne = eq(q, Quaternion(1.0, 2.0, 3.0, 4.0))
    assert is_eq.shape == ()
    assert is_eq.dtype == jnp.bool_
    assert is_eq
    assert not is_ne

    for other in [
        Quaternion(0.0, 2.0, 3.0, 4.0),
        Quaternion(1.0, 0.0, 3.0, 4.0),
        Quaternion(1.0, 2.0, 0.0, 4.0),
        Quaternion(1.0, 2.0, 3.0, 0.0),
    ]:
        is_eq, is_ne = eq(q, other)
        assert not is_eq
        assert is_ne


def test_eq_not_rotation_equivalence():
    """Test that q and -q are not equal, although they represent the same rotation."""
    q = Quaternion(1.0, 2.0, 3.0, 4.0).normalize()
    assert not q == -q
    assert q != -q


@pytest.mark.parametrize('do_jit', [False, True])
def test_eq_elementwise(do_jit: bool):
    """Test element-wise equality of quaternion tensors."""

    def eq(q, p):
        return q == p, q != p

    if do_jit:
        eq = jax.jit(eq)

    q = Quaternion.from_array(jnp.arange(24.0).reshape(2, 3, 4))
    p = Quaternion.from_array(q.wxyz.at[0, 1, 2].set(-1).at[1, 2, 0].set(-1))
    is_eq, is_ne = eq(q, p)
    expected = np.array([[True, False, True], [True, True, False]])
    np.testing.assert_array_equal(is_eq, expected)
    np.testing.assert_array_equal(is_ne, ~expected)


def test_eq_broadcasting():
    """Test that comparison broadcasts the quaternion shapes."""
    q = Quaternion.from_array(jnp.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
    p = Quaternion.from_array(jnp.array([[[0.0, 1, 0, 0]], [[0, 0, 1, 0]]]))
    np.testing.assert_array_equal(
        q == p, np.array([[False, True, False], [False, False, True]])
    )
    np.testing.assert_array_equal(q == Quaternion(0, 1), np.array([False, True, False]))


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize('scalar', [2, 2.0, np.float32(2), jnp.array(2.0)])
def test_eq_real_scalar(do_jit: bool, scalar):
    """Test that real scalars compare as quaternions with a zero vector part."""

    def eq(q, s):
        return q == s, s == q, q != s, s != q

    if do_jit:
        eq = jax.jit(eq)

    q = Quaternion.from_array(jnp.array([[2.0, 0, 0, 0], [2, 1, 0, 0], [3, 0, 0, 0]]))
    expected = np.array([True, False, False])
    for actual, exp in zip(eq(q, scalar), [expected, expected, ~expected, ~expected]):
        np.testing.assert_array_equal(actual, exp)


@pytest.mark.parametrize('array_type', [np.array, jnp.array])
def test_eq_real_array(array_type):
    """Test that real arrays compare element-wise as quaternions with a zero vector part."""
    q = Quaternion.from_array(jnp.array([[2.0, 0, 0, 0], [2, 1, 0, 0], [3, 0, 0, 0]]))
    other = array_type([2.0, 2.0, 2.0])
    expected = np.array([True, False, False])
    np.testing.assert_array_equal(q == other, expected)
    np.testing.assert_array_equal(other == q, expected)
    np.testing.assert_array_equal(q != other, ~expected)
    np.testing.assert_array_equal(other != q, ~expected)


def test_eq_nan():
    """Test that NaN components are never equal, as in NumPy."""
    q = Quaternion(1.0, jnp.nan, 0.0, 0.0)
    assert not q == q
    assert q != q


def test_eq_complex():
    """Test that comparison with complex numbers is not implemented."""
    q = Quaternion(1.0)
    with pytest.raises(NotImplementedError):
        q == 1j
    with pytest.raises(NotImplementedError):
        q != 1j


@pytest.mark.parametrize('other', [None, 'quaternion', object()])
def test_eq_unrelated_type(other):
    """Test that comparison with unrelated types falls back to identity."""
    q = Quaternion(1.0)
    assert not q == other
    assert q != other
    assert not other == q
    assert other != q


def test_unhashable():
    """Test that quaternions are not hashable, since equality is element-wise."""
    with pytest.raises(TypeError):
        hash(Quaternion(1.0))
