"""Tensor operations tests for Quaternion class.

Tests for shape, ndim, size, dtype, itemsize, __len__, reshape, flatten, ravel, squeeze.
"""

import jax
import jax.numpy as jnp
import pytest

from fastquat.quaternion import Quaternion


# Shape, ndim, size properties
@pytest.mark.parametrize(
    'q_factory,expected_shape,expected_ndim,expected_size',
    [
        (lambda: Quaternion(1.0), (), 0, 1),
        (lambda: Quaternion.ones((5,)), (5,), 1, 5),
        (lambda: Quaternion.ones((2, 3)), (2, 3), 2, 6),
        (lambda: Quaternion.ones((2, 3, 4)), (2, 3, 4), 3, 24),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_shape_ndim_size(q_factory, expected_shape, expected_ndim, expected_size, do_jit):
    """Test shape, ndim, and size properties."""

    def func(q):
        return q.shape, q.ndim, q.size

    if do_jit:
        func = jax.jit(func)

    q = q_factory()
    shape, ndim, size = func(q)

    assert shape == expected_shape
    assert ndim == expected_ndim
    assert size == expected_size


# dtype property
@pytest.mark.parametrize(
    'dtype',
    [jnp.float16, jnp.float32],
)
def test_dtype(dtype):
    """Test dtype property."""
    q = Quaternion.from_array(jnp.ones((2, 4), dtype=dtype))
    assert q.dtype == dtype


# itemsize property
@pytest.mark.parametrize(
    'dtype,expected_itemsize',
    [
        (jnp.float16, 4 * 2),  # 4 components * 2 bytes
        (jnp.float32, 4 * 4),  # 4 components * 4 bytes
    ],
)
def test_itemsize(dtype, expected_itemsize):
    """Test itemsize property."""
    q = Quaternion.from_array(jnp.ones((2, 4), dtype=dtype))
    assert q.itemsize == expected_itemsize


# __len__ method
@pytest.mark.parametrize(
    'shape,expected_len',
    [
        ((5,), 5),
        ((3, 7), 3),
        ((2, 3, 4), 2),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_len(shape, expected_len, do_jit):
    """Test __len__ method."""

    def func(q):
        return len(q)

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.ones(shape)
    assert func(q) == expected_len


def test_len_scalar_raises():
    """Test __len__ raises TypeError for scalar quaternion."""
    q = Quaternion(1.0)
    with pytest.raises(TypeError):
        len(q)


# reshape method
@pytest.mark.parametrize(
    'initial_shape,new_shape',
    [
        ((6,), (2, 3)),
        ((12,), (3, 4)),
        ((2, 3), (6,)),
        ((12,), (2, 2, 3)),
        ((2, 3, 4), (24,)),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_reshape(initial_shape, new_shape, do_jit):
    """Test reshape operation."""

    def func(q):
        return q.reshape(new_shape)

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.ones(initial_shape)
    q_reshaped = func(q)
    assert q_reshaped.shape == new_shape


def test_reshape_multiple_args():
    """Test reshape with multiple arguments."""
    q = Quaternion.ones((6,))
    q_reshaped = q.reshape(2, 3)
    assert q_reshaped.shape == (2, 3)


def test_reshape_forms_equivalence():
    """Test that both reshape forms give equivalent results."""
    q = Quaternion.from_array(jnp.arange(24).reshape(6, 4))

    q_tuple = q.reshape((2, 3))
    q_args = q.reshape(2, 3)

    assert q_tuple.shape == q_args.shape == (2, 3)
    assert jnp.allclose(q_tuple.wxyz, q_args.wxyz)


def test_reshape_empty_raises():
    """Test reshape with empty shape raises ValueError."""
    q = Quaternion.ones((6,))
    with pytest.raises(ValueError):
        q.reshape()


# flatten and ravel methods
@pytest.mark.parametrize(
    'shape,expected_flat_shape',
    [
        ((6,), (6,)),
        ((2, 3), (6,)),
        ((2, 3, 4), (24,)),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_flatten(shape, expected_flat_shape, do_jit):
    """Test flatten operation."""

    def func(q):
        return q.flatten()

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.ones(shape)
    q_flat = func(q)
    assert q_flat.shape == expected_flat_shape


@pytest.mark.parametrize('do_jit', [False, True])
def test_flatten_preserves_data(do_jit):
    """Test flatten preserves quaternion data."""

    def func(q):
        return q.flatten()

    if do_jit:
        func = jax.jit(func)

    data = jnp.arange(8).reshape(2, 4)
    q = Quaternion.from_array(data)
    q_flat = func(q)

    assert jnp.allclose(q_flat.wxyz, data)


@pytest.mark.parametrize('do_jit', [False, True])
def test_ravel_equals_flatten(do_jit):
    """Test ravel equals flatten."""

    def func(q):
        return q.ravel(), q.flatten()

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.ones((2, 3))
    q_ravel, q_flat = func(q)

    assert q_ravel.shape == q_flat.shape
    assert jnp.allclose(q_ravel.wxyz, q_flat.wxyz)


# squeeze method
@pytest.mark.parametrize(
    'shape,expected_shape',
    [
        ((1, 3, 1), (3,)),
        ((1, 1, 3), (3,)),
        ((2, 3), (2, 3)),  # no effect
        ((1,), ()),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_squeeze(shape, expected_shape, do_jit):
    """Test squeeze removes single dimensions."""

    def func(q):
        return q.squeeze()

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.ones(shape)
    q_squeezed = func(q)
    assert q_squeezed.shape == expected_shape


@pytest.mark.parametrize(
    'shape,axis,expected_shape',
    [
        ((1, 3, 1), 0, (3, 1)),
        ((1, 3, 1), 2, (1, 3)),
        ((1, 1, 3), 0, (1, 3)),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_squeeze_axis(shape, axis, expected_shape, do_jit):
    """Test squeeze with specific axis."""

    def func(q, axis):
        return q.squeeze(axis=axis)

    if do_jit:
        func = jax.jit(func, static_argnames=['axis'])

    q = Quaternion.ones(shape)
    q_squeezed = func(q, axis)
    assert q_squeezed.shape == expected_shape


# Empty tensor
@pytest.mark.parametrize('do_jit', [False, True])
def test_empty_tensor(do_jit):
    """Test tensor operations with empty arrays."""

    def func(q):
        return q.shape, q.ndim, q.size

    if do_jit:
        func = jax.jit(func)

    q = Quaternion.ones((0,))
    shape, ndim, size = func(q)

    assert shape == (0,)
    assert ndim == 1
    assert size == 0
