"""Tensor operations tests for Quaternion class.

Tests for shape, ndim, size, dtype, itemsize, __len__, reshape, flatten, ravel, squeeze.
"""

import jax
import jax.numpy as jnp
import pytest

from fastquat.quaternion import Quaternion


# Shape property
@pytest.mark.parametrize('do_jit', [False, True])
def test_shape_scalar(do_jit):
    """Test shape property for scalar quaternion."""

    def get_shape(q):
        return q.shape

    if do_jit:
        get_shape = jax.jit(get_shape)

    q = Quaternion.ones()
    shape = get_shape(q)
    assert shape == ()


@pytest.mark.parametrize('do_jit', [False, True])
def test_shape_tensor(do_jit):
    """Test shape property for tensor quaternion."""

    def get_shape(q):
        return q.shape

    if do_jit:
        get_shape = jax.jit(get_shape)

    q = Quaternion.ones((2, 3))
    shape = get_shape(q)
    assert shape == (2, 3)


# ndim property
@pytest.mark.parametrize('do_jit', [False, True])
def test_ndim_scalar(do_jit):
    """Test ndim property for scalar quaternion."""

    def get_ndim(q):
        return q.ndim

    if do_jit:
        get_ndim = jax.jit(get_ndim)

    q = Quaternion.ones()
    ndim = get_ndim(q)
    assert ndim == 0


@pytest.mark.parametrize('do_jit', [False, True])
def test_ndim_tensor(do_jit):
    """Test ndim property for tensor quaternion."""

    def get_ndim(q):
        return q.ndim

    if do_jit:
        get_ndim = jax.jit(get_ndim)

    q = Quaternion.ones((2, 3))
    ndim = get_ndim(q)
    assert ndim == 2


# size property
@pytest.mark.parametrize('do_jit', [False, True])
def test_size_scalar(do_jit):
    """Test size property for scalar quaternion."""

    def get_size(q):
        return q.size

    if do_jit:
        get_size = jax.jit(get_size)

    q = Quaternion.ones()
    size = get_size(q)
    assert size == 1


@pytest.mark.parametrize('do_jit', [False, True])
def test_size_tensor(do_jit):
    """Test size property for tensor quaternion."""

    def get_size(q):
        return q.size

    if do_jit:
        get_size = jax.jit(get_size)

    q = Quaternion.ones((2, 3))
    size = get_size(q)
    assert size == 6  # 2 * 3 quaternions


# dtype property
def test_dtype_float32():
    """Test dtype property for float32 quaternion."""
    q = Quaternion.from_array(jnp.ones((2, 4), dtype=jnp.float32))
    dtype = q.dtype
    assert dtype == jnp.float32


def test_dtype_float16():
    """Test dtype property for float64 quaternion."""
    q = Quaternion.from_array(jnp.ones((2, 4), dtype=jnp.float16))
    assert q.dtype == jnp.float16


# itemsize property
def test_itemsize_float32():
    """Test itemsize property for float32 quaternion."""
    q = Quaternion.from_array(jnp.ones((2, 4), dtype=jnp.float32))
    itemsize = q.itemsize
    assert itemsize == 4 * 4  # 4 components * 4 bytes each


def test_itemsize_float16():
    """Test itemsize property for float64 quaternion."""
    q = Quaternion.from_array(jnp.ones((2, 4), dtype=jnp.float16))
    assert q.itemsize == 4 * 2  # 4 components * 2 bytes each


# __len__ method
@pytest.mark.parametrize('do_jit', [False, True])
def test_len_1d(do_jit):
    """Test __len__ for 1D quaternion array."""

    def get_length(q):
        return len(q)

    if do_jit:
        get_length = jax.jit(get_length)

    q = Quaternion.ones((5,))
    length = get_length(q)
    assert length == 5


@pytest.mark.parametrize('do_jit', [False, True])
def test_len_2d(do_jit):
    """Test __len__ for 2D quaternion array."""

    def get_length(q):
        return len(q)

    if do_jit:
        get_length = jax.jit(get_length)

    q = Quaternion.ones((3, 7))
    length = get_length(q)
    assert length == 3


def test_len_scalar_raises():
    """Test __len__ raises TypeError for scalar quaternion."""
    q = Quaternion.ones()
    with pytest.raises(TypeError):
        len(q)


# reshape method
@pytest.mark.parametrize('do_jit', [False, True])
def test_reshape_basic(do_jit):
    """Test basic reshape operation."""

    def reshape_quaternion(q):
        return q.reshape((2, 3))

    if do_jit:
        reshape_quaternion = jax.jit(reshape_quaternion)

    q = Quaternion.ones((6,))
    q_reshaped = reshape_quaternion(q)
    assert q_reshaped.shape == (2, 3)


@pytest.mark.parametrize('do_jit', [False, True])
def test_reshape_tuple_arg(do_jit):
    """Test reshape with tuple argument."""

    def reshape_quaternion(q):
        return q.reshape((2, 3))

    if do_jit:
        reshape_quaternion = jax.jit(reshape_quaternion)

    q = Quaternion.ones((6,))
    q_reshaped = reshape_quaternion(q)
    assert q_reshaped.shape == (2, 3)


def test_reshape_multiple_args():
    """Test reshape with multiple arguments (non-JIT only due to static shape requirements)."""
    q = Quaternion.ones((6,))
    q_reshaped = q.reshape(2, 3)
    assert q_reshaped.shape == (2, 3)

    # Test with different shapes
    q = Quaternion.ones((12,))
    q_reshaped = q.reshape(3, 4)
    assert q_reshaped.shape == (3, 4)

    # Test with single dimension
    q = Quaternion.ones((2, 3))
    q_reshaped = q.reshape(6)
    assert q_reshaped.shape == (6,)


def test_reshape_forms_equivalence():
    """Test that both reshape forms give equivalent results."""
    q = Quaternion.from_array(jnp.arange(24).reshape(6, 4))

    # Both forms should give identical results
    q_tuple = q.reshape((2, 3))
    q_args = q.reshape(2, 3)

    assert q_tuple.shape == q_args.shape == (2, 3)
    assert jnp.allclose(q_tuple.wxyz, q_args.wxyz)

    # Test with more complex shapes
    q = Quaternion.from_array(jnp.arange(48).reshape(12, 4))
    q_tuple = q.reshape((2, 2, 3))
    q_args = q.reshape(2, 2, 3)

    assert q_tuple.shape == q_args.shape == (2, 2, 3)
    assert jnp.allclose(q_tuple.wxyz, q_args.wxyz)


def test_reshape_empty_raises():
    """Test reshape with empty shape raises ValueError."""
    q = Quaternion.ones((6,))
    with pytest.raises(ValueError):
        q.reshape()


# flatten method
@pytest.mark.parametrize('do_jit', [False, True])
def test_flatten(do_jit):
    """Test flatten operation."""

    def flatten_quaternion(q):
        return q.flatten()

    if do_jit:
        flatten_quaternion = jax.jit(flatten_quaternion)

    q = Quaternion.ones((2, 3))
    q_flat = flatten_quaternion(q)
    assert q_flat.shape == (6,)  # 2 * 3 = 6


@pytest.mark.parametrize('do_jit', [False, True])
def test_flatten_preserves_data(do_jit):
    """Test flatten preserves quaternion data."""

    def flatten_quaternion(q):
        return q.flatten()

    if do_jit:
        flatten_quaternion = jax.jit(flatten_quaternion)

    # Create quaternion with distinct values
    data = jnp.arange(8).reshape(2, 4)
    q = Quaternion.from_array(data)
    q_flat = flatten_quaternion(q)

    assert jnp.allclose(q_flat.wxyz, data)


# ravel method
@pytest.mark.parametrize('do_jit', [False, True])
def test_ravel(do_jit):
    """Test ravel operation."""

    def ravel_quaternion(q):
        return q.ravel()

    if do_jit:
        ravel_quaternion = jax.jit(ravel_quaternion)

    q = Quaternion.ones((2, 3))
    q_ravel = ravel_quaternion(q)
    assert q_ravel.shape == (6,)  # 2 * 3 = 6


@pytest.mark.parametrize('do_jit', [False, True])
def test_ravel_equals_flatten(do_jit):
    """Test ravel equals flatten."""

    def ravel_flatten_quaternion(q):
        return q.ravel(), q.flatten()

    if do_jit:
        ravel_flatten_quaternion = jax.jit(ravel_flatten_quaternion)

    q = Quaternion.ones((2, 3))
    q_ravel, q_flat = ravel_flatten_quaternion(q)

    assert jnp.allclose(q_ravel.wxyz, q_flat.wxyz)


# squeeze method
@pytest.mark.parametrize('do_jit', [False, True])
def test_squeeze_single_dim(do_jit):
    """Test squeeze removes single dimensions."""

    def squeeze_quaternion(q):
        return q.squeeze()

    if do_jit:
        squeeze_quaternion = jax.jit(squeeze_quaternion)

    q = Quaternion.ones((1, 3, 1))
    q_squeezed = squeeze_quaternion(q)
    assert q_squeezed.shape == (3,)


@pytest.mark.parametrize('do_jit', [False, True])
def test_squeeze_specific_axis(do_jit):
    """Test squeeze with specific axis."""

    def squeeze_quaternion(q, axis):
        return q.squeeze(axis=axis)

    if do_jit:
        squeeze_quaternion = jax.jit(squeeze_quaternion, static_argnames=['axis'])

    q = Quaternion.ones((1, 3, 1))
    q_squeezed = squeeze_quaternion(q, axis=0)
    assert q_squeezed.shape == (3, 1)


@pytest.mark.parametrize('do_jit', [False, True])
def test_squeeze_no_effect(do_jit):
    """Test squeeze has no effect when no single dimensions."""

    def squeeze_quaternion(q):
        return q.squeeze()

    if do_jit:
        squeeze_quaternion = jax.jit(squeeze_quaternion)

    q = Quaternion.ones((2, 3))
    q_squeezed = squeeze_quaternion(q)
    assert q_squeezed.shape == (2, 3)


# Edge cases
@pytest.mark.parametrize('do_jit', [False, True])
def test_batch_tensor_operations(do_jit):
    """Test tensor operations work with batched quaternions."""

    def batch_operations(q):
        return q.shape, q.ndim, q.size

    if do_jit:
        batch_operations = jax.jit(batch_operations)

    # Create complex batch shape
    q = Quaternion.ones((2, 3, 5))
    shape, ndim, size = batch_operations(q)

    assert shape == (2, 3, 5)
    assert ndim == 3
    assert size == 30  # 2 * 3 * 5
    # Test dtype separately without JIT
    assert q.dtype == jnp.float32


@pytest.mark.parametrize('do_jit', [False, True])
def test_reshape_chain_operations(do_jit):
    """Test chaining reshape operations."""

    def chain_reshape(q):
        return q.reshape((6,)).reshape((2, 3)).flatten()

    if do_jit:
        chain_reshape = jax.jit(chain_reshape)

    q = Quaternion.ones((6,))
    q_result = chain_reshape(q)
    assert q_result.shape == (6,)


@pytest.mark.parametrize('do_jit', [False, True])
def test_squeeze_reshape_combination(do_jit):
    """Test combining squeeze and reshape operations."""

    def squeeze_reshape_combo(q):
        return q.squeeze().reshape((3, 2))

    if do_jit:
        squeeze_reshape_combo = jax.jit(squeeze_reshape_combo)

    q = Quaternion.ones((1, 6))
    q_result = squeeze_reshape_combo(q)
    assert q_result.shape == (3, 2)


@pytest.mark.parametrize('do_jit', [False, True])
def test_empty_tensor_operations(do_jit):
    """Test tensor operations with empty arrays."""

    def empty_operations(q):
        return q.shape, q.ndim, q.size

    if do_jit:
        empty_operations = jax.jit(empty_operations)

    q = Quaternion.ones((0,))
    shape, ndim, size = empty_operations(q)

    assert shape == (0,)
    assert ndim == 1
    assert size == 0
