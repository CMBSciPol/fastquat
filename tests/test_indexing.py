"""Indexing tests for Quaternion class.

Tests for __getitem__: integers, slices, boolean masks, fancy integer arrays, None
(newaxis), Ellipsis, and combinations thereof. The component axis (w, x, y, z) must
never be reachable by these indices -- only the batch shape is indexed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fastquat.quaternion import Quaternion


def _arange_quat(*shape: int) -> Quaternion:
    """A Quaternion whose wxyz array is `jnp.arange(prod(shape) * 4).reshape(*shape, 4)`."""
    size = 1
    for dim in shape:
        size *= dim
    return Quaternion.from_array(jnp.arange(size * 4, dtype=jnp.float32).reshape(*shape, 4))


@pytest.mark.parametrize(
    'idx',
    [0, -1, 2, slice(1, 4), slice(None, None, 2), slice(None, None, -1), slice(2, None)],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_getitem_int_and_slice(idx, do_jit):
    """Basic and strided/negative slicing over the leading axis."""

    # `idx` is a Python-level constant closed over by `func`, not a traced argument.
    def func(q):
        return q[idx]

    if do_jit:
        func = jax.jit(func)

    q = _arange_quat(6)
    result = func(q)
    expected = q.wxyz[idx]
    assert result.wxyz.shape == expected.shape
    assert jnp.allclose(result.wxyz, expected)


def test_getitem_boolean_mask():
    """Boolean-mask indexing preserves the component axis.

    Not parametrized over `do_jit`: a traced boolean mask is non-concrete under `jit`, exactly
    as it would be for a plain JAX array (`arr[mask]`) -- this is a JAX limitation, not
    something `Quaternion.__getitem__` needs to work around.
    """
    mask = jnp.array([True, False, True, True, False])
    q = _arange_quat(5)
    result = q[mask]
    assert result.shape == (3,)
    assert jnp.allclose(result.wxyz, q.wxyz[mask])


@pytest.mark.parametrize('do_jit', [False, True])
def test_getitem_fancy_integer_array(do_jit):
    """Gather with a dynamic integer index array (e.g. batched detector selection)."""
    idx = jnp.array([3, 0, 0, 4])

    def func(q, i):
        return q[i]

    if do_jit:
        func = jax.jit(func)

    q = _arange_quat(5)
    result = func(q, idx)
    assert result.shape == (4,)
    assert jnp.allclose(result.wxyz, q.wxyz[idx])


@pytest.mark.parametrize('do_jit', [False, True])
def test_getitem_fancy_integer_array_out_of_range_clamps(do_jit):
    """Out-of-range indices clamp like default JAX gather semantics (no IndexError)."""
    idx = jnp.array([0, 10, -10])

    def func(q, i):
        return q[i]

    if do_jit:
        func = jax.jit(func)

    q = _arange_quat(5)
    result = func(q, idx)
    assert jnp.allclose(result.wxyz, q.wxyz[idx])


@pytest.mark.parametrize(
    'idx,expected_shape',
    [
        ((None, slice(None)), (1, 5)),
        ((slice(None), None), (5, 1)),
        ((None, slice(None), None), (1, 5, 1)),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_getitem_none_newaxis(idx, expected_shape, do_jit):
    """None/newaxis inserts a batch axis without disturbing the components."""

    def func(q):
        return q[idx]

    if do_jit:
        func = jax.jit(func)

    q = _arange_quat(5)
    result = func(q)
    assert result.shape == expected_shape
    assert jnp.allclose(result.wxyz, q.wxyz[idx])


@pytest.mark.parametrize('do_jit', [False, True])
def test_getitem_ellipsis_with_none(do_jit):
    """Ellipsis combined with None on a multi-dimensional batch shape."""

    def func(q):
        return q[None, ...]

    if do_jit:
        func = jax.jit(func)

    q = _arange_quat(2, 3)
    result = func(q)
    assert result.shape == (1, 2, 3)
    assert jnp.allclose(result.wxyz, q.wxyz[None, ...])


@pytest.mark.parametrize('do_jit', [False, True])
def test_getitem_multidim_tuple(do_jit):
    """A tuple index spanning multiple batch dimensions."""

    def func(q):
        return q[1:, 0]

    if do_jit:
        func = jax.jit(func)

    q = _arange_quat(3, 4)
    result = func(q)
    assert result.shape == (2,)
    assert jnp.allclose(result.wxyz, q.wxyz[1:, 0])


def test_getitem_scalar_index_reduces_ndim():
    """Indexing with a plain int drops that axis, exactly like a JAX array."""
    q = _arange_quat(6)
    result = q[2]
    assert result.shape == ()
    assert jnp.allclose(result.wxyz, q.wxyz[2])


def test_getitem_never_indexes_component_axis():
    """A full-rank slice tuple must still leave all 4 components intact."""
    q = _arange_quat(3)
    # `q[:]` only has one explicit index; the hidden component axis must stay whole.
    result = q[:]
    assert result.wxyz.shape == (3, 4)
    assert jnp.allclose(result.wxyz, q.wxyz)


@pytest.mark.parametrize('do_jit', [False, True])
def test_getitem_broadcast_multiplication(do_jit):
    """The pattern furax needs: q1[None, :] * q2[:, None] broadcasts like raw arrays would."""

    def func(q1, q2):
        return q1[None, :] * q2[:, None]

    if do_jit:
        func = jax.jit(func)

    q1 = Quaternion.random(jax.random.key(0), shape=(5,))
    q2 = Quaternion.random(jax.random.key(1), shape=(3,))
    result = func(q1, q2)
    assert result.shape == (3, 5)

    # Cross-check against the equivalent reshape-based broadcast (the old workaround).
    expected = q1.reshape((1, 5)) * q2.reshape((3, 1))
    assert jnp.allclose(result.wxyz, expected.wxyz)


def test_getitem_preserves_dtype():
    q = Quaternion.from_array(jnp.ones((4, 4), dtype=jnp.float16))
    assert q[1:3].dtype == jnp.float16


def test_getitem_pytree_leaf_indexing_under_vmap():
    """Quaternion indexing composes with jax.vmap, exercising the pytree registration."""
    q = _arange_quat(3, 4)

    def first_row(qi: Quaternion) -> Quaternion:
        return qi[0]

    result = jax.vmap(first_row)(q)
    assert result.shape == (3,)
    assert jnp.allclose(result.wxyz, q.wxyz[:, 0])


def test_getitem_boolean_mask_matches_numpy_reference():
    rng = np.random.default_rng(0)
    mask = rng.random(7) > 0.5
    q = _arange_quat(7)
    result = q[jnp.asarray(mask)]
    assert jnp.allclose(result.wxyz, q.wxyz[mask])
