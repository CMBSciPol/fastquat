"""Base functionality tests for Quaternion class.

Tests for factories, properties (w, x, y, z, vector), to_components, and __repr__.
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.typing import DTypeLike

from fastquat.quaternion import Quaternion


@pytest.mark.parametrize(
    'components, dtype, expected_dtype',
    [
        ((1.0, 0.0, 0.0, 0.0), None, jnp.float64),
        ((1.0, jnp.float16(0), False, np.float64(0)), None, jnp.float64),
        ((np.bool_(True), np.int32(0), 0, np.float16(0)), None, jnp.float16),
        ((1.0, 0.0, 0.0, 0.0), jnp.float32, jnp.float32),
        ((1.0, jnp.float16(0), False, np.float64(0)), jnp.float32, jnp.float32),
        ((np.bool_(True), np.int32(0), 0, np.float16(0)), jnp.int8, jnp.int8),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_init_creation_dtype(enable_x64: None, components, dtype, expected_dtype, do_jit: bool):
    """Test quaternion creation from individual components."""

    def create(w, x, y, z, dtype_):
        return Quaternion(w, x, y, z, dtype=dtype_)

    if do_jit:
        create = jax.jit(create, static_argnums=4)

    q = create(*components, dtype)
    assert q.wxyz.shape == (4,)
    assert q.wxyz.dtype == expected_dtype
    assert jnp.allclose(q.wxyz, jnp.array([1, 0, 0, 0]))


@pytest.mark.parametrize(
    'components, expected',
    [
        ({}, Quaternion(0.0, 0.0, 0.0, 0.0)),
        ({'w': 1.0}, Quaternion(1.0, 0.0, 0.0, 0.0)),
        ({'x': 1.0}, Quaternion(0.0, 1.0, 0.0, 0.0)),
        ({'y': 1.0}, Quaternion(0.0, 0.0, 1.0, 0.0)),
        ({'z': 1.0}, Quaternion(0.0, 0.0, 0.0, 1.0)),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_init_creation_partial(components: dict[str, float], expected: Quaternion, do_jit: bool):
    """Test quaternion creation from individual components."""

    def create(components_):
        return Quaternion(**components_)

    if do_jit:
        create = jax.jit(create)

    q = create(components)
    assert q.shape == ()
    assert jnp.allclose(q.wxyz, expected.wxyz)


@pytest.mark.parametrize('do_jit', [False, True])
def test_init_creation_broadcasting(do_jit: bool):
    """Test quaternion creation with broadcasting."""

    def create(w, x, y, z) -> Quaternion:
        return Quaternion(w, x, y, z)

    if do_jit:
        create = jax.jit(create)

    # Test broadcasting scalar with array
    w = 1.0
    x = jnp.array([1.0, 2.0])
    y = 0.0
    z = 0.0

    q = create(w, x, y, z)
    assert q.shape == (2,)
    expected_wxyz = jnp.array([[1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
    assert jnp.allclose(q.wxyz, expected_wxyz)


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_array_creation(do_jit: bool):
    """Test quaternion creation from array."""

    def create(array_):
        return Quaternion.from_array(array_)

    if do_jit:
        create = jax.jit(create)

    array = jnp.array([1.0, 0.0, 0.0, 0.0])
    q = create(array)
    assert q.shape == ()
    assert jnp.allclose(q.wxyz, array)


# Edge cases
@pytest.mark.parametrize('do_jit', [False, True])
def test_from_array_wrong_shape(do_jit: bool):
    """Test from_array with wrong shape raises ValueError."""

    def create(arr):
        return Quaternion.from_array(arr)

    if do_jit:
        create = jax.jit(create)

    with pytest.raises(ValueError):
        create(jnp.ones((3, 5)))  # Wrong last dimension


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_scalar_vector_creation(do_jit: bool):
    """Test quaternion creation from scalar and vector parts."""

    def create(scalar_, vector_):
        return Quaternion.from_scalar_vector(scalar_, vector_)

    if do_jit:
        create = jax.jit(create)

    scalar = jnp.array(1.0)
    vector = jnp.array([0.0, 0.0, 0.0])
    q = create(scalar, vector)
    expected = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(q.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_scalar_vector_wrong_shape(do_jit: bool):
    """Test from_scalar_vector with wrong vector shape raises ValueError."""

    def create(scalar_, vector_):
        return Quaternion.from_scalar_vector(scalar_, vector_)

    if do_jit:
        create = jax.jit(create)

    scalar = jnp.array(1.0)
    vector = jnp.ones((2,))  # Wrong vector dimension
    with pytest.raises(ValueError):
        create(scalar, vector)


@pytest.mark.parametrize(
    'shape, dtype',
    [
        ((), jnp.float16),
        ((0,), jnp.float32),
        ((2, 3), jnp.float64),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_zeros_factory(enable_x64: None, shape: tuple[int, ...], dtype: DTypeLike, do_jit: bool):
    """Test Quaternion.zeros() creates quaternions with all components zero."""

    def create(shape_: tuple[int, ...], dtype_: DTypeLike) -> Quaternion:
        return Quaternion.zeros(shape_, dtype_)

    if do_jit:
        create = jax.jit(create, static_argnums=(0, 1))

    q = create(shape, dtype)

    assert q.wxyz.shape == shape + (4,)
    assert q.wxyz.dtype == dtype
    expected_array = jnp.zeros(shape + (4,), dtype)
    assert jnp.allclose(q.wxyz, expected_array)


@pytest.mark.parametrize(
    'shape, dtype',
    [
        ((), jnp.float16),
        ((0,), jnp.float32),
        ((2, 3), jnp.float64),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_ones_factory(enable_x64: None, shape: tuple[int, ...], dtype: DTypeLike, do_jit: bool):
    """Test Quaternion(1.0) creates quaternions with w=1, x=y=z=0."""

    def create(shape_: tuple[int, ...], dtype_: DTypeLike) -> Quaternion:
        return Quaternion.ones(shape_, dtype_)

    if do_jit:
        create = jax.jit(create, static_argnums=(0, 1))

    q = create(shape, dtype)

    assert q.wxyz.shape == shape + (4,)
    assert q.wxyz.dtype == dtype
    expected_array = jnp.broadcast_to(jnp.array([1, 0, 0, 0], dtype=dtype), shape + (4,))
    assert jnp.allclose(q.wxyz, expected_array)


@pytest.mark.parametrize(
    'shape, dtype',
    [
        ((), jnp.float16),
        ((0,), jnp.float32),
        ((2, 3), jnp.float64),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_full_factory(enable_x64: None, shape: tuple[int, ...], dtype: DTypeLike, do_jit: bool):
    """Test Quaternion(1.0) creates quaternions with w=1, x=y=z=0."""

    def create(shape_: tuple[int, ...], dtype_: DTypeLike) -> Quaternion:
        return Quaternion.full(shape_, 2.5, dtype_)

    if do_jit:
        create = jax.jit(create, static_argnums=(0, 1))

    q = create(shape, dtype)

    assert q.wxyz.shape == shape + (4,)
    assert q.wxyz.dtype == dtype
    expected_array = jnp.broadcast_to(jnp.array([2.5, 0, 0, 0], dtype=dtype), shape + (4,))
    assert jnp.allclose(q.wxyz, expected_array)


@pytest.mark.parametrize(
    'shape, dtype',
    [
        ((), jnp.float16),
        ((0,), jnp.float32),
        ((2, 3), jnp.float64),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_random_factory(enable_x64: None, shape: tuple[int, ...], dtype: DTypeLike, do_jit: bool):
    """Test random quaternion creation."""

    def create(key_, shape_, dtype_):
        return Quaternion.random(key_, shape_, dtype_)

    if do_jit:
        create = jax.jit(create, static_argnums=(1, 2))

    key = jr.key(42)
    q = create(key, shape, dtype)
    assert q.wxyz.shape == shape + (4,)
    assert q.wxyz.dtype == dtype
    assert jnp.allclose(q.norm(), 1.0)  # Should be normalized


def test_factories_default_dtype():
    """Test factory methods respect dtype parameter."""
    shape = (2, 3)

    # Test float32 (default)
    q_zeros = Quaternion.zeros(shape)
    assert q_zeros.dtype == jnp.float32

    q_ones = Quaternion.ones(shape)
    assert q_ones.dtype == jnp.float32

    q_full = Quaternion.full(shape, 1.5)
    assert q_full.dtype == jnp.float32


def test_factory_methods_empty_shape():
    """Test factory methods work with empty shapes."""
    q = Quaternion.zeros((0,))
    assert q.shape == (0,)
    assert q.wxyz.shape == (0, 4)

    q = Quaternion.ones((0,))
    assert q.shape == (0,)
    assert q.wxyz.shape == (0, 4)

    q = Quaternion.full((0,), 2.0)
    assert q.shape == (0,)
    assert q.wxyz.shape == (0, 4)

    key = jr.key(0)
    q = Quaternion.random(key, (0,))
    assert q.shape == (0,)
    assert q.wxyz.shape == (0, 4)


@pytest.mark.parametrize('component', 'wxyz')
@pytest.mark.parametrize('do_jit', [False, True])
def test_component_property(do_jit: bool, component: str):
    """Test w property access."""

    def get(q, component_):
        return getattr(q, component_)

    if do_jit:
        get = jax.jit(get, static_argnums=1)

    values = {'w': 1.0, 'x': 2.0, 'y': 3.0, 'z': 4.0}
    q = Quaternion(**values)
    value = get(q, component)
    assert jnp.allclose(value, values[component])


@pytest.mark.parametrize('do_jit', [False, True])
def test_batch_component_properties(do_jit: bool):
    """Test properties work with batched quaternions."""

    def get(q: Quaternion):
        return q.w, q.x, q.y, q.z, q.vector

    if do_jit:
        get = jax.jit(get)

    # Create batch of quaternions
    q_batch = Quaternion.from_array(jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))

    w, x, y, z, vector = get(q_batch)

    assert jnp.allclose(w, jnp.array([1.0, 5.0]))
    assert jnp.allclose(x, jnp.array([2.0, 6.0]))
    assert jnp.allclose(y, jnp.array([3.0, 7.0]))
    assert jnp.allclose(z, jnp.array([4.0, 8.0]))
    assert jnp.allclose(vector, jnp.array([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]]))


@pytest.mark.parametrize('do_jit', [False, True])
def test_vector_property(do_jit: bool):
    """Test vector property access."""

    def get(q):
        return q.vector

    if do_jit:
        get = jax.jit(get)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    vector = get(q)
    expected = jnp.array([2.0, 3.0, 4.0])
    assert jnp.allclose(vector, expected)


# to_components method
@pytest.mark.parametrize('do_jit', [False, True])
def test_to_components(do_jit: bool):
    """Test to_components method."""

    def get(q):
        return q.to_components()

    if do_jit:
        get = jax.jit(get)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    w, x, y, z = get(q)
    assert jnp.allclose(w, 1.0)
    assert jnp.allclose(x, 2.0)
    assert jnp.allclose(y, 3.0)
    assert jnp.allclose(z, 4.0)


@pytest.mark.parametrize('shape', [(0,), (2,), (2, 3)])
@pytest.mark.parametrize('do_jit', [False, True])
def test_iter(do_jit: bool, shape: tuple[int, ...]):
    """Test iteration over 1-dimensional quaternion array."""

    def iterate(q_array: Quaternion):
        # This function uses iteration implicitly
        return list(q_array)

    if do_jit:
        iterate = jax.jit(iterate)

    arrays = jnp.arange(math.prod(shape) * 4).reshape(*shape, 4)
    quaternions = Quaternion.from_array(arrays)
    iterated_quaternions = iterate(quaternions)

    assert len(iterated_quaternions) == shape[0]
    for quaternion, array in zip(quaternions, arrays):
        assert quaternion.ndim == len(shape[1:])
        assert quaternion.shape == shape[1:]
        assert jnp.allclose(quaternion.wxyz, jnp.asarray(array))


# Iteration tests
def test_iter_0d():
    """Test that iteration over 0-d quaternion raises TypeError."""
    q = Quaternion(1.0)  # 0-dimensional quaternion
    assert q.ndim == 0

    with pytest.raises(TypeError, match='iteration over a 0-d quaternion'):
        _ = list(q)

    with pytest.raises(TypeError, match='iteration over a 0-d quaternion'):
        for _ in q:
            pass


# __repr__ method
def test_repr_scalar():
    """Test __repr__ for scalar quaternion."""
    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    repr_str = repr(q)
    assert '1.0 + 2.0i + 3.0j + 4.0k' in repr_str


def test_repr_tensor():
    """Test __repr__ for tensor quaternion."""
    q = Quaternion.from_array(jnp.ones((2, 3, 4)))
    repr_str = repr(q)
    assert 'shape=(2, 3)' in repr_str
    assert 'dtype=' in repr_str
