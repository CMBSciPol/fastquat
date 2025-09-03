"""Base functionality tests for Quaternion class.

Tests for factories, properties (w, x, y, z, vector), to_components, and __repr__.
"""

import jax
import jax.numpy as jnp
import pytest

from fastquat.quaternion import Quaternion


# Factories
@pytest.mark.parametrize('do_jit', [False, True])
def test_init_creation(do_jit: bool):
    """Test quaternion creation from individual components."""

    def create_quaternion(w, x, y, z):
        return Quaternion(w, x, y, z)

    if do_jit:
        create_quaternion = jax.jit(create_quaternion)

    q = create_quaternion(1.0, 0.0, 0.0, 0.0)
    assert q.shape == ()
    assert jnp.allclose(q.wxyz, jnp.array([1.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_array_creation(do_jit: bool):
    """Test quaternion creation from array."""

    def create_from_array(arr):
        return Quaternion.from_array(arr)

    if do_jit:
        create_from_array = jax.jit(create_from_array)

    arr = jnp.array([1.0, 0.0, 0.0, 0.0])
    q = create_from_array(arr)
    assert q.shape == ()
    assert jnp.allclose(q.wxyz, arr)


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_scalar_vector_creation(do_jit: bool):
    """Test quaternion creation from scalar and vector parts."""

    def create_from_scalar_vector(scalar, vector):
        return Quaternion.from_scalar_vector(scalar, vector)

    if do_jit:
        create_from_scalar_vector = jax.jit(create_from_scalar_vector)

    scalar = jnp.array(1.0)
    vector = jnp.array([0.0, 0.0, 0.0])
    q = create_from_scalar_vector(scalar, vector)
    expected = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(q.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_random_creation(do_jit: bool):
    """Test random quaternion creation."""

    def create_random(key):
        return Quaternion.random(key, ())

    def create_random_batch(key):
        return Quaternion.random(key, (5, 3))

    if do_jit:
        create_random = jax.jit(create_random)
        create_random_batch = jax.jit(create_random_batch)

    key = jax.random.PRNGKey(42)
    q = create_random(key)
    assert q.shape == ()
    assert jnp.allclose(q.norm(), 1.0)  # Should be normalized

    # Test batch creation
    q_batch = create_random_batch(key)
    assert q_batch.shape == (5, 3)
    assert jnp.allclose(q_batch.norm(), 1.0)


# Properties
@pytest.mark.parametrize('do_jit', [False, True])
def test_w_property(do_jit: bool):
    """Test w property access."""

    def get_w(q):
        return q.w

    if do_jit:
        get_w = jax.jit(get_w)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    w = get_w(q)
    assert jnp.allclose(w, 1.0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_x_property(do_jit: bool):
    """Test x property access."""

    def get_x(q):
        return q.x

    if do_jit:
        get_x = jax.jit(get_x)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    x = get_x(q)
    assert jnp.allclose(x, 2.0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_y_property(do_jit: bool):
    """Test y property access."""

    def get_y(q):
        return q.y

    if do_jit:
        get_y = jax.jit(get_y)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    y = get_y(q)
    assert jnp.allclose(y, 3.0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_z_property(do_jit: bool):
    """Test z property access."""

    def get_z(q):
        return q.z

    if do_jit:
        get_z = jax.jit(get_z)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    z = get_z(q)
    assert jnp.allclose(z, 4.0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_vector_property(do_jit: bool):
    """Test vector property access."""

    def get_vector(q):
        return q.vector

    if do_jit:
        get_vector = jax.jit(get_vector)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    vector = get_vector(q)
    expected = jnp.array([2.0, 3.0, 4.0])
    assert jnp.allclose(vector, expected)


# to_components method
@pytest.mark.parametrize('do_jit', [False, True])
def test_to_components(do_jit: bool):
    """Test to_components method."""

    def get_components(q):
        return q.to_components()

    if do_jit:
        get_components = jax.jit(get_components)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    w, x, y, z = get_components(q)
    assert jnp.allclose(w, 1.0)
    assert jnp.allclose(x, 2.0)
    assert jnp.allclose(y, 3.0)
    assert jnp.allclose(z, 4.0)


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


# Edge cases
@pytest.mark.parametrize('do_jit', [False, True])
def test_from_array_wrong_shape(do_jit: bool):
    """Test from_array with wrong shape raises ValueError."""

    def create_from_wrong_array(arr):
        return Quaternion.from_array(arr)

    if do_jit:
        create_from_wrong_array = jax.jit(create_from_wrong_array)

    with pytest.raises(ValueError):
        create_from_wrong_array(jnp.ones((3, 5)))  # Wrong last dimension


@pytest.mark.parametrize('do_jit', [False, True])
def test_from_scalar_vector_wrong_shape(do_jit: bool):
    """Test from_scalar_vector with wrong vector shape raises ValueError."""

    def create_from_wrong_vector(scalar, vector):
        return Quaternion.from_scalar_vector(scalar, vector)

    if do_jit:
        create_from_wrong_vector = jax.jit(create_from_wrong_vector)

    scalar = jnp.array(1.0)
    vector = jnp.ones((2,))  # Wrong vector dimension
    with pytest.raises(ValueError):
        create_from_wrong_vector(scalar, vector)


@pytest.mark.parametrize('do_jit', [False, True])
def test_broadcasting_creation(do_jit: bool):
    """Test quaternion creation with broadcasting."""

    def create_with_broadcasting(w, x, y, z) -> Quaternion:
        return Quaternion(w, x, y, z)

    if do_jit:
        create_with_broadcasting = jax.jit(create_with_broadcasting)

    # Test broadcasting scalar with array
    w = 1.0
    x = jnp.array([1.0, 2.0])
    y = 0.0
    z = 0.0

    q = create_with_broadcasting(w, x, y, z)
    assert q.shape == (2,)
    expected_wxyz = jnp.array([[1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
    assert jnp.allclose(q.wxyz, expected_wxyz)


@pytest.mark.parametrize('do_jit', [False, True])
def test_batch_properties(do_jit: bool):
    """Test properties work with batched quaternions."""

    def get_all_properties(q: Quaternion):
        return q.w, q.x, q.y, q.z, q.vector

    if do_jit:
        get_all_properties = jax.jit(get_all_properties)

    # Create batch of quaternions
    q_batch = Quaternion.from_array(jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))

    w, x, y, z, vector = get_all_properties(q_batch)

    assert jnp.allclose(w, jnp.array([1.0, 5.0]))
    assert jnp.allclose(x, jnp.array([2.0, 6.0]))
    assert jnp.allclose(y, jnp.array([3.0, 7.0]))
    assert jnp.allclose(z, jnp.array([4.0, 8.0]))
    assert jnp.allclose(vector, jnp.array([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]]))


# Factory methods tests
@pytest.mark.parametrize('do_jit', [False, True])
def test_zeros_factory(do_jit: bool):
    """Test Quaternion.zeros() creates quaternions with all components zero."""

    def create_zeros(shape: tuple[int, ...]) -> Quaternion:
        return Quaternion.zeros(shape)

    if do_jit:
        create_zeros = jax.jit(create_zeros, static_argnames=['shape'])

    # Test scalar
    q = create_zeros(())
    assert q.shape == ()
    assert jnp.allclose(q.wxyz, jnp.zeros(4))

    # Test tensor
    q = create_zeros((2, 3))
    assert q.shape == (2, 3)
    assert jnp.allclose(q.wxyz, jnp.zeros((2, 3, 4)))


@pytest.mark.parametrize('do_jit', [False, True])
def test_ones_factory(do_jit: bool):
    """Test Quaternion.ones() creates quaternions with w=1, x=y=z=0."""

    def create_ones(shape: tuple[int, ...]) -> Quaternion:
        return Quaternion.ones(shape)

    if do_jit:
        create_ones = jax.jit(create_ones, static_argnames=['shape'])

    # Test scalar
    q = create_ones(())
    assert q.shape == ()
    assert jnp.allclose(q.wxyz, jnp.array([1.0, 0.0, 0.0, 0.0]))

    # Test tensor
    q = create_ones((2, 3))
    assert q.shape == (2, 3)
    expected = jnp.zeros((2, 3, 4))
    expected = expected.at[..., 0].set(1.0)
    assert jnp.allclose(q.wxyz, expected)


@pytest.mark.parametrize('do_jit', [False, True])
def test_full_factory(do_jit: bool):
    """Test Quaternion.full() creates quaternions with w=fill_value, x=y=z=0."""

    def create_full(shape: tuple[int, ...], fill_value: float) -> Quaternion:
        return Quaternion.full(shape, fill_value)

    if do_jit:
        create_full = jax.jit(create_full, static_argnames=['shape'])

    fill_value = 2.5

    # Test scalar
    q = create_full((), fill_value)
    assert q.shape == ()
    assert jnp.allclose(q.wxyz, jnp.array([fill_value, 0.0, 0.0, 0.0]))

    # Test tensor
    q = create_full((2, 3), fill_value)
    assert q.shape == (2, 3)
    expected = jnp.zeros((2, 3, 4))
    expected = expected.at[..., 0].set(fill_value)
    assert jnp.allclose(q.wxyz, expected)


def test_factory_methods_dtype():
    """Test factory methods respect dtype parameter."""
    shape = (2, 3)

    # Test float32 (default)
    q_zeros = Quaternion.zeros(shape)
    assert q_zeros.dtype == jnp.float32

    q_ones = Quaternion.ones(shape)
    assert q_ones.dtype == jnp.float32

    q_full = Quaternion.full(shape, 1.5)
    assert q_full.dtype == jnp.float32

    # Test float16
    q_zeros_f16 = Quaternion.zeros(shape, dtype=jnp.float16)
    assert q_zeros_f16.dtype == jnp.float16

    q_ones_f16 = Quaternion.ones(shape, dtype=jnp.float16)
    assert q_ones_f16.dtype == jnp.float16

    q_full_f16 = Quaternion.full(shape, 1.5, dtype=jnp.float16)
    assert q_full_f16.dtype == jnp.float16


def test_factory_methods_empty_shape():
    """Test factory methods work with empty shapes."""
    q_zeros = Quaternion.zeros((0,))
    assert q_zeros.shape == (0,)
    assert q_zeros.wxyz.shape == (0, 4)

    q_ones = Quaternion.ones((0,))
    assert q_ones.shape == (0,)
    assert q_ones.wxyz.shape == (0, 4)

    q_full = Quaternion.full((0,), 2.0)
    assert q_full.shape == (0,)
    assert q_full.wxyz.shape == (0, 4)


# Iteration tests
def test_iter_0d_quaternion():
    """Test that iteration over 0-d quaternion raises TypeError."""
    q = Quaternion.ones()  # 0-dimensional quaternion
    assert q.ndim == 0

    with pytest.raises(TypeError, match='iteration over a 0-d quaternion'):
        _ = list(q)

    with pytest.raises(TypeError, match='iteration over a 0-d quaternion'):
        for _ in q:
            pass


def test_iter_1d_quaternion():
    """Test iteration over 1-dimensional quaternion array."""
    # Create a 1D array of quaternions
    q_array = Quaternion.from_array(
        jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    )

    assert q_array.ndim == 1
    assert q_array.shape == (3,)

    # Test iteration
    quaternions = list(q_array)

    assert len(quaternions) == 3

    # Check each quaternion
    assert jnp.allclose(quaternions[0].wxyz, jnp.array([1.0, 0.0, 0.0, 0.0]))
    assert jnp.allclose(quaternions[1].wxyz, jnp.array([0.0, 1.0, 0.0, 0.0]))
    assert jnp.allclose(quaternions[2].wxyz, jnp.array([0.0, 0.0, 1.0, 0.0]))

    # Each element should be 0-dimensional
    for q in quaternions:
        assert q.ndim == 0
        assert q.shape == ()


def test_iter_2d_quaternion():
    """Test iteration over 2-dimensional quaternion array."""
    # Create a 2D array of quaternions (2x3)
    q_array = Quaternion.ones((2, 3))  # All identity quaternions

    assert q_array.ndim == 2
    assert q_array.shape == (2, 3)

    # Test iteration - should iterate over first axis
    rows = list(q_array)

    assert len(rows) == 2

    # Each row should be 1-dimensional with shape (3,)
    for row in rows:
        assert row.ndim == 1
        assert row.shape == (3,)

        # Each element in the row should be identity quaternion
        for q in row:
            assert q.ndim == 0
            assert jnp.allclose(q.wxyz, jnp.array([1.0, 0.0, 0.0, 0.0]))


def test_iter_empty_quaternion():
    """Test iteration over empty quaternion array."""
    q_empty = Quaternion.zeros((0,))

    assert q_empty.ndim == 1
    assert q_empty.shape == (0,)

    # Should be iterable but produce no elements
    quaternions = list(q_empty)
    assert len(quaternions) == 0


@pytest.mark.parametrize('do_jit', [False, True])
def test_iter(do_jit: bool):
    """Test that iteration works with JIT compilation context."""

    def use_iteration(q_array: Quaternion):
        # This function uses iteration implicitly
        result = []
        for q in q_array:
            result.append(q.norm())
        return jnp.array(result)

    if do_jit:
        use_iteration = jax.jit(use_iteration)

    q_array = Quaternion.from_array(
        jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    )

    norms = use_iteration(q_array)
    expected_norms = jnp.array([1.0, 1.0, 1.0])
    assert jnp.allclose(norms, expected_norms)
