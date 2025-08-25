# Quick Start Guide

This guide will get you started with FastQuat's core functionality.

## Creating Quaternions

There are several ways to create quaternions:

```python
import jax.numpy as jnp
from fastquat import Quaternion

# From components (w, x, y, z)
q1 = Quaternion(1.0, 0.0, 0.0, 0.0)  # Identity quaternion

# Convenience constructors
q_identity = Quaternion.ones()  # Identity quaternion
q_zero = Quaternion.zeros()     # Zero quaternion

# From arrays
array = jnp.array([1.0, 0.0, 0.0, 0.0])
q2 = Quaternion.from_array(array)

# Random normalized quaternions
import jax
key = jax.random.PRNGKey(42)
q_random = Quaternion.random(key)
```

## Basic Operations

Quaternions support standard mathematical operations:

```python
# Arithmetic
q_sum = q1 + q2
q_diff = q1 - q2
q_product = q1 * q2
q_power = q1 ** p

# Normalization
q_unit = q1.normalize()  # Unit quaternion
norm = q1.norm()         # Quaternion norm

# Conjugation and inverse
q_conj = q1.conj()       # Conjugate
q_inv = 1 / q1           # Inverse, or q1 ** -1

# Other operations
q_log = q1.log()
q_exp = q1.exp()
```

## Vector Rotation

One of the most common uses of quaternions is rotating 3D vectors:

```python
# Create a 90° rotation around the z-axis
angle = jnp.pi / 2
q_rot = Quaternion(jnp.cos(angle/2), 0.0, 0.0, jnp.sin(angle/2))

# Rotate a vector
vector = jnp.array([1.0, 0.0, 0.0])  # Unit vector along x
rotated = q_rot.rotate_vector(vector)
print(rotated)  # Should be approximately [0, 1, 0]
```

## Conversion to/from Rotation Matrices

FastQuat can convert between quaternions and rotation matrices:

```python
# Quaternion to rotation matrix
R = q_rot.to_rotation_matrix()
print(R.shape)  # (3, 3)

# Rotation matrix to quaternion
q_from_matrix = Quaternion.from_rotation_matrix(R)
```

## Spherical Linear Interpolation (SLERP)

SLERP provides smooth interpolation between quaternions:

```python
# Two different orientations
q_start = Quaternion.ones()  # Identity
q_end = Quaternion(0.7071, 0.7071, 0.0, 0.0)  # 90° around x

# Interpolate between them
t = 0.5  # Halfway point
q_mid = q_start.slerp(q_end, t)

# Batch interpolation
t_values = jnp.linspace(0, 1, 10)
interpolated = q_start.slerp(q_end, t_values)
print(interpolated.shape)  # (10,) - 10 quaternions
```

## JAX Integration

FastQuat is fully compatible with JAX transformations:

```python
import jax

# JIT compilation
@jax.jit
def rotate_and_normalize(q, v):
    rotated = q.rotate_vector(v)
    return rotated / jnp.linalg.norm(rotated)

# Vectorization
batch_rotate = jax.vmap(lambda q, v: q.rotate_vector(v))

# Create batches
q_batch = Quaternion.random(key, shape=(100,))
v_batch = jax.random.normal(key, (100, 3))

# Process entire batch at once
rotated_batch = batch_rotate(q_batch, v_batch)

# Automatic differentiation
def loss_function(q_params):
    q = Quaternion.from_array(q_params)
    rotated = q.rotate_vector(vector)
    return jnp.sum(rotated**2)

grad_fn = jax.grad(loss_function)
gradients = grad_fn(jnp.array([1.0, 0.1, 0.1, 0.1]))
```

## Performance Tips

1. **Use JIT compilation** for repeated operations:

   ```python
   @jax.jit
   def batch_operation(quaternions):
       return quaternions.normalize()
   ```

2. **Prefer batch operations** over loops:

   ```python
   # Good: vectorized operation
   results = q_batch.rotate_vector(v_batch)

   # Avoid: Python loops
   # results = [q.rotate_vector(v) for q, v in zip(q_batch, v_batch)]
   ```

3. **Normalize quaternions** when needed for rotations:

   ```python
   q_unit = q.normalize()  # Ensure unit quaternion for rotations
   ```

## Next Steps

* Explore the [examples](examples/index.md) for detailed use cases
* Check the [API reference](api/index.md) for complete API documentation
* See advanced interpolation techniques with SLERP
