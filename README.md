# FastQuat - High-Performance Quaternions with JAX

[![PyPI version](https://img.shields.io/pypi/v/fastquat)](https://pypi.org/project/fastquat/)
[![Python versions](https://img.shields.io/pypi/pyversions/fastquat)](https://pypi.org/project/fastquat/)
[![Tests](https://github.com/CMBSciPol/fastquat/actions/workflows/tests.yml/badge.svg)](https://github.com/CMBSciPol/fastquat/actions)

FastQuat provides optimized quaternion operations with full JAX compatibility, featuring:

- 🚀 **Hardware-accelerated** computations (CPU/GPU/TPU)
- 🔄 **Automatic differentiation** support
- 🧩 **Seamless integration** with JAX transformations (`jit`, `grad`, `vmap`)
- 📦 **Efficient storage** using interleaved memory layout

## Installation

```bash
pip install fastquat
```
This will install FastQuat with CPU support. For GPU support, you may need to install JAX with CUDA support:

```bash
pip install "jax[cuda12]" fastquat
```


## Quick Start

```python
import jax.numpy as jnp
from fastquat import Quaternion

# Create quaternions
q1 = Quaternion.ones()  # Identity quaternion
q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0)  # 90° rotation around x-axis

# Quaternion operations
q3 = q1 * q2  # Multiplication
q_inv = 1 / q1           # Inverse, or q1 ** -1
q_norm = q1.normalize()  # Normalization

# Rotate vectors
vector = jnp.array([1.0, 0.0, 0.0])
rotated = q2.rotate_vector(vector)

# Spherical interpolation (SLERP)
interpolated = q1.slerp(q2, t=0.5)  # Halfway between q1 and q2
```

## Features

### Core Operations
- **Quaternion arithmetic**: Addition, multiplication, conjugation, inverse, power, exponentiation, logarithm
- **Normalization**: Efficient unit quaternion computation
- **Conversion**: To/from rotation matrices, Euler angles
- **Vector rotation**: Direct vector transformation

### Advanced Interpolation
- **SLERP (Spherical Linear Interpolation)**: Smooth rotation interpolation
  - Automatically handles shortest path selection
  - Numerically stable for close quaternions
  - Supports batched operations and array-valued parameters

### JAX Integration
- **JIT compilation**: Compile quaternion operations for maximum performance
- **Automatic differentiation**: Compute gradients through quaternion operations
- **Vectorization**: Process batches of quaternions efficiently
- **Device support**: Run on CPU, GPU, or TPU

## Performance

FastQuat is optimized for high-performance computing:
- Memory-efficient interleaved storage
- SIMD-optimized operations on supported hardware
- Zero-copy integration with JAX arrays
- Minimal Python overhead through JIT compilation

## Examples

### Basic Usage
```python
import jax
import jax.numpy as jnp
from fastquat import Quaternion

# Create random quaternions
key = jax.random.PRNGKey(42)
q_batch = Quaternion.random(key, shape=(1000,))

# JIT-compiled batch operations
@jax.jit
def batch_rotate(quaternions, vectors):
    return quaternions.rotate_vector(vectors)

vectors = jax.random.normal(key, (1000, 3))
rotated_batch = batch_rotate(q_batch, vectors)
```

### SLERP
```python
# Smooth rotation interpolation
q_start = Quaternion.ones()
q_end = Quaternion.from_rotation_matrix(rotation_matrix)

# Generate smooth interpolation
t_values = jnp.linspace(0, 1, 100)
interpolated_rotations = q_start.slerp(q_end, t_values)

# Apply to object vertices for smooth animation
animated_vertices = interpolated_rotations.rotate_vector(object_vertices)
```

## Documentation

Full documentation is available at [fastquat.readthedocs.io](https://fastquat.readthedocs.io)

## Contributing

Contributions are welcome! Please see our [development guide](https://fastquat.readthedocs.io/en/latest/development.html) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.
