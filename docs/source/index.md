# FastQuat - High-Performance Quaternions with JAX

[![PyPI version](https://img.shields.io/pypi/v/fastquat)](https://pypi.org/project/fastquat/)
[![Python versions](https://img.shields.io/pypi/pyversions/fastquat)](https://pypi.org/project/fastquat/)
[![Tests](https://github.com/CMBSciPol/fastquat/actions/workflows/tests.yml/badge.svg)](https://github.com/CMBSciPol/fastquat/actions)

FastQuat provides optimized quaternion operations with full JAX compatibility, featuring:

🚀 **Hardware-accelerated** computations (CPU/GPU/TPU)

🔄 **Automatic differentiation** support

🧩 **Seamless integration** with JAX transformations (`jit`, `grad`, `vmap`)

📦 **Efficient storage** using interleaved memory layout

🌐 **SLERP interpolation** for smooth rotation animations

## Quick Start

```python
from fastquat import Quaternion

# Create quaternions
q1 = Quaternion(1.0)  # Identity quaternion
q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0)  # 90° rotation around x-axis
```

## Quaternion Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Addition | `q1 + q2` | Component-wise addition |
| Subtraction | `q1 - q2` | Component-wise subtraction |
| Multiplication | `q1 * q2` | Hamilton product |
| Exponentiation | `q ** p` | Quaternion power |
| Inverse | `1 / q` | Multiplicative inverse |
| Norm | `abs(q)` | Quaternion magnitude |
| Normalization | `q.normalize()` | Unit quaternion |
| Conjugate | `q.conj()` | Quaternion conjugate |
| Rotation | `q.rotate_vector(v)` | Rotate 3D vector |
| SLERP | `q1.slerp(q2, t)` | Spherical interpolation |
| Log | `q.log()` | Quaternion logarithm |
| Exp | `q.exp()` | Quaternion exponential |

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/installation
user-guide/getting-started
user-guide/tutorial-rotations
user-guide/tutorial-slerp
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/quaternion
```

```{toctree}
:maxdepth: 1
:caption: Development

development
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
