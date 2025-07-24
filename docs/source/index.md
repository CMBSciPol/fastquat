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
import jax.numpy as jnp
from fastquat import Quaternion

# Create quaternions
q1 = Quaternion.ones()  # Identity quaternion
q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0)  # 90° rotation around x-axis

# Quaternion operations
q3 = q1 * q2  # Multiplication
q_inv = 1 / q1  # Inverse
q_norm = q1.normalize()  # Normalization

# Rotate vectors
vector = jnp.array([1.0, 0.0, 0.0])
rotated = q2.rotate_vector(vector)

# Spherical interpolation (SLERP)
interpolated = q1.slerp(q2, t=0.5)  # Halfway between q1 and q2
```

## Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

installation.md
quickstart.md
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/index.md
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Development

development.md
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
