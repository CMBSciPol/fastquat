# Installation

## Requirements

FastQuat requires:

* Python 3.9 or later
* JAX 0.4.0 or later
* JAXLib 0.4.0 or later

## Installing from PyPI

The recommended way to install FastQuat is via pip:

```bash
pip install fastquat
```

This will install FastQuat with CPU support. For GPU support, you may need to install JAX with CUDA support:

```bash
pip install "jax[cuda12]" fastquat
```

## Installing from Source

To install the latest development version:

```bash
git clone https://github.com/CMBSciPol/fastquat.git
cd fastquat
pip install -e .
```

## Development Installation

For development, install with additional dependencies:

```bash
git clone https://github.com/CMBSciPol/fastquat.git
cd fastquat
pip install -e ".[dev]"
```

This includes:

* pytest for testing
* ruff for code formatting and linting
* ipython for interactive development

## Verification

To verify your installation, run:

```python
import fastquat
from fastquat import Quaternion

# Create a simple quaternion
q = Quaternion.ones()
print(f"Identity quaternion: {q}")

# Test SLERP functionality
q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0)
interpolated = q.slerp(q2, 0.5)
print(f"SLERP result: {interpolated}")
```

If this runs without errors, FastQuat is properly installed!
