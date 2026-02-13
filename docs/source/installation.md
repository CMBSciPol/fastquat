# Installation

## Requirements

FastQuat requires:

* Python 3.10 or later
* JAX 0.4.0 or later

## Installing from PyPI

The recommended way to install FastQuat is via pip:

```bash
pip install fastquat
```

This will install FastQuat with CPU support. For GPU support, you may need to install JAX with CUDA support:

```bash
pip install "jax[cuda12]" fastquat
```

## Development Installation

For development, additional dependencies should be installed and it can be streamlined using the
[uv package manager](https://docs.astral.sh/uv/getting-started/installation).

```bash
git clone https://github.com/CMBSciPol/fastquat
cd fastquat
uv sync --group cuda12
source .venv/bin/activate

# To run QA
uv tool install pre-commit
pre-commit install
```

This includes:

* pytest for testing
* pre-commit for code formatting and linting


## Verification

To verify your installation, run:

```python
from fastquat import Quaternion

# Create a simple quaternion
q = Quaternion(1.0)
print(f"Identity quaternion: {q}")

# Test SLERP functionality
q2 = Quaternion(0.7071, 0.7071, 0.0, 0.0)
interpolated = q.slerp(q2, 0.5)
print(f"SLERP result: {interpolated}")
```

If this runs without errors, FastQuat is properly installed!
