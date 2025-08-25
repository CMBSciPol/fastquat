# Documentation Build Instructions

This documentation uses **PEP 735 dependency groups** instead of separate requirements files to avoid duplication.

## Building Documentation Locally

### Using uv (recommended)

```bash
# From the docs/ directory
cd docs/
uv run --group docs make html

# Or from the root directory
uv run --group docs sphinx-build docs/source docs/build/html
```

### Using other tools

If you're not using uv, you can install the docs dependencies manually:

```bash
# Install the project with docs dependencies
pip install -e .[docs]

# Then build the docs
cd docs/
make html
```

## What Changed

### ✅ Before (duplicated dependencies)
- Dependencies listed in both `pyproject.toml` and `docs/requirements.txt`
- ReadTheDocs config pointed to `docs/requirements.txt`

### ✅ After (PEP 735 dependency groups)
- Dependencies only in `pyproject.toml` under `[dependency-groups.docs]`
- ReadTheDocs config uses `extra_requirements: [docs]`
- No more `docs/requirements.txt` file

## PEP 735 Benefits

1. **Single source of truth** - dependencies defined once in `pyproject.toml`
2. **No duplication** - eliminates sync issues between files
3. **Modern standard** - follows PEP 735 specification
4. **Tool support** - works with uv, pip, and ReadTheDocs

## ReadTheDocs Configuration

The `.readthedocs.yaml` configuration now uses:

```yaml
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

This automatically installs the project with the `docs` dependency group.

## Available Commands

```bash
# Build HTML documentation
uv run --group docs make html

# Clean and rebuild
uv run --group docs make clean html

# Auto-rebuild on changes (requires sphinx-autobuild)
uv run --group docs sphinx-autobuild source build/html

# View available make targets
uv run --group docs make help
```

## Dependencies in the `docs` Group

The documentation dependencies are defined in `pyproject.toml`:

```toml
[dependency-groups]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
    "nbsphinx>=0.9.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.7.0",
    "numpy>=1.24.0",
    "pillow>=9.0.0",
]
```
