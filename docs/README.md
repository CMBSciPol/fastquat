# Documentation Build Instructions

This documentation uses **PEP 735 dependency groups** instead of separate requirements files to avoid duplication.


## Building Documentation Locally

The dependency group `docs` is included in the group `dev` which is installed by default when using  ̀uv` (recommended).
No extra step other than setting up and activating the virtual environment is required:
```bash
# From the docs/ directory
uv sync --locked --group cuda12
source .venv/bin/activate
```

If you're not using uv, you can install the docs dependencies manually:
```bash
# Install the project with docs dependencies (pip >= 25.1)
pip install --upgrade pip
pip install -e . --group dev
```

Then build the docs
```bash
cd docs
make html
```
