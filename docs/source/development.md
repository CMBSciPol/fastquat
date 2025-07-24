# Development Guide

This guide covers how to contribute to FastQuat development.

## Setting up Development Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/CMBSciPol/fastquat.git
   cd fastquat
   ```

2. Install in development mode:

   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks (optional but recommended):

   ```bash
   pre-commit install
   ```

## Running Tests

FastQuat uses pytest for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fastquat

# Run specific test file
pytest tests/test_rotation.py

# Run tests with JIT compilation
pytest -k "jit"
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and linting:

```bash
# Format code
ruff format

# Check for linting issues
ruff check

# Fix linting issues automatically
ruff check --fix
```

## Project Structure

```
fastquat/
├── src/fastquat/           # Main package
│   ├── __init__.py
│   └── quaternion.py       # Core Quaternion class
├── tests/                  # Test suite
│   ├── test_base.py        # Basic operations
│   ├── test_math.py        # Mathematical operations
│   ├── test_rotation.py    # Rotation and SLERP tests
│   └── test_tensor.py      # Tensor operations
├── docs/                   # Documentation
├── benchmarks/             # Performance benchmarks
└── pyproject.toml          # Project configuration
```

## Adding New Features

When adding new features:

1. **Write tests first**: Add tests in the appropriate test file
2. **Follow existing patterns**: Look at existing code for style and structure
3. **Add documentation**: Update docstrings and add examples
4. **Consider JAX compatibility**: Ensure your code works with JIT, vmap, and grad
5. **Benchmark performance**: Add benchmarks for performance-critical code

## Contributing Guidelines

1. **Fork the repository** and create a feature branch
2. **Write comprehensive tests** for new functionality
3. **Update documentation** including docstrings and examples
4. **Run the test suite** to ensure nothing is broken
5. **Submit a pull request** with a clear description

## Performance Considerations

FastQuat is designed for high performance. When contributing:

* **Avoid Python loops**: Use JAX operations that can be vectorized
* **Consider memory layout**: Operations should work efficiently with batched data
* **Profile your code**: Use JAX profiling tools to identify bottlenecks
* **Test with JIT**: Ensure your code compiles and runs efficiently under JIT

## Example: Adding a New Method

Here's how to add a new method to the Quaternion class:

```python
def new_method(self, parameter: Array) -> Quaternion:
    """Brief description of what the method does.

    Args:
        parameter: Description of the parameter

    Returns:
        Description of the return value
    """
    # Implementation using JAX operations
    result = jnp.some_operation(self.wxyz, parameter)
    return Quaternion.from_array(result)
```

Then add tests:

```python
@pytest.mark.parametrize('do_jit', [False, True])
def test_new_method(do_jit):
    """Test the new method."""
    def test_fn(q, param):
        return q.new_method(param)

    if do_jit:
        test_fn = jax.jit(test_fn)

    # Test implementation
    q = Quaternion.ones()
    result = test_fn(q, parameter)
    assert jnp.allclose(result.wxyz, expected_result)
```

## Documentation

Documentation is built using Sphinx with MyST parser for Markdown support. To build locally:

```bash
cd docs
make html
```

The documentation will be available in `docs/build/html/`.

## Release Process

1. Update version number in `pyproject.toml`
2. Update `CHANGELOG.md` with new features and fixes
3. Create a git tag: `git tag v0.2.0`
4. Push to GitHub: `git push --tags`
5. Create a GitHub release
6. The CI will automatically build and upload to PyPI

## Getting Help

* **GitHub Issues**: Report bugs or request features
* **Discussions**: Ask questions or discuss ideas
* **Email**: Contact the maintainers directly

We welcome contributions of all kinds!
