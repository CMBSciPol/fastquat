from collections.abc import Iterator

import jax
import pytest


@pytest.fixture
def enable_x64() -> Iterator[None]:
    try:
        jax.config.update('jax_enable_x64', True)
        yield
    finally:
        jax.config.update('jax_enable_x64', False)
