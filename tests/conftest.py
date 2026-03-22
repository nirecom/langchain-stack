"""Shared pytest configuration."""

import pytest


# pytest-asyncio: auto mode so @pytest.mark.asyncio is not required
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
