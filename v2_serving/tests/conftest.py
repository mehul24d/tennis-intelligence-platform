"""conftest.py — shared fixtures. `job_store`/`render_job_store` are
module-level singletons (see each module's own docstring on why, for a
single-process dev server) -- cleared before each test so tests don't leak
state into one another."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from v2_serving.job_store import job_store
from v2_serving.main import app
from v2_serving.render_job_store import render_job_store


@pytest.fixture(autouse=True)
def _clear_job_store():
    job_store._jobs.clear()
    render_job_store._jobs.clear()
    yield
    job_store._jobs.clear()
    render_job_store._jobs.clear()


@pytest.fixture
def client():
    return TestClient(app)
