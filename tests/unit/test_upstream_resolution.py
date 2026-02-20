from __future__ import annotations

import pytest

from experiments.adapter_prompt_v1.upstream_resolution import (
    build_upstream_base_url,
    resolve_upstream_host,
)


def test_build_upstream_base_url() -> None:
    assert build_upstream_base_url(host="localhost", port=8101) == "http://localhost:8101/v1"


def test_resolve_upstream_host_uses_environment_override() -> None:
    resolved = resolve_upstream_host(env={"UPSTREAM_HOST": "host.docker.internal"})
    assert resolved == "host.docker.internal"


def test_resolve_upstream_host_defaults_to_localhost() -> None:
    resolved = resolve_upstream_host(env={})
    assert resolved == "localhost"


def test_resolve_upstream_host_falls_back_when_environment_override_is_blank() -> None:
    resolved = resolve_upstream_host(env={"UPSTREAM_HOST": "   "})
    assert resolved == "localhost"


def test_resolve_upstream_host_rejects_scheme() -> None:
    with pytest.raises(ValueError, match="bare host without scheme"):
        resolve_upstream_host(env={"UPSTREAM_HOST": "http://localhost"})


def test_resolve_upstream_host_rejects_port() -> None:
    with pytest.raises(ValueError, match="must not include a port"):
        resolve_upstream_host(env={"UPSTREAM_HOST": "localhost:8101"})


def test_resolve_upstream_host_rejects_path() -> None:
    with pytest.raises(ValueError, match="without path, query, or fragment"):
        resolve_upstream_host(env={"UPSTREAM_HOST": "localhost/v1"})
