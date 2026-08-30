from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
import requests

from knowledge.api.utils import serve_utils
from knowledge.api.utils.serve_utils import ToolServiceError, post_once
from knowledge.api.vision import sam3
from workflow.research import libero_backend
from workflow.research.util import add_loopback_no_proxy


def test_loopback_no_proxy_is_merged_for_both_environment_keys() -> None:
    environment = {
        "HTTP_PROXY": "http://127.0.0.1:7898",
        "NO_PROXY": "internal.example,localhost",
        "no_proxy": "other.example",
    }

    add_loopback_no_proxy(environment)

    assert environment["NO_PROXY"] == "127.0.0.1,localhost,::1,internal.example"
    assert environment["no_proxy"] == "127.0.0.1,localhost,::1,other.example"
    with patch.dict(os.environ, environment, clear=True):
        assert requests.utils.get_environ_proxies("http://127.0.0.1:14014") == {}


def test_loopback_post_does_not_trust_proxy_environment(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, bool]:
            return {"ok": True}

    class FakeSession:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, url, *, json, timeout):
            observed.update(
                url=url,
                payload=json,
                timeout=timeout,
                trust_env=self.trust_env,
            )
            return FakeResponse()

    monkeypatch.setattr(serve_utils.requests, "Session", FakeSession)

    result = serve_utils.post_with_retries(
        "http://127.0.0.1:14014/segment",
        {"prompt": "pan"},
        max_retries=1,
    )

    assert result == {"ok": True}
    assert observed["trust_env"] is False


def test_post_once_normalizes_transport_errors(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise requests.ConnectionError("service unavailable")

    monkeypatch.setattr(serve_utils, "_post", fail)

    with pytest.raises(ToolServiceError, match="service unavailable"):
        post_once("http://127.0.0.1:8115/plan", {})


def test_sam3_communication_failure_is_not_converted_to_empty_masks(
    monkeypatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise ToolServiceError("proxy returned 502")

    monkeypatch.setattr(sam3, "post_with_retries", fail)
    segment = sam3.init_sam3()

    with pytest.raises(ToolServiceError, match="proxy returned 502"):
        segment(np.zeros((8, 8, 3), dtype=np.uint8), "frying pan")


def test_libero_backend_propagates_tool_outage_as_infrastructure(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise ToolServiceError("service unavailable")

    monkeypatch.setattr(libero_backend, "execute_program", fail)

    with pytest.raises(ToolServiceError, match="service unavailable"):
        libero_backend._execute_repair_program(
            "pass",
            functions={},
            observation={},
        )


def test_libero_backend_still_captures_policy_errors(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("bad policy")

    monkeypatch.setattr(libero_backend, "execute_program", fail)
    error = libero_backend._execute_repair_program(
        "pass",
        functions={},
        observation={},
    )

    assert error is not None
    assert "RuntimeError: bad policy" in error
