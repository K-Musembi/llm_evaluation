import pytest
import requests

from lughagen_eval.backends import APIBackend, BackendError, FakeBackend
from lughagen_eval.core_types import GenerationConfig


def test_fake_backend_generate_matches_by_substring():
    backend = FakeBackend(generate_responses={"category:": "travel"})
    output = backend.generate("m", "some prompt\ncategory:", GenerationConfig())
    assert output.ok
    assert output.generated_text == "travel"


def test_fake_backend_generate_falls_back_to_default():
    backend = FakeBackend(default_generate="fallback")
    output = backend.generate("m", "unrelated prompt", GenerationConfig())
    assert output.generated_text == "fallback"


def test_fake_backend_generate_can_simulate_failure():
    backend = FakeBackend(fail_on={"BROKEN"})
    output = backend.generate("m", "this prompt is BROKEN", GenerationConfig())
    assert not output.ok
    assert "BROKEN" in output.error


def test_fake_backend_loglikelihood_returns_scores_per_continuation():
    backend = FakeBackend(loglik_responses={"Answer:": [-0.1, -5.0, -3.0, -9.0]})
    output = backend.loglikelihood("m", "Q: ...\nAnswer:", ["A", "B", "C", "D"], GenerationConfig())
    assert output.choice_logprobs == [-0.1, -5.0, -3.0, -9.0]


def test_fake_backend_records_calls():
    backend = FakeBackend()
    backend.generate("m", "prompt one", GenerationConfig())
    backend.loglikelihood("m", "prompt two", ["A", "B"], GenerationConfig())
    assert backend.calls == [("generate", "prompt one"), ("loglikelihood", "prompt two")]


class _FakeResponse:
    def __init__(self, json_body, status=200):
        self._json = json_body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json


def test_api_backend_generate_success(monkeypatch):
    def fake_post(url, json, timeout):
        assert url == "https://example.com/generate"
        assert json["prompt"] == "hello"
        return _FakeResponse({"generated_text": "world", "latency_ms": 42.0})

    monkeypatch.setattr(requests, "post", fake_post)
    backend = APIBackend(endpoint="https://example.com/generate")
    output = backend.generate("model-a", "hello", GenerationConfig(retries=0))
    assert output.ok
    assert output.generated_text == "world"
    assert output.latency_ms == 42.0


def test_api_backend_generate_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, json, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("boom")
        return _FakeResponse({"generated_text": "ok"})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)  # don't actually wait in tests
    backend = APIBackend(endpoint="https://example.com/generate")
    output = backend.generate("model-a", "hello", GenerationConfig(retries=3))
    assert output.ok
    assert calls["n"] == 3


def test_api_backend_generate_gives_up_after_max_retries(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    backend = APIBackend(endpoint="https://example.com/generate")
    output = backend.generate("model-a", "hello", GenerationConfig(retries=2))
    assert not output.ok
    assert "3 attempts" in output.error  # retries + 1


def test_api_backend_generate_rejects_malformed_response(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse({"unexpected": "shape"})

    monkeypatch.setattr(requests, "post", fake_post)
    backend = APIBackend(endpoint="https://example.com/generate")
    output = backend.generate("model-a", "hello", GenerationConfig(retries=0))
    assert not output.ok


def test_api_backend_loglikelihood_without_score_endpoint_raises():
    backend = APIBackend(endpoint="https://example.com/generate")  # no score_endpoint
    with pytest.raises(BackendError):
        backend.loglikelihood("model-a", "prompt", ["A", "B"], GenerationConfig())


def test_api_backend_loglikelihood_with_score_endpoint(monkeypatch):
    def fake_post(url, json, timeout):
        assert url == "https://example.com/score"
        assert json["continuations"] == ["A", "B"]
        return _FakeResponse({"logprobs": [-1.0, -2.0]})

    monkeypatch.setattr(requests, "post", fake_post)
    backend = APIBackend(endpoint="https://example.com/generate", score_endpoint="https://example.com/score")
    output = backend.loglikelihood("model-a", "prompt", ["A", "B"], GenerationConfig())
    assert output.ok
    assert output.choice_logprobs == [-1.0, -2.0]
