import os

import pytest

from rag_engine.generate import (
    GeminiClient, MissingAPIKeyError, RetrievedContext, build_prompt,
)


@pytest.fixture(autouse=True)
def _clear_gemini_env(monkeypatch):
    """Every test in this module should see a clean slate regardless of what's
    actually set in the real environment (e.g. a developer's own GEMINI_API_KEY)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_build_prompt_includes_retrieved_context_and_question():
    ctx = [
        RetrievedContext(
            doc_id="match:1", text="Sinner defeated Ruud 6-0, 6-2.",
            metadata={"doc_type": "match_summary"},
        ),
    ]
    prompt = build_prompt("How did Sinner perform against Ruud?", ctx)
    assert "Sinner defeated Ruud 6-0, 6-2." in prompt
    assert "How did Sinner perform against Ruud?" in prompt
    assert "ONLY" in prompt  # grounding instruction present


def test_build_prompt_handles_empty_context():
    prompt = build_prompt("Some question with no matches", [])
    assert "no retrieved context available" in prompt
    assert "Some question with no matches" in prompt


def test_missing_api_key_raises_immediately_at_construction():
    """Fails at GeminiClient() construction, not on first generate() call --
    retrieval/index code that never imports this module is unaffected either way,
    but a caller that DOES construct a client should learn about a missing key
    immediately, not deep inside a generate() call."""
    with pytest.raises(MissingAPIKeyError):
        GeminiClient()


def test_gemini_api_key_preferred_over_google_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-value")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key-value")
    client = GeminiClient()
    assert client._api_key == "gemini-key-value"


def test_google_api_key_used_as_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key-value")
    client = GeminiClient()
    assert client._api_key == "google-key-value"


def test_client_construction_succeeds_with_a_key(monkeypatch):
    """Construction itself (client setup, no network call) should succeed with any
    syntactically-present key -- this test doesn't call the real API."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-construction-only")
    client = GeminiClient()
    assert client.model == "gemini-3.5-flash"
    assert hasattr(client._client.models, "generate_content")


def test_generate_retries_on_503_then_succeeds(monkeypatch):
    """A transient 503 UNAVAILABLE (Google's own "high demand" response, observed
    directly against a real key) should be retried, not surfaced immediately -- this
    test mocks generate_content to fail twice with 503 then succeed, and asserts the
    retry logic actually invokes the underlying call multiple times rather than just
    trusting the decorator is wired up correctly."""
    from google.genai.errors import ServerError

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-retry-test")
    client = GeminiClient()

    calls = {"count": 0}

    class _FakeResponse:
        text = "OK after retries"

    def _flaky_generate_content(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ServerError(
                503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}},
            )
        return _FakeResponse()

    monkeypatch.setattr(client._client.models, "generate_content", _flaky_generate_content)
    # Skip real sleeping between retries -- this test only cares that retry happens.
    client.generate.retry.wait = lambda *a, **k: 0

    result = client.generate("some prompt")
    assert result == "OK after retries"
    assert calls["count"] == 3


def test_generate_does_not_retry_on_non_503_errors(monkeypatch):
    """A 429 (quota exhausted) or 401 (bad key) should propagate immediately -- these
    aren't transient, so retrying just delays a failure the caller should see right
    away."""
    from google.genai.errors import ClientError

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-retry-test")
    client = GeminiClient()

    calls = {"count": 0}

    def _quota_exhausted(*args, **kwargs):
        calls["count"] += 1
        raise ClientError(
            429, {"error": {"code": 429, "message": "quota exhausted", "status": "RESOURCE_EXHAUSTED"}},
        )

    monkeypatch.setattr(client._client.models, "generate_content", _quota_exhausted)

    with pytest.raises(ClientError):
        client.generate("some prompt")
    assert calls["count"] == 1
