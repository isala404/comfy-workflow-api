"""Core functionality tests."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import WebhookContext, WebhookField, WebhookEvent, WebhookResult, OutputInfo
from core.context import register_webhook_context, get_webhook_context, unregister_webhook_context


def test_context_defaults():
    ctx = WebhookContext()
    assert ctx.callback_url == ""
    assert ctx.timeout == 60
    assert ctx.max_retries == 3
    assert ctx.request_id
    # start_time is None until register_webhook_context() is called
    assert ctx.start_time is None
    assert ctx.timestamp is not None


def test_context_with_values():
    ctx = WebhookContext(
        callback_url="https://example.com/webhook",
        auth_header="Authorization",
        auth_value="Bearer token123",
        timeout=30,
    )
    assert ctx.callback_url == "https://example.com/webhook"
    assert ctx.auth_header == "Authorization"
    assert ctx.timeout == 30


def test_auth_headers():
    ctx = WebhookContext(auth_header="X-API-Key", auth_value="secret")
    assert ctx.get_auth_headers() == {"X-API-Key": "secret"}

    ctx_no_auth = WebhookContext()
    assert ctx_no_auth.get_auth_headers() == {}


def test_field_access():
    field = WebhookField(name="prompt", value="hello world", size_bytes=11)
    ctx = WebhookContext(fields={"prompt": field})

    assert ctx.has_field("prompt")
    assert not ctx.has_field("missing")
    assert ctx.get_field("prompt") == field
    assert ctx.get_field_value("prompt") == "hello world"
    assert ctx.get_field_value("missing", "default") == "default"


def test_legacy_inputs():
    ctx = WebhookContext(inputs={"old_field": "old_value"})
    assert ctx.get_field_value("old_field") == "old_value"
    assert ctx.has_field("old_field")


def test_censor_auth():
    ctx = WebhookContext(auth_header="Authorization", auth_value="Bearer secrettoken123")
    censored = ctx.censor_auth()
    assert "Bear" in censored
    assert "123" in censored
    assert "secrettoken" not in censored


def test_context_registry():
    ctx = WebhookContext(prompt_id="test-123", callback_url="https://example.com")
    register_webhook_context(ctx)
    assert get_webhook_context("test-123") is ctx
    unregister_webhook_context("test-123")
    assert get_webhook_context("test-123") is None


def test_output_info():
    info = OutputInfo(type="image", filename="test.png", width=1024, height=768)
    d = info.to_dict()
    assert d["type"] == "image"
    assert d["filename"] == "test.png"
    assert d["width"] == 1024


def test_webhook_events():
    assert WebhookEvent.STARTED.value == "workflow.started"
    assert WebhookEvent.COMPLETED.value == "workflow.completed"
    assert WebhookEvent.ERROR.value == "workflow.error"


def run_all():
    tests = [
        test_context_defaults,
        test_context_with_values,
        test_auth_headers,
        test_field_access,
        test_legacy_inputs,
        test_censor_auth,
        test_context_registry,
        test_output_info,
        test_webhook_events,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} - {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("Running core tests...")
    success = run_all()
    sys.exit(0 if success else 1)
