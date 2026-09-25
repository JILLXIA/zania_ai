"""Optional LangSmith tracing; no client or uploads when tracing is disabled."""

import sys
from contextlib import contextmanager

from langsmith import Client, get_tracing_context, trace, tracing_context

from app.config import Settings
from app.logging import event


def create_trace_client(settings: Settings) -> Client | None:
    if not settings.langsmith_tracing:
        return None
    if not settings.langsmith_api_key.get_secret_value():
        event("tracing", code="missing_langsmith_api_key")
        return None
    try:
        return Client(
            api_key=settings.langsmith_api_key.get_secret_value(),
            api_url=settings.langsmith_endpoint,
            workspace_id=settings.langsmith_workspace_id or None,
            hide_inputs=settings.langsmith_hide_inputs,
            hide_outputs=settings.langsmith_hide_outputs,
            timeout_ms=5000,
            tracing_error_callback=lambda _error: event("tracing", code="upload_failed"),
        )
    except Exception:
        event("tracing", code="client_setup_failed")
        return None


@contextmanager
def trace_step(name: str, **kwargs):
    # Do not create even a local RunTree/default client for ordinary untraced requests.
    context = get_tracing_context()
    if context.get("enabled") is not True:
        yield None
        return
    # SDK upload errors must not replace an application result or exception.
    # The outer context also restores the parent if SDK teardown fails midway.
    with tracing_context(**context):
        manager = trace(name, **kwargs)
        try:
            run = manager.__enter__()
        except Exception:
            event("tracing", code="span_start_failed")
            yield None
            return
        try:
            yield run
        finally:
            try:
                manager.__exit__(*sys.exc_info())
            except Exception:
                event("tracing", code="span_finish_failed")
