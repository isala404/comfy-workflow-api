"""Core components for webhook functionality."""

from .types import WebhookContext, WebhookEvent, WebhookResult, OutputInfo, WebhookField
from .context import (
    register_webhook_context,
    get_webhook_context,
    unregister_webhook_context,
    get_all_contexts,
    clear_all_contexts,
    cleanup_old_contexts,
    # Legacy aliases
    register_context,
    get_context,
    get_context_by_prompt,
    update_context,
    unregister_context,
)
from .client import get_webhook_client, WebhookClient
from .collector import collect_outputs

__all__ = [
    # Types
    "WebhookContext",
    "WebhookEvent",
    "WebhookResult",
    "OutputInfo",
    "WebhookField",
    # Context functions
    "register_webhook_context",
    "get_webhook_context",
    "unregister_webhook_context",
    "get_all_contexts",
    "clear_all_contexts",
    "cleanup_old_contexts",
    # Legacy aliases
    "register_context",
    "get_context",
    "get_context_by_prompt",
    "update_context",
    "unregister_context",
    # Client
    "get_webhook_client",
    "WebhookClient",
    "collect_outputs",
]
