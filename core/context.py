"""
Webhook context registry for managing active webhook requests.
Provides thread-safe storage keyed by prompt_id.
"""

import threading
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from .types import WebhookContext

logger = logging.getLogger(__name__)

# Thread-safe registry keyed by prompt_id
_lock = threading.RLock()
_contexts: Dict[str, WebhookContext] = {}


def register_webhook_context(context: WebhookContext) -> None:
    """
    Register a webhook context for tracking.

    Args:
        context: The WebhookContext to register
    """
    with _lock:
        if context.prompt_id:
            _contexts[context.prompt_id] = context
            context.start_time = datetime.now()
            logger.debug(
                f"Registered webhook context: prompt={context.prompt_id}, "
                f"request={context.request_id}"
            )
        else:
            logger.warning("Cannot register context without prompt_id")


def get_webhook_context(prompt_id: str) -> Optional[WebhookContext]:
    """
    Get a context by prompt ID.

    Args:
        prompt_id: The ComfyUI prompt ID

    Returns:
        The WebhookContext if found, None otherwise
    """
    with _lock:
        return _contexts.get(prompt_id)


def unregister_webhook_context(prompt_id: str) -> Optional[WebhookContext]:
    """
    Remove and return a context from the registry.

    Args:
        prompt_id: The prompt ID to unregister

    Returns:
        The removed WebhookContext if found, None otherwise
    """
    with _lock:
        context = _contexts.pop(prompt_id, None)
        if context:
            logger.debug(f"Unregistered webhook context: {prompt_id}")
        return context


def get_all_contexts() -> Dict[str, WebhookContext]:
    """
    Get a copy of all registered contexts.
    Useful for debugging and monitoring.

    Returns:
        Dictionary mapping prompt_id to WebhookContext
    """
    with _lock:
        return dict(_contexts)


def clear_all_contexts() -> int:
    """
    Clear all registered contexts.
    Useful for testing and cleanup.

    Returns:
        Number of contexts that were cleared
    """
    with _lock:
        count = len(_contexts)
        _contexts.clear()
        logger.debug(f"Cleared {count} webhook contexts")
        return count


def cleanup_old_contexts(max_age_hours: int = 24) -> int:
    """
    Remove old completed contexts.

    Args:
        max_age_hours: Remove contexts older than this

    Returns:
        Number of contexts removed
    """
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    count = 0

    with _lock:
        to_remove = []

        for prompt_id, context in _contexts.items():
            # Clean up old contexts based on completed_at or start_time
            if context.completed_at and context.completed_at < cutoff:
                to_remove.append(prompt_id)
            elif context.start_time and context.start_time < cutoff:
                to_remove.append(prompt_id)

        for prompt_id in to_remove:
            _contexts.pop(prompt_id, None)
            count += 1

    if count > 0:
        logger.info(f"Cleaned up {count} old webhook contexts")

    return count


# Legacy function aliases for compatibility
def register_context(context: WebhookContext) -> None:
    """Legacy alias for register_webhook_context."""
    register_webhook_context(context)


def get_context(request_id: str) -> Optional[WebhookContext]:
    """
    Get a context by request ID (searches all contexts).

    Args:
        request_id: The request ID to look up

    Returns:
        The WebhookContext if found, None otherwise
    """
    with _lock:
        for context in _contexts.values():
            if context.request_id == request_id:
                return context
        return None


def get_context_by_prompt(prompt_id: str) -> Optional[WebhookContext]:
    """Legacy alias for get_webhook_context."""
    return get_webhook_context(prompt_id)


def update_context(request_id: str, **kwargs) -> bool:
    """
    Update fields on a context by request ID.

    Args:
        request_id: The request ID to update
        **kwargs: Fields to update

    Returns:
        True if context was found and updated
    """
    with _lock:
        for context in _contexts.values():
            if context.request_id == request_id:
                for key, value in kwargs.items():
                    if hasattr(context, key):
                        setattr(context, key, value)
                return True
        return False


def unregister_context(request_id: str) -> Optional[WebhookContext]:
    """
    Remove a context by request ID.

    Args:
        request_id: The request ID to unregister

    Returns:
        The removed WebhookContext if found, None otherwise
    """
    with _lock:
        to_remove = None
        for prompt_id, context in _contexts.items():
            if context.request_id == request_id:
                to_remove = prompt_id
                break

        if to_remove:
            return _contexts.pop(to_remove, None)
        return None
