"""WebhookReceiver node - Entry point for webhook-triggered workflows."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict

from ..core.types import WebhookContext, WebhookField
from ..core.context import register_webhook_context
from ..core.interceptor import send_started_event
from ..utils.helpers import format_size

logger = logging.getLogger(__name__)


class WebhookReceiver:
    """
    Entry point for webhook-triggered workflows.

    Extracts webhook configuration from extra_pnginfo (passed from /api/webhook)
    and creates a WebhookContext for use by downstream nodes.

    Configuration comes from webhook_config in extra_pnginfo:
    - callback_url: Where to send results
    - auth_header: Auth header name (e.g., "Authorization")
    - auth_value: Auth header value (e.g., "Bearer token123")
    - request_id: Unique request identifier
    - timeout: HTTP timeout
    - max_retries: Retry attempts

    Fields/uploads come from webhook_inputs in extra_pnginfo.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "default_callback_url": ("STRING", {
                    "default": "",
                    "tooltip": "Fallback callback URL if not provided in request"
                }),
                "default_timeout": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 300,
                    "tooltip": "HTTP request timeout in seconds"
                }),
                "default_max_retries": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Number of retry attempts on failure"
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable debug logging"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("WEBHOOK_CONTEXT",)
    RETURN_NAMES = ("webhook_context",)
    FUNCTION = "receive"
    CATEGORY = "webhook"
    DESCRIPTION = "Entry point for webhook-triggered workflows. Extracts webhook config and creates context for downstream nodes."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always return a unique value to prevent caching."""
        # Webhook requests should never be cached as each request has unique data
        import uuid
        return str(uuid.uuid4())

    def receive(
        self,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        default_callback_url="",
        default_timeout=60,
        default_max_retries=3,
        debug=False,
    ):
        """
        Extract webhook configuration and create context.

        The context is registered for output interception and passed
        to WebhookTransformer nodes for field extraction.
        """
        # Extract webhook_config and webhook_inputs from extra_data
        extra_data = extra_pnginfo or {}
        webhook_config = extra_data.get("webhook_config", {})
        webhook_inputs = extra_data.get("webhook_inputs", {})

        # Extract configuration from webhook_config
        callback_url = webhook_config.get("callback_url") or default_callback_url
        auth_header = webhook_config.get("auth_header")
        auth_value = webhook_config.get("auth_value")
        request_id = webhook_config.get("request_id") or str(uuid.uuid4())
        timeout = webhook_config.get("timeout") or default_timeout
        max_retries = webhook_config.get("max_retries") or default_max_retries

        # Extract request metadata
        remote_ip = webhook_config.get("remote_ip", "unknown")
        content_type = webhook_config.get("content_type", "multipart/form-data")
        content_length = webhook_config.get("content_length")
        user_agent = webhook_config.get("user_agent", "unknown")

        # Build fields dictionary from webhook_inputs
        fields: Dict[str, WebhookField] = {}
        for name, value in webhook_inputs.items():
            # Skip reserved fields that start with _
            if name.startswith("_"):
                continue

            # Detect if this is a file upload (has special structure)
            is_file = False
            filename = None
            content_type_field = None
            size_bytes = None

            if isinstance(value, dict) and value.get("_is_file"):
                # This is a file upload - load from saved path
                is_file = True
                filename = value.get("_filename")
                content_type_field = value.get("_content_type")
                size_bytes = value.get("_size_bytes", 0)
                saved_path = value.get("_saved_path")

                # Load file data from disk
                if saved_path:
                    try:
                        from pathlib import Path
                        value = Path(saved_path).read_bytes()
                    except Exception as e:
                        logger.warning(f"Failed to load file from {saved_path}: {e}")
                        value = b""
                else:
                    value = b""
            elif isinstance(value, bytes):
                # Raw bytes = file
                is_file = True
                size_bytes = len(value)
            elif isinstance(value, str):
                size_bytes = len(value.encode("utf-8"))

            fields[name] = WebhookField(
                name=name,
                value=value,
                is_file=is_file,
                filename=filename,
                content_type=content_type_field,
                size_bytes=size_bytes,
            )

        # Create context
        context = WebhookContext(
            callback_url=callback_url,
            request_id=request_id,
            auth_header=auth_header,
            auth_value=auth_value,
            timeout=timeout,
            max_retries=max_retries,
            fields=fields,
            remote_ip=remote_ip,
            content_type=content_type,
            content_length=content_length,
            user_agent=user_agent,
            prompt_id=unique_id,
            inputs=webhook_inputs,  # Raw inputs for fallback
            send_progress=webhook_config.get("send_progress", True),
            progress_interval_ms=webhook_config.get("progress_interval_ms", 500),
            include_workflow_in_response=webhook_config.get("include_workflow_in_response", False),
        )

        # Debug output
        if debug:
            self._print_debug(context, fields)

        # Register context for output interception
        if context.prompt_id:
            register_webhook_context(context)
            if not debug:
                logger.info(
                    f"WebhookReceiver: Registered context for prompt {context.prompt_id}, "
                    f"callback: {callback_url or '(none)'}, "
                    f"fields: {list(fields.keys())}"
                )

            # Send started event if callback URL is configured
            if context.callback_url:
                try:
                    # Try to get running loop (Python 3.10+ safe)
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(send_started_event(context))
                except RuntimeError:
                    # No running loop - run synchronously
                    try:
                        asyncio.run(send_started_event(context))
                    except RuntimeError:
                        # Fallback for nested event loops
                        from ..utils.helpers import run_async
                        run_async(send_started_event(context))
                except Exception as e:
                    logger.warning(f"Failed to send started event: {e}")

        return (context,)

    def _print_debug(self, context: WebhookContext, fields: Dict[str, WebhookField]):
        """Print debug information about the incoming request."""
        timestamp = context.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        lines = [
            "",
            "[Webhook Receiver] Incoming Request",
            "=" * 50,
            f"  Remote IP:      {context.remote_ip or 'unknown'}",
            f"  Timestamp:      {timestamp}",
            f"  Request ID:     {context.request_id}",
            f"  Content-Type:   {context.content_type or 'unknown'}",
            f"  Content-Length: {format_size(context.content_length)}",
            f"  User-Agent:     {context.user_agent or 'unknown'}",
            "",
            "  Configuration:",
            f"    Callback URL:   {context.callback_url or '(none)'}",
            f"    Auth Header:    {context.censor_auth()}",
            f"    Timeout:        {context.timeout}s",
            f"    Max Retries:    {context.max_retries}",
            "",
            f"  Fields Received ({len(fields)}):",
        ]

        if fields:
            field_names = sorted(fields.keys())
            for i, name in enumerate(field_names):
                field = fields[name]
                prefix = "    +-" if i == len(field_names) - 1 else "    |-"

                if field.is_file:
                    size_str = format_size(field.size_bytes)
                    ct = field.content_type or "application/octet-stream"
                    lines.append(f"{prefix} {name:20} (file) {ct}, {size_str}")
                else:
                    size_str = format_size(field.size_bytes)
                    lines.append(f"{prefix} {name:20} (text) {size_str}")
        else:
            lines.append("    (no fields)")

        lines.append("=" * 50)

        print("\n".join(lines))
