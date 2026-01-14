"""Output interception system for webhook delivery and progress events."""

import logging
import traceback
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Optional
import uuid

from .types import WebhookContext, WebhookEvent, OutputInfo
from .context import get_webhook_context, unregister_webhook_context
from .client import get_webhook_client

logger = logging.getLogger(__name__)

_original_task_done = None
_original_progress_hook = None
_interceptor_installed = False
_progress_hook_installed = False
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="webhook_")

# Progress throttling - track last send time per prompt
_progress_last_sent: dict = {}
_PROGRESS_THROTTLE_MS = 500  # Minimum ms between progress updates


def install_output_interceptor() -> bool:
    """Install the output interceptor into ComfyUI's execution system."""
    global _original_task_done, _interceptor_installed
    if _interceptor_installed:
        return True
    try:
        import execution
        _original_task_done = execution.PromptQueue.task_done
        execution.PromptQueue.task_done = _intercepted_task_done
        _interceptor_installed = True
        logger.info("Webhook output interceptor installed")

        # Also install progress hook
        install_progress_hook()

        return True
    except Exception as e:
        logger.error(f"Failed to install output interceptor: {e}")
        return False


def install_progress_hook() -> bool:
    """Install progress hook by wrapping the ProgressBar class."""
    global _progress_hook_installed

    if _progress_hook_installed:
        return True

    try:
        import comfy.utils

        # Wrap the ProgressBar.update_absolute method to intercept progress
        original_update = comfy.utils.ProgressBar.update_absolute

        def wrapped_update(self, value, total=None, preview=None):
            # Call original first
            result = original_update(self, value, total, preview)

            # Then forward to webhook if applicable
            try:
                _handle_progress_update(value, self.total, preview, self.node_id)
            except Exception as e:
                logger.debug(f"Progress webhook error: {e}")

            return result

        comfy.utils.ProgressBar.update_absolute = wrapped_update
        _progress_hook_installed = True

        logger.info("Webhook progress hook installed")
        return True

    except Exception as e:
        logger.error(f"Failed to install progress hook: {e}")
        return False


def _handle_progress_update(value, total, preview_image, node_id):
    """Handle progress update and forward to webhooks."""
    global _progress_last_sent

    # Get the current prompt_id from ComfyUI's server
    try:
        from server import PromptServer
        prompt_id = PromptServer.instance.last_prompt_id
    except Exception:
        return

    if not prompt_id:
        return

    # Look up webhook context for this prompt
    context = get_webhook_context(prompt_id)
    if context is None or not context.callback_url or not context.send_progress:
        return

    # Throttle progress updates
    now_ms = time.time() * 1000
    last_sent = _progress_last_sent.get(prompt_id, 0)

    # Always send first and last updates
    is_first = value <= 1
    is_last = value >= total

    if not is_first and not is_last:
        if now_ms - last_sent < _PROGRESS_THROTTLE_MS:
            return

    _progress_last_sent[prompt_id] = now_ms

    # Send progress event asynchronously
    _executor.submit(
        _send_progress_sync,
        context.callback_url,
        context.request_id,
        prompt_id,
        node_id or "unknown",
        value,
        total,
        context.get_auth_headers() if hasattr(context, 'get_auth_headers') else {}
    )


def _webhook_progress_hook(value, total, preview_image=None, prompt_id=None, node_id=None):
    """Legacy progress hook wrapper - kept for compatibility."""
    global _original_progress_hook, _progress_last_sent

    # Always call the original hook first (ComfyUI's websocket progress)
    if _original_progress_hook is not None:
        try:
            _original_progress_hook(value, total, preview_image, prompt_id, node_id)
        except Exception as e:
            logger.warning(f"Original progress hook error: {e}")

    # Now handle webhook progress
    if prompt_id is None:
        return

    # Look up webhook context for this prompt
    context = get_webhook_context(prompt_id)
    if context is None or not context.callback_url or not context.send_progress:
        return

    # Throttle progress updates
    now_ms = time.time() * 1000
    last_sent = _progress_last_sent.get(prompt_id, 0)

    # Always send first (value=1) and last (value=total) updates
    is_first = value == 1 or value == 0
    is_last = value >= total

    if not is_first and not is_last:
        if now_ms - last_sent < _PROGRESS_THROTTLE_MS:
            return

    _progress_last_sent[prompt_id] = now_ms

    # Send progress event asynchronously
    _executor.submit(
        _send_progress_sync,
        context.callback_url,
        context.request_id,
        prompt_id,
        node_id or "unknown",
        value,
        total,
        context.get_auth_headers() if hasattr(context, 'get_auth_headers') else {}
    )


def _send_progress_sync(callback_url, request_id, prompt_id, node_id, value, total, headers):
    """Send progress event synchronously (for thread pool)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _send_progress_async(callback_url, request_id, prompt_id, node_id, value, total, headers)
        )
    except Exception as e:
        logger.debug(f"Progress send error: {e}")
    finally:
        loop.close()


async def _send_progress_async(callback_url, request_id, prompt_id, node_id, value, total, headers):
    """Send progress event to callback URL."""
    payload = {
        "event": WebhookEvent.PROGRESS.value,
        "request_id": request_id,
        "prompt_id": prompt_id,
        "timestamp": datetime.now().isoformat(),
        "node_id": node_id,
        "progress": {
            "value": value,
            "max": total
        }
    }

    try:
        client = get_webhook_client()
        result = await client.send_json(
            url=callback_url,
            payload=payload,
            headers=headers
        )

        if result.success:
            logger.debug(f"Progress {value}/{total} sent for {prompt_id}")
        else:
            logger.debug(f"Progress send failed: {result.error}")

    except Exception as e:
        logger.debug(f"Progress send exception: {e}")


def uninstall_output_interceptor() -> bool:
    """Uninstall the output interceptor."""
    global _original_task_done, _interceptor_installed
    if not _interceptor_installed:
        return True
    try:
        import execution
        if _original_task_done is not None:
            execution.PromptQueue.task_done = _original_task_done
            _original_task_done = None
        _interceptor_installed = False
        return True
    except Exception as e:
        logger.error(f"Failed to uninstall output interceptor: {e}")
        return False


def _intercepted_task_done(self, item_id: Any, history_result: dict, status: Any, process_item: Optional[Callable] = None):
    """Intercepted task_done to send completion events."""
    global _original_task_done

    prompt_tuple = self.currently_running.get(item_id)
    prompt_id = None
    extra_data = {}

    if prompt_tuple:
        prompt_id = prompt_tuple[1] if len(prompt_tuple) > 1 else None
        extra_data = prompt_tuple[3] if len(prompt_tuple) > 3 else {}

    # Check if this is a webhook request (has webhook_config in extra_data)
    webhook_config = extra_data.get("webhook_config") if isinstance(extra_data, dict) else None

    if webhook_config and prompt_id:
        try:
            outputs = history_result.get("outputs", {})
            meta = history_result.get("meta", {})

            # Submit webhook delivery to thread pool
            _executor.submit(
                _send_webhook_completion_sync,
                webhook_config,
                prompt_id,
                outputs,
                meta,
                status
            )
            logger.debug(f"Webhook delivery queued for prompt {prompt_id}")

        except Exception as e:
            logger.error(f"Failed to queue webhook send: {e}")

    # Call original task_done - MUST happen for ComfyUI to work correctly
    if _original_task_done:
        return _original_task_done(self, item_id, history_result, status, process_item)


def _send_webhook_completion_sync(
    webhook_config: dict,
    prompt_id: str,
    outputs: dict,
    meta: dict,
    status: Any
):
    """Send webhook completion in sync context (for thread pool)."""
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _send_webhook_completion(webhook_config, prompt_id, outputs, meta, status)
        )
    except Exception as e:
        logger.error(f"Webhook delivery failed for prompt {prompt_id}: {e}\n{traceback.format_exc()}")
    finally:
        loop.close()


async def _send_webhook_completion(
    webhook_config: dict,
    prompt_id: str,
    outputs: dict,
    meta: dict,
    status: Any
):
    """Send completion event with outputs to callback URL."""
    from ..processors.outputs import (
        process_outputs_for_webhook,
        load_output_file,
        enrich_output_info
    )

    callback_url = webhook_config.get("callback_url")
    if not callback_url:
        logger.debug(f"No callback_url in webhook_config for prompt {prompt_id}")
        return

    request_id = webhook_config.get("request_id", str(uuid.uuid4()))
    start_time = datetime.now()

    try:
        # Process all outputs from all nodes
        all_outputs = process_outputs_for_webhook(outputs, meta)

        if not all_outputs:
            all_outputs = []

        # Enrich outputs with dimensions, file sizes
        for output in all_outputs:
            try:
                enrich_output_info(output)
            except Exception as e:
                logger.warning(f"Failed to enrich output {output.filename}: {e}")

        # Build files list for multipart
        files = []
        output_metadata = []

        for idx, output in enumerate(all_outputs):
            output_metadata.append(output.to_dict())

            if output.filename:
                try:
                    file_data = load_output_file(output)
                    if file_data:
                        files.append((
                            f"file_{idx}",
                            output.filename,
                            file_data,
                            output.mime_type or "application/octet-stream"
                        ))
                except Exception as e:
                    logger.warning(f"Failed to load output file {output.filename}: {e}")

        # Extract status information
        status_str = "unknown"
        status_messages = []
        if status is not None:
            if hasattr(status, "status_str"):
                status_str = status.status_str
            elif hasattr(status, "_asdict"):
                status_dict = status._asdict()
                status_str = status_dict.get("status_str", "unknown")
                status_messages = status_dict.get("messages", [])

        # Calculate execution time from context if available
        execution_time_ms = None
        context = get_webhook_context(prompt_id)
        if context and context.start_time:
            execution_time_ms = (datetime.now() - context.start_time).total_seconds() * 1000

        # Build metadata payload
        event_type = WebhookEvent.COMPLETED.value if status_str == "success" else WebhookEvent.ERROR.value
        metadata = {
            "event": event_type,
            "request_id": request_id,
            "prompt_id": prompt_id,
            "timestamp": datetime.now().isoformat(),
            "status": status_str,
            "outputs": output_metadata,
            "output_count": len(all_outputs),
            "file_count": len(files),
            "execution_time_ms": execution_time_ms,
            "nodes_executed": len(outputs),
        }

        if status_messages:
            metadata["messages"] = status_messages

        # Get auth headers
        headers = _get_auth_headers(webhook_config)

        # Send the webhook
        client = get_webhook_client()

        if files:
            result = await client.send_multipart(
                url=callback_url,
                metadata=metadata,
                files=files,
                headers=headers
            )
        else:
            result = await client.send_json(
                url=callback_url,
                payload=metadata,
                headers=headers
            )

        if result.success:
            logger.info(
                f"Webhook delivered for prompt {prompt_id}: "
                f"{len(files)} files, {len(output_metadata)} outputs, "
                f"status={result.status}, {result.elapsed_ms:.0f}ms"
            )
        else:
            logger.error(
                f"Webhook failed for prompt {prompt_id}: "
                f"status={result.status}, error={result.error}"
            )

    except Exception as e:
        logger.error(f"Error sending webhook for prompt {prompt_id}: {e}\n{traceback.format_exc()}")

    finally:
        # Clean up context
        try:
            unregister_webhook_context(prompt_id)
        except Exception:
            pass


def _get_auth_headers(webhook_config: dict) -> dict:
    """Build authentication headers from webhook config."""
    headers = {}

    auth_header = webhook_config.get("auth_header")
    auth_value = webhook_config.get("auth_value")

    if auth_header and auth_value:
        headers[auth_header] = auth_value

    custom_headers = webhook_config.get("headers")
    if isinstance(custom_headers, dict):
        headers.update(custom_headers)

    return headers


async def send_progress_event(
    context: WebhookContext,
    node_id: str,
    node_type: str,
    progress: float,
    message: str = ""
):
    """Send a progress event to the callback URL."""
    if not context.send_progress or not context.callback_url:
        return

    elapsed_ms = None
    if context.start_time:
        elapsed_ms = (datetime.now() - context.start_time).total_seconds() * 1000

    payload = {
        "event": WebhookEvent.PROGRESS.value,
        "request_id": context.request_id,
        "prompt_id": context.prompt_id,
        "timestamp": datetime.now().isoformat(),
        "node_id": node_id,
        "node_type": node_type,
        "progress": progress,
        "message": message,
        "elapsed_ms": elapsed_ms,
    }

    try:
        client = get_webhook_client()
        result = await client.send_json(
            url=context.callback_url,
            payload=payload,
            headers=context.get_auth_headers()
        )

        if not result.success:
            logger.warning(f"Progress event failed: {result.error}")

    except Exception as e:
        logger.warning(f"Failed to send progress event: {e}")


async def send_started_event(context: WebhookContext):
    """Send workflow started event."""
    if not context.callback_url:
        return

    payload = {
        "event": WebhookEvent.STARTED.value,
        "request_id": context.request_id,
        "prompt_id": context.prompt_id,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        client = get_webhook_client()
        await client.send_json(
            url=context.callback_url,
            payload=payload,
            headers=context.get_auth_headers()
        )
    except Exception as e:
        logger.warning(f"Failed to send started event: {e}")


async def send_error_event(
    context: WebhookContext,
    error_type: str,
    error_message: str,
    node_id: Optional[str] = None,
    node_type: Optional[str] = None,
    traceback_str: Optional[str] = None
):
    """Send workflow error event."""
    if not context.callback_url:
        return

    elapsed_ms = None
    if context.start_time:
        elapsed_ms = (datetime.now() - context.start_time).total_seconds() * 1000

    payload = {
        "event": WebhookEvent.ERROR.value,
        "request_id": context.request_id,
        "prompt_id": context.prompt_id,
        "timestamp": datetime.now().isoformat(),
        "error": {
            "type": error_type,
            "message": error_message,
            "node_id": node_id,
            "node_type": node_type,
        },
        "elapsed_ms": elapsed_ms,
    }

    if traceback_str:
        payload["error"]["traceback"] = traceback_str[:2000]

    try:
        client = get_webhook_client()
        await client.send_json(
            url=context.callback_url,
            payload=payload,
            headers=context.get_auth_headers()
        )
    except Exception as e:
        logger.warning(f"Failed to send error event: {e}")


def shutdown():
    """Shutdown the webhook executor gracefully."""
    global _executor
    try:
        _executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Webhook executor shut down")
    except Exception as e:
        logger.warning(f"Error shutting down webhook executor: {e}")
