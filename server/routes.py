"""HTTP routes for webhook integration."""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)


def register_routes():
    """Register webhook routes with ComfyUI's PromptServer."""
    try:
        from server import PromptServer

        routes = PromptServer.instance.routes

        @routes.post("/api/webhook")
        async def webhook_submit(request):
            """
            Submit a workflow via multipart/form-data.

            Fields:
            - workflow: JSON workflow definition (required)
            - callback_url: URL to receive events and outputs (required)
            - auth_header: Auth header name (optional)
            - auth_value: Auth header value (optional)
            - timeout: HTTP timeout in seconds (optional, default: 60)
            - max_retries: Max retry attempts (optional, default: 3)
            - <any file>: Uploaded files, accessible by field name
            - <any field>: Text fields, accessible by field name

            Returns:
            - request_id: Unique identifier for this request
            - prompt_id: ComfyUI prompt ID
            - status: "queued"
            """
            from .uploads import save_upload

            request_id = str(uuid.uuid4())

            try:
                # Parse multipart request
                reader = await request.multipart()

                workflow = None
                callback_url = None
                auth_header = None
                auth_value = None
                timeout = 60
                max_retries = 3
                send_progress = True
                progress_interval_ms = 500
                webhook_inputs = {}

                # Get request metadata
                remote_ip = request.remote
                content_type = request.content_type
                content_length = request.content_length
                user_agent = request.headers.get("User-Agent", "unknown")

                async for part in reader:
                    name = part.name

                    if name == "workflow":
                        content = await part.text()
                        try:
                            workflow = json.loads(content)
                        except json.JSONDecodeError as e:
                            return web.json_response(
                                {"error": f"Invalid workflow JSON: {e}"},
                                status=400
                            )

                    elif name == "callback_url":
                        callback_url = await part.text()

                    elif name == "auth_header":
                        auth_header = await part.text()

                    elif name == "auth_value":
                        auth_value = await part.text()

                    elif name == "timeout":
                        try:
                            timeout = int(await part.text())
                        except ValueError:
                            pass

                    elif name == "max_retries":
                        try:
                            max_retries = int(await part.text())
                        except ValueError:
                            pass

                    elif name == "send_progress":
                        value = await part.text()
                        send_progress = value.lower() in ("true", "1", "yes")

                    elif name == "progress_interval_ms":
                        try:
                            progress_interval_ms = int(await part.text())
                        except ValueError:
                            pass

                    elif part.filename:
                        # File upload - save to disk and store path (not raw data to avoid serialization issues)
                        data = await part.read()
                        filename = part.filename
                        file_content_type = part.headers.get("Content-Type", "application/octet-stream")

                        # Save to ComfyUI input directory
                        saved_path = save_upload(data, filename, request_id)

                        # Store file metadata (path, not raw bytes) in webhook_inputs
                        webhook_inputs[name] = {
                            "_is_file": True,
                            "_filename": filename,
                            "_content_type": file_content_type,
                            "_saved_path": saved_path,
                            "_size_bytes": len(data),
                        }
                        logger.debug(f"Saved upload '{name}' ({len(data)} bytes) to {saved_path}")

                    else:
                        # Text field
                        webhook_inputs[name] = await part.text()

                # Validate required fields
                if workflow is None:
                    return web.json_response(
                        {"error": "Missing required field: workflow"},
                        status=400
                    )

                if callback_url is None:
                    return web.json_response(
                        {"error": "Missing required field: callback_url"},
                        status=400
                    )

                # Generate prompt_id
                prompt_id = str(uuid.uuid4())

                # Build webhook_config for extra_data
                webhook_config = {
                    "callback_url": callback_url,
                    "auth_header": auth_header,
                    "auth_value": auth_value,
                    "request_id": request_id,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "send_progress": send_progress,
                    "progress_interval_ms": progress_interval_ms,
                    "remote_ip": remote_ip,
                    "content_type": content_type,
                    "content_length": content_length,
                    "user_agent": user_agent,
                }

                # Build extra_data for workflow
                # ComfyUI passes extra_data['extra_pnginfo'] to nodes with EXTRA_PNGINFO hidden input
                extra_data = {
                    "webhook_config": webhook_config,
                    "webhook_inputs": webhook_inputs,
                    "extra_pnginfo": {
                        "webhook_config": webhook_config,
                        "webhook_inputs": webhook_inputs,
                    },
                }

                # Validate prompt and get outputs_to_execute
                import execution
                valid = await execution.validate_prompt(prompt_id, workflow, None)

                if not valid[0]:
                    # Validation failed
                    logger.warning(f"Invalid workflow: {valid[1]}")
                    return web.json_response(
                        {"error": valid[1], "node_errors": valid[3]},
                        status=400
                    )

                outputs_to_execute = valid[2]

                # Register webhook context for progress tracking
                from ..core.context import register_webhook_context
                from ..core.types import WebhookContext

                context = WebhookContext(
                    request_id=request_id,
                    prompt_id=prompt_id,
                    callback_url=callback_url,
                    auth_header=auth_header,
                    auth_value=auth_value,
                    timeout=timeout,
                    max_retries=max_retries,
                    send_progress=send_progress,
                    progress_interval_ms=progress_interval_ms,
                )
                register_webhook_context(context)

                # Queue the workflow
                queue = PromptServer.instance.prompt_queue

                # ComfyUI expects: (priority, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
                queue.put((0, prompt_id, workflow, extra_data, outputs_to_execute, {}))

                logger.info(f"Webhook request {request_id} queued as prompt {prompt_id}")

                return web.json_response({
                    "request_id": request_id,
                    "prompt_id": prompt_id,
                    "status": "queued",
                })

            except Exception as e:
                logger.error(f"Webhook submit error: {e}", exc_info=True)
                return web.json_response(
                    {"error": str(e)},
                    status=500
                )

        @routes.get("/api/webhook/status/{request_id}")
        async def webhook_status(request):
            """
            Get status of a webhook request.

            Returns:
            - request_id: The request ID
            - prompt_id: ComfyUI prompt ID
            - status: queued|running|completed|error|interrupted
            - progress: Current progress info (if running)
            - error: Error details (if error)
            """
            from ..core.context import get_context

            request_id = request.match_info["request_id"]
            context = get_context(request_id)

            if context is None:
                return web.json_response(
                    {"error": "Request not found"},
                    status=404
                )

            response = {
                "request_id": context.request_id,
                "prompt_id": context.prompt_id,
                "callback_url": context.callback_url,
                "fields": context.get_field_names(),
            }

            if context.start_time:
                response["started_at"] = context.start_time.isoformat()

            if context.completed_at:
                response["completed_at"] = context.completed_at.isoformat()
                if context.start_time:
                    elapsed = (context.completed_at - context.start_time).total_seconds() * 1000
                    response["execution_time_ms"] = elapsed

            return web.json_response(response)

        @routes.delete("/api/webhook/{request_id}")
        async def webhook_cancel(request):
            """Cancel a queued or running webhook request."""
            from ..core.context import get_context, update_context

            request_id = request.match_info["request_id"]
            context = get_context(request_id)

            if context is None:
                return web.json_response(
                    {"error": "Request not found"},
                    status=404
                )

            # Try to interrupt the execution
            try:
                if context.prompt_id:
                    PromptServer.instance.prompt_queue.delete_queue_item(context.prompt_id)

                update_context(request_id, completed_at=datetime.now())

                return web.json_response({
                    "request_id": request_id,
                    "status": "cancelled",
                })
            except Exception as e:
                logger.error(f"Failed to cancel request {request_id}: {e}")
                return web.json_response(
                    {"error": str(e)},
                    status=500
                )

        logger.info("Webhook routes registered: POST /api/webhook, GET /api/webhook/status/{id}")
        return True

    except ImportError as e:
        logger.warning(f"Could not register webhook routes (ComfyUI server not available): {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to register webhook routes: {e}", exc_info=True)
        return False
