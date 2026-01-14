"""
ComfyUI Workflow API - Server extension for workflow integration.

Provides:
1. HTTP endpoint: POST /api/webhook for submitting workflows
2. WebhookReceiver node - Entry point for API-triggered workflows
3. WebhookTransformer node - Extract individual fields by name
4. WebhookSend node - Send outputs back to callback URL
5. Real-time progress updates via callback URL
"""

__version__ = "1.0.0"

import logging

logger = logging.getLogger(__name__)

# Import nodes
from .nodes.receiver import WebhookReceiver
from .nodes.transformer import WebhookTransformer
from .nodes.send import WebhookSend

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WebhookReceiver": WebhookReceiver,
    "WebhookTransformer": WebhookTransformer,
    "WebhookSend": WebhookSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebhookReceiver": "Workflow API Receiver",
    "WebhookTransformer": "Workflow API Transformer",
    "WebhookSend": "Workflow API Send",
}

# Initialize server extension
_initialized = False


def _initialize():
    """Initialize server extension and interceptor hooks."""
    global _initialized

    if _initialized:
        return

    try:
        # Register HTTP routes
        from .server.routes import register_routes
        if register_routes():
            logger.info(f"[comfy-workflow-api] v{__version__} - Routes registered")

        # Install output interceptor for automatic webhook delivery
        from .core.interceptor import install_output_interceptor
        if install_output_interceptor():
            logger.info(f"[comfy-workflow-api] v{__version__} - Output interceptor installed")

        _initialized = True
        logger.info(f"[comfy-workflow-api] v{__version__} loaded successfully")

    except Exception as e:
        logger.error(f"[comfy-workflow-api] Failed to initialize: {e}", exc_info=True)


# Initialize on import
_initialize()


# Cleanup function
def _cleanup():
    """Cleanup on unload."""
    try:
        from .core.interceptor import shutdown, uninstall_output_interceptor
        uninstall_output_interceptor()
        shutdown()
    except Exception as e:
        logger.warning(f"[comfy-workflow-api] Cleanup error: {e}")
