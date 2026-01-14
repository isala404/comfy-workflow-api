"""Server extension for ComfyUI webhook integration."""

from .routes import register_routes
from .uploads import save_upload, cleanup_uploads

__all__ = ["register_routes", "save_upload", "cleanup_uploads"]
