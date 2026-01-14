"""Handle file uploads for webhook requests."""

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Track uploads per request for cleanup
_upload_registry: dict[str, list[str]] = {}


def get_upload_directory() -> str:
    """Get the directory for webhook uploads."""
    try:
        import folder_paths
        # Use ComfyUI's input directory for uploads
        input_dir = folder_paths.get_input_directory()
        webhook_dir = os.path.join(input_dir, "webhook_uploads")
    except ImportError:
        # Fallback for testing
        webhook_dir = os.path.join(os.getcwd(), "input", "webhook_uploads")

    os.makedirs(webhook_dir, exist_ok=True)
    return webhook_dir


def save_upload(data: bytes, filename: str, request_id: str) -> str:
    """
    Save an uploaded file to the webhook uploads directory.

    Args:
        data: File content as bytes
        filename: Original filename
        request_id: Request ID for organizing uploads

    Returns:
        Full path to the saved file
    """
    upload_dir = get_upload_directory()

    # Create request-specific subdirectory
    request_dir = os.path.join(upload_dir, request_id)
    os.makedirs(request_dir, exist_ok=True)

    # Sanitize filename and make unique
    safe_filename = sanitize_filename(filename)
    base, ext = os.path.splitext(safe_filename)

    # Add short UUID to avoid collisions
    unique_filename = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(request_dir, unique_filename)

    # Write file
    with open(file_path, "wb") as f:
        f.write(data)

    # Track for cleanup
    if request_id not in _upload_registry:
        _upload_registry[request_id] = []
    _upload_registry[request_id].append(file_path)

    logger.debug(f"Saved upload: {file_path} ({len(data)} bytes)")
    return file_path


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage."""
    # Remove path separators and null bytes
    filename = filename.replace("/", "_").replace("\\", "_").replace("\x00", "")

    # Remove leading dots (hidden files)
    while filename.startswith("."):
        filename = filename[1:]

    # Default name if empty
    if not filename:
        filename = "upload"

    # Limit length
    if len(filename) > 200:
        base, ext = os.path.splitext(filename)
        filename = base[:200 - len(ext)] + ext

    return filename


def cleanup_uploads(request_id: str) -> int:
    """
    Clean up uploaded files for a request.

    Args:
        request_id: Request ID to clean up

    Returns:
        Number of files removed
    """
    count = 0

    # Remove tracked files
    if request_id in _upload_registry:
        for file_path in _upload_registry[request_id]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to remove upload {file_path}: {e}")

        del _upload_registry[request_id]

    # Also try to remove the request directory
    upload_dir = get_upload_directory()
    request_dir = os.path.join(upload_dir, request_id)

    try:
        if os.path.exists(request_dir):
            shutil.rmtree(request_dir)
            logger.debug(f"Removed upload directory: {request_dir}")
    except Exception as e:
        logger.warning(f"Failed to remove upload directory {request_dir}: {e}")

    return count


def get_upload_path(request_id: str, field_name: str) -> Optional[str]:
    """
    Get the path to an uploaded file by field name.

    This is used by WebhookInput node to load files.

    Args:
        request_id: Request ID
        field_name: Field name from the upload

    Returns:
        File path if found, None otherwise
    """
    # This is primarily resolved via extra_data["webhook_uploads"]
    # But we can search the registry as fallback
    if request_id in _upload_registry:
        for path in _upload_registry[request_id]:
            if field_name in path:
                return path
    return None


def cleanup_old_uploads(max_age_hours: int = 24) -> int:
    """
    Clean up old upload directories.

    Args:
        max_age_hours: Remove directories older than this

    Returns:
        Number of directories removed
    """
    import time

    upload_dir = get_upload_directory()
    if not os.path.exists(upload_dir):
        return 0

    count = 0
    cutoff = time.time() - (max_age_hours * 3600)

    try:
        for entry in os.scandir(upload_dir):
            if entry.is_dir():
                try:
                    mtime = entry.stat().st_mtime
                    if mtime < cutoff:
                        shutil.rmtree(entry.path)
                        count += 1
                        logger.debug(f"Cleaned up old upload directory: {entry.path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {entry.path}: {e}")
    except Exception as e:
        logger.warning(f"Failed to scan upload directory: {e}")

    if count > 0:
        logger.info(f"Cleaned up {count} old upload directories")

    return count
