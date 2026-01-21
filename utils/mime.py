"""MIME type utilities for webhook file handling."""

import os
from typing import Optional

# Comprehensive MIME type mappings for ComfyUI output types
MIME_TYPES = {
    # Images
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",

    # Audio
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",

    # Video
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",

    # 3D Models
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".obj": "text/plain",
    ".stl": "model/stl",
    ".fbx": "application/octet-stream",
    ".ply": "application/x-ply",

    # 3D/Depth formats
    ".exr": "image/x-exr",
    ".spz": "application/octet-stream",

    # Data/Other
    ".json": "application/json",
    ".safetensors": "application/octet-stream",
    ".pt": "application/octet-stream",
    ".pth": "application/octet-stream",
    ".ckpt": "application/octet-stream",
    ".latent": "application/octet-stream",
    ".txt": "text/plain",
    ".csv": "text/csv",
}

# Reverse mapping for getting extension from MIME type
EXTENSION_FOR_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/flac": ".flac",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "model/gltf-binary": ".glb",
    "model/gltf+json": ".gltf",
    "application/json": ".json",
    "text/plain": ".txt",
}


def guess_mime_type(filename: str) -> str:
    """
    Guess MIME type from filename extension.

    Args:
        filename: Filename with extension

    Returns:
        MIME type string, defaults to "application/octet-stream"
    """
    if not filename:
        return "application/octet-stream"

    ext = os.path.splitext(filename.lower())[1]
    return MIME_TYPES.get(ext, "application/octet-stream")


def get_extension_for_mime(mime_type: str) -> Optional[str]:
    """
    Get file extension for a MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        File extension including dot, or None if unknown
    """
    return EXTENSION_FOR_MIME.get(mime_type)


def get_output_type_from_mime(mime_type: str) -> str:
    """
    Determine output type category from MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        Output type: image, audio, video, mesh, text, or unknown
    """
    if mime_type.startswith("image/"):
        return "image"
    elif mime_type.startswith("audio/"):
        return "audio"
    elif mime_type.startswith("video/"):
        return "video"
    elif mime_type.startswith("model/") or mime_type in ("application/octet-stream",):
        # Check if it's a known 3D format
        return "mesh"
    elif mime_type.startswith("text/") or mime_type == "application/json":
        return "text"
    else:
        return "unknown"


def get_format_from_filename(filename: str) -> Optional[str]:
    """
    Extract format (extension without dot) from filename.

    Args:
        filename: Filename with extension

    Returns:
        Format string (e.g., "png", "mp3") or None
    """
    if not filename:
        return None
    ext = os.path.splitext(filename.lower())[1]
    return ext[1:] if ext else None
