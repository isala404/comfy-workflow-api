"""File path resolution and utilities."""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_output_path(
    filename: str,
    subfolder: str = "",
    folder_type: str = "output"
) -> str:
    """
    Resolve the full path to an output file.

    Args:
        filename: Name of the file
        subfolder: Subfolder within the directory
        folder_type: Type of folder (output, temp, input)

    Returns:
        Full filesystem path to the file
    """
    try:
        import folder_paths

        if folder_type == "output":
            base_dir = folder_paths.get_output_directory()
        elif folder_type == "temp":
            base_dir = folder_paths.get_temp_directory()
        elif folder_type == "input":
            base_dir = folder_paths.get_input_directory()
        else:
            base_dir = folder_paths.get_output_directory()

    except ImportError:
        # Fallback if folder_paths not available
        base_dir = os.path.join(os.getcwd(), folder_type)

    if subfolder:
        return os.path.join(base_dir, subfolder, filename)
    return os.path.join(base_dir, filename)


def get_image_dimensions(file_path: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Get dimensions of an image file.

    Args:
        file_path: Path to the image file

    Returns:
        Tuple of (width, height) or (None, None) if unable to determine
    """
    if not os.path.exists(file_path):
        return None, None

    try:
        from PIL import Image
        with Image.open(file_path) as img:
            return img.size  # (width, height)
    except ImportError:
        logger.debug("PIL not available for image dimension detection")
    except Exception as e:
        logger.debug(f"Failed to get image dimensions: {e}")

    return None, None


def get_temp_directory() -> str:
    """Get the webhook temp directory."""
    try:
        import folder_paths
        temp_dir = folder_paths.get_temp_directory()
    except ImportError:
        temp_dir = os.path.join(os.getcwd(), "temp")

    webhook_temp = os.path.join(temp_dir, "webhook")
    os.makedirs(webhook_temp, exist_ok=True)
    return webhook_temp


def save_temp_file(data: bytes, filename: str, request_id: str = "") -> str:
    """
    Save data to a temp file.

    Args:
        data: File content
        filename: Filename to use
        request_id: Optional request ID for organization

    Returns:
        Path to saved file
    """
    temp_dir = get_temp_directory()

    if request_id:
        subdir = os.path.join(temp_dir, request_id)
        os.makedirs(subdir, exist_ok=True)
        file_path = os.path.join(subdir, filename)
    else:
        file_path = os.path.join(temp_dir, filename)

    with open(file_path, "wb") as f:
        f.write(data)

    return file_path


def cleanup_temp_files(request_id: str) -> int:
    """
    Clean up temp files for a request.

    Args:
        request_id: Request ID

    Returns:
        Number of files removed
    """
    import shutil

    temp_dir = get_temp_directory()
    subdir = os.path.join(temp_dir, request_id)

    if not os.path.exists(subdir):
        return 0

    try:
        count = len(os.listdir(subdir))
        shutil.rmtree(subdir)
        return count
    except Exception as e:
        logger.warning(f"Failed to clean up temp files: {e}")
        return 0


def detect_file_type(file_path: str) -> Optional[str]:
    """
    Detect file type from magic bytes.

    Args:
        file_path: Path to file

    Returns:
        Detected type: "image", "audio", "video", "3d", or None
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            header = f.read(32)
    except Exception:
        return None

    # Image formats
    if header.startswith(b"\x89PNG"):
        return "image"
    if header.startswith(b"\xff\xd8\xff"):
        return "image"
    if header.startswith(b"RIFF") and b"WEBP" in header[:16]:
        return "image"
    if header.startswith(b"GIF8"):
        return "image"
    if header.startswith(b"BM"):
        return "image"

    # Audio formats
    if header.startswith(b"fLaC"):
        return "audio"
    if header.startswith(b"RIFF") and b"WAVE" in header[:16]:
        return "audio"
    if header.startswith(b"ID3") or header.startswith(b"\xff\xfb"):
        return "audio"
    if header.startswith(b"OggS"):
        return "audio"

    # Video formats
    if header[4:8] == b"ftyp":  # MP4/MOV
        return "video"
    if header.startswith(b"\x1a\x45\xdf\xa3"):  # WebM/MKV
        return "video"

    # 3D formats
    if header.startswith(b"glTF"):
        return "3d"
    if b"solid " in header.lower() or header.startswith(b"\x00"):  # STL
        return "3d"

    return None
