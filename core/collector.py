"""
Output collector for webhook delivery.

IMPORTANT: This collector does NOT re-encode outputs.
It reads the actual files saved by ComfyUI output nodes and sends them as-is.

- User saves as WebP → WebP is sent
- User saves as FLAC → FLAC is sent
- User saves as GIF → GIF is sent

Zero information loss, zero re-encoding overhead.
"""

import logging
import os
from typing import Optional

from .types import OutputInfo
from ..utils.mime import guess_mime_type
from ..utils.files import get_output_path, get_image_dimensions

logger = logging.getLogger(__name__)


async def collect_outputs(
    outputs: dict,
    meta: Optional[dict] = None
) -> tuple[list[tuple[str, str, bytes, str]], list[dict]]:
    """
    Collect all outputs from a workflow execution.

    This function reads the ACTUAL files saved by output nodes.
    No re-encoding - files are sent exactly as saved.

    Args:
        outputs: Dictionary mapping node_id -> UI output dict
                 e.g., {"9": {"images": [{"filename": "out.webp", ...}]}}
        meta: Optional metadata dict mapping node_id -> node metadata

    Returns:
        Tuple of:
        - files: List of (field_name, filename, data, mime_type) tuples
        - metadata: List of output info dicts for the metadata payload
    """
    files = []
    metadata = []
    file_index = 0

    for node_id, ui_output in outputs.items():
        if not ui_output or not isinstance(ui_output, dict):
            continue

        # Get node type from metadata if available
        node_type = None
        if meta and node_id in meta:
            node_meta = meta[node_id]
            if isinstance(node_meta, dict):
                node_type = node_meta.get("node_type")

        # Process all output keys generically
        # ComfyUI uses: images, audio, video, gifs, 3d, latents, text, etc.
        for output_key, items in ui_output.items():
            if not isinstance(items, list):
                # Handle non-list outputs (e.g., text as tuple)
                if output_key == "text":
                    text_content = _extract_text(items)
                    if text_content:
                        info = OutputInfo(
                            filename=f"text_{node_id}.txt",
                            mime_type="text/plain",
                            node_id=node_id,
                            node_type=node_type,
                            output_key=output_key,
                            file_size_bytes=len(text_content.encode("utf-8")),
                        )
                        files.append((
                            f"file_{file_index}",
                            info.filename,
                            text_content.encode("utf-8"),
                            info.mime_type,
                        ))
                        metadata.append(info.to_dict())
                        file_index += 1
                continue

            # Process list of file references
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue

                if "filename" not in item:
                    continue

                # This is a file reference
                filename = item["filename"]
                subfolder = item.get("subfolder", "")
                folder_type = item.get("type", "output")

                # Get full path and read file
                file_path = get_output_path(filename, subfolder, folder_type)

                if not os.path.exists(file_path):
                    logger.warning(f"Output file not found: {file_path}")
                    continue

                try:
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                except Exception as e:
                    logger.error(f"Failed to read output file {file_path}: {e}")
                    continue

                # Determine MIME type from actual filename
                mime_type = guess_mime_type(filename)

                # Build output info
                info = OutputInfo(
                    filename=filename,
                    mime_type=mime_type,
                    node_id=node_id,
                    node_type=node_type,
                    output_key=output_key,
                    file_size_bytes=len(file_data),
                    subfolder=subfolder if subfolder else None,
                    folder_type=folder_type,
                )

                # Add type-specific metadata
                _enrich_output_info(info, file_path, file_data)

                # Add batch info if multiple items
                if len(items) > 1:
                    info.batch_index = idx
                    info.batch_size = len(items)

                files.append((
                    f"file_{file_index}",
                    filename,
                    file_data,
                    mime_type,
                ))
                metadata.append(info.to_dict())
                file_index += 1

    logger.debug(f"Collected {len(files)} output files")
    return files, metadata


def _extract_text(value) -> Optional[str]:
    """Extract text content from various formats."""
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and len(value) > 0:
        return str(value[0])
    if isinstance(value, list) and len(value) > 0:
        return str(value[0])
    return None


def _enrich_output_info(info: OutputInfo, file_path: str, file_data: bytes):
    """
    Add type-specific metadata to OutputInfo.

    This reads metadata from the file without modifying it.
    """
    # Extract format from filename
    if "." in info.filename:
        info.format = info.filename.rsplit(".", 1)[1].lower()

    # Image dimensions
    if info.mime_type.startswith("image/"):
        try:
            width, height = get_image_dimensions(file_path)
            if width and height:
                info.width = width
                info.height = height
        except Exception as e:
            logger.debug(f"Could not get image dimensions: {e}")

    # Audio metadata
    elif info.mime_type.startswith("audio/"):
        try:
            _extract_audio_metadata(info, file_data)
        except Exception as e:
            logger.debug(f"Could not get audio metadata: {e}")

    # Video metadata
    elif info.mime_type.startswith("video/"):
        try:
            _extract_video_metadata(info, file_path)
        except Exception as e:
            logger.debug(f"Could not get video metadata: {e}")


def _extract_audio_metadata(info: OutputInfo, file_data: bytes):
    """Extract audio metadata without decoding the full file."""
    # Simple detection based on file headers
    if file_data[:4] == b"fLaC":
        # FLAC - could parse header for sample rate
        pass
    elif file_data[:4] == b"RIFF":
        # WAV - parse header
        try:
            import struct
            if len(file_data) >= 44:
                # Channels at offset 22 (2 bytes)
                # Sample rate at offset 24 (4 bytes)
                sample_rate = struct.unpack("<I", file_data[24:28])[0]
                info.sample_rate = sample_rate
        except Exception:
            pass


def _extract_video_metadata(info: OutputInfo, file_path: str):
    """Extract video metadata using available tools."""
    try:
        import av
        container = av.open(file_path)
        video_stream = container.streams.video[0]

        info.width = video_stream.width
        info.height = video_stream.height

        if video_stream.average_rate:
            info.fps = float(video_stream.average_rate)

        if video_stream.frames:
            info.frame_count = video_stream.frames

        if video_stream.duration and video_stream.time_base:
            info.duration_seconds = float(video_stream.duration * video_stream.time_base)

        container.close()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"av video metadata extraction failed: {e}")
