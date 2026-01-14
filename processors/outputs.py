"""
Output detection and processing for webhook delivery.
Handles extracting output information from ComfyUI UI structures.
"""

import json
import logging
import os
from typing import Optional

try:
    import folder_paths
except ImportError:
    folder_paths = None

# Handle imports for both package and standalone usage
try:
    from ..core.types import OutputInfo
    from ..utils.mime import guess_mime_type, get_format_from_filename
except ImportError:
    from core.types import OutputInfo
    from utils.mime import guess_mime_type, get_format_from_filename

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
    if folder_paths is None:
        # Fallback if folder_paths not available
        base_dir = os.getcwd()
        return os.path.join(base_dir, folder_type, subfolder, filename)

    if folder_type == "output":
        base_dir = folder_paths.get_output_directory()
    elif folder_type == "temp":
        base_dir = folder_paths.get_temp_directory()
    elif folder_type == "input":
        base_dir = folder_paths.get_input_directory()
    else:
        base_dir = folder_paths.get_output_directory()

    if subfolder:
        return os.path.join(base_dir, subfolder, filename)
    return os.path.join(base_dir, filename)


def detect_output_type(
    ui_output: dict,
    node_id: Optional[str] = None,
    node_type: Optional[str] = None
) -> list[OutputInfo]:
    """
    Detect output type from ComfyUI UI return structure.

    Known UI structures:
    - {"images": [{filename, subfolder, type}, ...]}
    - {"audio": [{filename, subfolder, type}, ...]}
    - {"video": [{filename, subfolder, type}, ...]}
    - {"3d": [{filename, subfolder, type}, ...]}
    - {"latents": [{filename, subfolder, type}, ...]}
    - {"text": (value,)} or {"text": ["value"]}
    - {"result": [model_file, camera_info, bg_image]}

    Args:
        ui_output: The UI output dictionary from a node
        node_id: Optional node ID for metadata
        node_type: Optional node type for metadata

    Returns:
        List of OutputInfo objects describing the outputs
    """
    outputs = []

    if not ui_output or not isinstance(ui_output, dict):
        return outputs

    # Handle images
    if "images" in ui_output:
        images = ui_output["images"]
        if isinstance(images, list):
            for idx, item in enumerate(images):
                if isinstance(item, dict) and "filename" in item:
                    info = _create_output_info_from_file(
                        item, "image", node_id, node_type
                    )
                    outputs.append(info)

    # Handle audio
    if "audio" in ui_output:
        audio_items = ui_output["audio"]
        if isinstance(audio_items, list):
            for item in audio_items:
                if isinstance(item, dict) and "filename" in item:
                    info = _create_output_info_from_file(
                        item, "audio", node_id, node_type
                    )
                    outputs.append(info)

    # Handle video
    if "video" in ui_output:
        video_items = ui_output["video"]
        if isinstance(video_items, list):
            for item in video_items:
                if isinstance(item, dict) and "filename" in item:
                    info = _create_output_info_from_file(
                        item, "video", node_id, node_type
                    )
                    outputs.append(info)

    # Handle 3D meshes
    if "3d" in ui_output:
        mesh_items = ui_output["3d"]
        if isinstance(mesh_items, list):
            for item in mesh_items:
                if isinstance(item, dict) and "filename" in item:
                    info = _create_output_info_from_file(
                        item, "mesh", node_id, node_type
                    )
                    outputs.append(info)

    # Handle latents
    if "latents" in ui_output:
        latent_items = ui_output["latents"]
        if isinstance(latent_items, list):
            for item in latent_items:
                if isinstance(item, dict) and "filename" in item:
                    info = _create_output_info_from_file(
                        item, "latent", node_id, node_type
                    )
                    outputs.append(info)

    # Handle text outputs
    if "text" in ui_output:
        text_data = ui_output["text"]
        if isinstance(text_data, tuple) and len(text_data) > 0:
            # Text is often returned as a tuple
            text_value = str(text_data[0])
        elif isinstance(text_data, list) and len(text_data) > 0:
            text_value = str(text_data[0])
        elif isinstance(text_data, str):
            text_value = text_data
        else:
            text_value = str(text_data)

        outputs.append(OutputInfo(
            type="text",
            inline_content=text_value,
            mime_type="text/plain",
            node_id=node_id,
            node_type=node_type,
        ))

    # Handle 3D preview (result structure)
    if "result" in ui_output and "images" not in ui_output:
        result_data = ui_output["result"]
        outputs.append(OutputInfo(
            type="3d_preview",
            inline_content=json.dumps(result_data),
            mime_type="application/json",
            node_id=node_id,
            node_type=node_type,
        ))

    # Handle animated flag (for videos/GIFs saved as images)
    if "animated" in ui_output:
        # Mark the last image output as animated if applicable
        for output in reversed(outputs):
            if output.type == "image":
                output.format = "animated"
                break

    # Fallback: try to extract any file references from unknown structures
    if not outputs:
        outputs = _extract_unknown_outputs(ui_output, node_id, node_type)

    return outputs


def _create_output_info_from_file(
    item: dict,
    output_type: str,
    node_id: Optional[str],
    node_type: Optional[str]
) -> OutputInfo:
    """Create OutputInfo from a file reference dictionary."""
    filename = item.get("filename", "")
    subfolder = item.get("subfolder", "")
    folder_type = item.get("type", "output")

    # Get file path and size
    file_path = get_output_path(filename, subfolder, folder_type)
    file_size = None
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)

    return OutputInfo(
        type=output_type,
        filename=filename,
        subfolder=subfolder,
        folder_type=folder_type,
        mime_type=guess_mime_type(filename),
        format=get_format_from_filename(filename),
        file_size_bytes=file_size,
        node_id=node_id,
        node_type=node_type,
    )


def _extract_unknown_outputs(
    ui_output: dict,
    node_id: Optional[str],
    node_type: Optional[str]
) -> list[OutputInfo]:
    """
    Extract outputs from unknown UI structures.
    Looks for any dictionary with 'filename' key.
    """
    outputs = []

    for key, value in ui_output.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "filename" in item:
                    info = _create_output_info_from_file(
                        item, "unknown", node_id, node_type
                    )
                    outputs.append(info)
                    logger.info(
                        f"Extracted unknown output type '{key}': {item.get('filename')}"
                    )

    # If still no outputs, serialize the entire structure as JSON
    if not outputs and ui_output:
        logger.warning(
            f"Unknown output format, serializing as JSON: {list(ui_output.keys())}"
        )
        outputs.append(OutputInfo(
            type="unknown",
            inline_content=json.dumps(ui_output),
            mime_type="application/json",
            node_id=node_id,
            node_type=node_type,
        ))

    return outputs


def process_outputs_for_webhook(
    outputs: dict,
    meta: Optional[dict] = None
) -> list[OutputInfo]:
    """
    Process all outputs from a workflow execution for webhook delivery.

    Args:
        outputs: Dictionary mapping node_id -> UI output dict
        meta: Optional metadata dict mapping node_id -> node metadata

    Returns:
        List of all OutputInfo objects from all nodes
    """
    all_outputs = []

    for node_id, ui_output in outputs.items():
        # Get node type from metadata if available
        node_type = None
        if meta and node_id in meta:
            node_meta = meta[node_id]
            if isinstance(node_meta, dict):
                node_type = node_meta.get("node_type")

        # Detect outputs for this node
        node_outputs = detect_output_type(ui_output, node_id, node_type)
        all_outputs.extend(node_outputs)

    return all_outputs


def load_output_file(output_info: OutputInfo) -> Optional[bytes]:
    """
    Load the binary content of an output file.

    Args:
        output_info: OutputInfo with file location details

    Returns:
        File bytes or None if file doesn't exist
    """
    if not output_info.filename:
        return None

    file_path = get_output_path(
        output_info.filename,
        output_info.subfolder or "",
        output_info.folder_type or "output"
    )

    if not os.path.exists(file_path):
        logger.warning(f"Output file not found: {file_path}")
        return None

    try:
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read output file {file_path}: {e}")
        return None


def get_image_dimensions(output_info: OutputInfo) -> tuple[Optional[int], Optional[int]]:
    """
    Get dimensions of an image output.

    Args:
        output_info: OutputInfo for an image

    Returns:
        Tuple of (width, height) or (None, None) if unable to determine
    """
    if output_info.type != "image" or not output_info.filename:
        return None, None

    try:
        from PIL import Image

        file_path = get_output_path(
            output_info.filename,
            output_info.subfolder or "",
            output_info.folder_type or "output"
        )

        if os.path.exists(file_path):
            with Image.open(file_path) as img:
                return img.size  # (width, height)
    except Exception as e:
        logger.warning(f"Failed to get image dimensions: {e}")

    return None, None


def enrich_output_info(output_info: OutputInfo) -> OutputInfo:
    """
    Enrich OutputInfo with additional metadata.

    Adds dimensions for images, file size, etc.

    Args:
        output_info: The OutputInfo to enrich

    Returns:
        The same OutputInfo with additional fields populated
    """
    # Get file size if not already set
    if output_info.file_size_bytes is None and output_info.filename:
        file_path = get_output_path(
            output_info.filename,
            output_info.subfolder or "",
            output_info.folder_type or "output"
        )
        if os.path.exists(file_path):
            output_info.file_size_bytes = os.path.getsize(file_path)

    # Get image dimensions
    if output_info.type == "image" and (output_info.width is None or output_info.height is None):
        width, height = get_image_dimensions(output_info)
        if width and height:
            output_info.width = width
            output_info.height = height

    return output_info
