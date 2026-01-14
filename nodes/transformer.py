"""WebhookTransformer node - Extract a single atomic field from webhook context."""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..core.types import WebhookContext, WebhookField
from ..processors.inputs import WebhookInputProcessor
from ..utils.helpers import format_size, run_async

logger = logging.getLogger(__name__)

_processor = None


def get_processor() -> WebhookInputProcessor:
    global _processor
    if _processor is None:
        _processor = WebhookInputProcessor()
    return _processor


class WebhookTransformer:
    """
    Extract a single atomic field from the webhook request.

    Field names are atomic (flat) - no JSON paths, no nesting.
    Outputs wildcard type (*) that connects to any ComfyUI input.

    Type Inference:
    - Text fields with numbers -> INT or FLOAT
    - Text fields -> STRING
    - File uploads -> IMAGE, AUDIO, VIDEO, MESH, LATENT based on content type
    - Boolean strings ("true"/"false") -> BOOLEAN

    Missing Field Handling:
    - If field missing AND default_value provided -> use default
    - If field missing AND NO default -> STOP workflow with error
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "webhook_context": ("WEBHOOK_CONTEXT",),
                "field": ("STRING", {
                    "default": "prompt",
                    "tooltip": "Field name to extract (atomic, no nesting)"
                }),
            },
            "optional": {
                "default_value": ("*", {
                    "tooltip": "Fallback value if field not found (leave empty to require field)"
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable debug logging"
                }),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "transform"
    CATEGORY = "webhook"
    DESCRIPTION = "Extract a single atomic field from webhook request. Outputs wildcard type that connects to any input."

    @classmethod
    def VALIDATE_INPUTS(cls, webhook_context=None, field="", default_value=None, debug=False):
        """Validate inputs before execution."""
        # Basic validation - actual field checking happens in transform()
        if not field or not field.strip():
            return "Field name cannot be empty"
        return True

    def transform(
        self,
        webhook_context,
        field: str,
        default_value=None,
        debug=False,
    ):
        """
        Extract field from webhook context.

        Raises:
            ValueError: If field is missing and no default provided
        """
        # Validate context (use duck typing for flexibility with module loading)
        if not hasattr(webhook_context, 'get_field_value') or not hasattr(webhook_context, 'has_field'):
            if default_value is not None:
                if debug:
                    self._print_debug_default(field, default_value, "Invalid webhook context")
                return (default_value,)
            raise ValueError(
                f"[Webhook Transformer] ERROR: Invalid webhook context for field '{field}'"
            )

        # Check if field exists
        has_field = webhook_context.has_field(field)

        if not has_field:
            # Field not found
            if default_value is not None:
                if debug:
                    self._print_debug_default(field, default_value, "Field not found")
                return (default_value,)

            # No default - STOP workflow
            available_fields = webhook_context.get_field_names()
            self._print_error(field, available_fields)
            raise ValueError(
                f"[Webhook Transformer] Required field '{field}' not found. "
                f"Available fields: {', '.join(available_fields) if available_fields else '(none)'}. "
                f"Provide field or set default_value."
            )

        # Get field value
        webhook_field = webhook_context.get_field(field)
        raw_value = webhook_context.get_field_value(field)

        # Process and convert value based on type inference
        processed_value, value_type, debug_info = self._process_value(
            raw_value, webhook_field
        )

        if debug:
            self._print_debug_found(field, processed_value, value_type, debug_info)

        return (processed_value,)

    def _process_value(
        self,
        raw_value: Any,
        webhook_field: Optional[WebhookField]
    ) -> Tuple[Any, str, Dict]:
        """
        Process raw value and infer type.

        Returns:
            tuple: (processed_value, type_name, debug_info_dict)
        """
        debug_info = {}

        # Handle file uploads
        if webhook_field and webhook_field.is_file:
            return self._process_file(raw_value, webhook_field)

        # Handle string values
        if isinstance(raw_value, str):
            # Try to infer type from string content

            # Boolean
            if raw_value.lower() in ("true", "false", "yes", "no", "1", "0"):
                bool_value = raw_value.lower() in ("true", "yes", "1")
                debug_info["original"] = raw_value
                return (bool_value, "BOOLEAN", debug_info)

            # Integer
            try:
                if raw_value.isdigit() or (raw_value.startswith("-") and raw_value[1:].isdigit()):
                    int_value = int(raw_value)
                    debug_info["original"] = raw_value
                    return (int_value, "INT", debug_info)
            except (ValueError, AttributeError):
                pass

            # Float
            try:
                if "." in raw_value or "e" in raw_value.lower():
                    float_value = float(raw_value)
                    debug_info["original"] = raw_value
                    return (float_value, "FLOAT", debug_info)
            except (ValueError, AttributeError):
                pass

            # Check if it's base64 encoded data
            if raw_value.startswith("data:"):
                return self._process_data_uri(raw_value)

            # Plain string
            debug_info["length"] = len(raw_value)
            return (raw_value, "STRING", debug_info)

        # Handle numeric types directly
        if isinstance(raw_value, bool):
            return (raw_value, "BOOLEAN", debug_info)
        if isinstance(raw_value, int):
            return (raw_value, "INT", debug_info)
        if isinstance(raw_value, float):
            return (raw_value, "FLOAT", debug_info)

        # Handle bytes (file data)
        if isinstance(raw_value, bytes):
            return self._process_bytes(raw_value)

        # Handle torch tensors
        if torch.is_tensor(raw_value):
            debug_info["shape"] = list(raw_value.shape)
            debug_info["dtype"] = str(raw_value.dtype)
            return (raw_value, "TENSOR", debug_info)

        # Handle dicts (could be LATENT, AUDIO, etc.)
        if isinstance(raw_value, dict):
            # Check for known dict types
            if "samples" in raw_value:
                return (raw_value, "LATENT", {"type": "latent_dict"})
            if "waveform" in raw_value:
                return (raw_value, "AUDIO", {"type": "audio_dict"})

            # Generic dict - return as is
            debug_info["keys"] = list(raw_value.keys())
            return (raw_value, "DICT", debug_info)

        # Handle lists
        if isinstance(raw_value, list):
            debug_info["length"] = len(raw_value)
            return (raw_value, "LIST", debug_info)

        # Default: return as-is
        debug_info["python_type"] = type(raw_value).__name__
        return (raw_value, "ANY", debug_info)

    def _process_file(
        self,
        raw_value: Any,
        webhook_field: WebhookField
    ) -> Tuple[Any, str, Dict]:
        """Process file upload based on content type."""
        content_type = webhook_field.content_type or ""
        debug_info = {
            "filename": webhook_field.filename,
            "content_type": content_type,
            "size_bytes": webhook_field.size_bytes,
        }

        data = raw_value if isinstance(raw_value, bytes) else None

        if data is None:
            return (raw_value, "UNKNOWN", debug_info)

        # Image types
        if content_type.startswith("image/"):
            try:
                tensor, info = self._decode_image(data)
                debug_info.update(info)
                return (tensor, "IMAGE", debug_info)
            except Exception as e:
                logger.warning(f"Failed to decode image: {e}")
                return (data, "BYTES", debug_info)

        # Audio types
        if content_type.startswith("audio/"):
            try:
                audio_dict, info = self._decode_audio(data)
                debug_info.update(info)
                return (audio_dict, "AUDIO", debug_info)
            except Exception as e:
                logger.warning(f"Failed to decode audio: {e}")
                return (data, "BYTES", debug_info)

        # Video types
        if content_type.startswith("video/"):
            try:
                video, info = self._decode_video(data)
                debug_info.update(info)
                return (video, "VIDEO", debug_info)
            except Exception as e:
                logger.warning(f"Failed to decode video: {e}")
                return (data, "BYTES", debug_info)

        # 3D mesh types
        if content_type in ("model/gltf-binary", "model/obj", "model/stl", "application/octet-stream"):
            # Try to detect from filename
            filename = webhook_field.filename or ""
            if filename.endswith((".glb", ".gltf", ".obj", ".stl")):
                try:
                    mesh, info = self._decode_mesh(data, filename)
                    debug_info.update(info)
                    return (mesh, "MESH", debug_info)
                except Exception as e:
                    logger.warning(f"Failed to decode mesh: {e}")
                    return (data, "BYTES", debug_info)

        # Safetensors (latent)
        if webhook_field.filename and webhook_field.filename.endswith(".safetensors"):
            try:
                latent, info = self._decode_latent(data)
                debug_info.update(info)
                return (latent, "LATENT", debug_info)
            except Exception as e:
                logger.warning(f"Failed to decode latent: {e}")
                return (data, "BYTES", debug_info)

        # Default: return raw bytes
        return (data, "BYTES", debug_info)

    def _process_data_uri(self, data_uri: str) -> Tuple[Any, str, Dict]:
        """Process data URI (base64 encoded data)."""
        processor = get_processor()
        debug_info = {}

        try:
            # Detect type from data URI
            if data_uri.startswith("data:image/"):
                tensor = run_async(processor.process_image(data_uri))
                debug_info["source"] = "base64"
                debug_info["dimensions"] = f"{tensor.shape[2]}x{tensor.shape[1]}"
                return (tensor, "IMAGE", debug_info)

            elif data_uri.startswith("data:audio/"):
                audio = run_async(processor.process_audio(data_uri))
                debug_info["source"] = "base64"
                return (audio, "AUDIO", debug_info)

            elif data_uri.startswith("data:video/"):
                video = run_async(processor.process_video(data_uri))
                debug_info["source"] = "base64"
                return (video, "VIDEO", debug_info)

        except Exception as e:
            logger.warning(f"Failed to process data URI: {e}")

        # Return as string if can't process
        debug_info["length"] = len(data_uri)
        return (data_uri, "STRING", debug_info)

    def _process_bytes(self, data: bytes) -> Tuple[Any, str, Dict]:
        """Process raw bytes by detecting format."""
        debug_info = {"size_bytes": len(data)}

        # Try to detect image from magic bytes
        if data[:8] == b'\x89PNG\r\n\x1a\n' or data[:2] == b'\xff\xd8':
            try:
                tensor, info = self._decode_image(data)
                debug_info.update(info)
                return (tensor, "IMAGE", debug_info)
            except Exception:
                pass

        # Try to detect audio
        if data[:4] == b'fLaC' or data[:4] == b'RIFF':
            try:
                audio, info = self._decode_audio(data)
                debug_info.update(info)
                return (audio, "AUDIO", debug_info)
            except Exception:
                pass

        # Default: return raw bytes
        return (data, "BYTES", debug_info)

    def _decode_image(self, data: bytes) -> Tuple[torch.Tensor, Dict]:
        """Decode image bytes to tensor."""
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")

        np_image = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image).unsqueeze(0)  # [1, H, W, C]

        info = {
            "dimensions": f"{img.width}x{img.height}",
            "channels": 3,
            "dtype": "float32",
        }
        return tensor, info

    def _decode_audio(self, data: bytes) -> Tuple[Dict, Dict]:
        """Decode audio bytes to audio dict."""
        try:
            import av
        except ImportError:
            raise ImportError("av library required for audio decoding")

        buffer = io.BytesIO(data)
        container = av.open(buffer)

        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate

        frames = []
        for frame in container.decode(audio=0):
            frames.append(frame.to_ndarray())

        container.close()

        audio_np = np.concatenate(frames, axis=1)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)

        info = {
            "sample_rate": sample_rate,
            "channels": audio_np.shape[0],
            "duration_seconds": audio_np.shape[1] / sample_rate,
        }

        return {"waveform": waveform, "sample_rate": sample_rate}, info

    def _decode_video(self, data: bytes) -> Tuple[Dict, Dict]:
        """Decode video bytes to video dict."""
        try:
            import av
        except ImportError:
            raise ImportError("av library required for video decoding")

        buffer = io.BytesIO(data)
        container = av.open(buffer)

        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)

        frames = []
        for frame in container.decode(video=0):
            np_frame = frame.to_ndarray(format="rgb24")
            frames.append(np_frame)

        container.close()

        if not frames:
            raise ValueError("No video frames found")

        frames_np = np.stack(frames)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0

        info = {
            "fps": fps,
            "frame_count": len(frames),
            "dimensions": f"{frames_np.shape[2]}x{frames_np.shape[1]}",
        }

        return {"images": frames_tensor, "fps": fps}, info

    def _decode_mesh(self, data: bytes, filename: str) -> Tuple[Dict, Dict]:
        """Decode mesh bytes."""
        processor = get_processor()

        # Use processor's mesh parsing
        if filename.endswith(".glb") or filename.endswith(".gltf"):
            mesh = processor._parse_glb(data)
        elif filename.endswith(".obj"):
            mesh = processor._parse_obj(data.decode("utf-8"))
        elif filename.endswith(".stl"):
            mesh = processor._parse_stl_binary(data)
        else:
            raise ValueError(f"Unsupported mesh format: {filename}")

        vertices = mesh.get("vertices")
        faces = mesh.get("faces")

        info = {
            "vertex_count": vertices.shape[1] if vertices is not None else 0,
            "face_count": faces.shape[1] if faces is not None else 0,
        }

        return mesh, info

    def _decode_latent(self, data: bytes) -> Tuple[Dict, Dict]:
        """Decode latent from safetensors."""
        from safetensors.torch import load

        buffer = io.BytesIO(data)
        tensors = load(buffer.read())

        samples = tensors.get("samples")
        if samples is None:
            raise ValueError("No 'samples' key in safetensors")

        info = {
            "shape": list(samples.shape),
            "dtype": str(samples.dtype),
        }

        return {"samples": samples}, info

    def _print_debug_found(
        self,
        field: str,
        value: Any,
        value_type: str,
        debug_info: dict
    ):
        """Print debug info for found field."""
        lines = [
            "",
            f"[Webhook Transformer] Field: \"{field}\"",
            "=" * 50,
            f"  Status:   Found",
            f"  Type:     {value_type}",
        ]

        # Add type-specific info
        if value_type == "STRING":
            preview = str(value)
            if len(preview) > 50:
                preview = preview[:47] + "..."
            lines.append(f"  Value:    \"{preview}\"")
            lines.append(f"  Length:   {debug_info.get('length', len(str(value)))} characters")

        elif value_type == "IMAGE":
            lines.append(f"  Dimensions: {debug_info.get('dimensions', 'unknown')}")
            lines.append(f"  Channels:   {debug_info.get('channels', 'unknown')}")
            lines.append(f"  Dtype:      {debug_info.get('dtype', 'unknown')}")
            if debug_info.get('size_bytes'):
                lines.append(f"  Size:       {format_size(debug_info['size_bytes'])}")
            lines.append(f"  Source:     {debug_info.get('source', 'file upload')}")

        elif value_type == "AUDIO":
            lines.append(f"  Sample Rate: {debug_info.get('sample_rate', 'unknown')} Hz")
            lines.append(f"  Channels:    {debug_info.get('channels', 'unknown')}")
            if debug_info.get('duration_seconds'):
                lines.append(f"  Duration:    {debug_info['duration_seconds']:.2f}s")

        elif value_type in ("INT", "FLOAT", "BOOLEAN"):
            lines.append(f"  Value:    {value}")
            if debug_info.get('original'):
                lines.append(f"  Original: \"{debug_info['original']}\"")

        elif value_type == "LATENT":
            if debug_info.get('shape'):
                lines.append(f"  Shape:    {debug_info['shape']}")

        elif value_type == "MESH":
            lines.append(f"  Vertices: {debug_info.get('vertex_count', 'unknown')}")
            lines.append(f"  Faces:    {debug_info.get('face_count', 'unknown')}")

        else:
            # Generic info
            preview = str(value)
            if len(preview) > 50:
                preview = preview[:47] + "..."
            lines.append(f"  Value:    {preview}")

        lines.append("=" * 50)
        print("\n".join(lines))

    def _print_debug_default(self, field: str, default_value: Any, reason: str):
        """Print debug info when using default value."""
        lines = [
            "",
            f"[Webhook Transformer] Field: \"{field}\"",
            "=" * 50,
            f"  Status:   Not Found -> Using Default",
            f"  Reason:   {reason}",
            f"  Type:     {type(default_value).__name__}",
        ]

        preview = str(default_value)
        if len(preview) > 50:
            preview = preview[:47] + "..."
        lines.append(f"  Value:    {preview}")

        lines.append("=" * 50)
        print("\n".join(lines))

    def _print_error(self, field: str, available_fields: List[str]):
        """Print error when field is missing and no default."""
        lines = [
            "",
            "[Webhook Transformer] ERROR",
            "=" * 50,
            f"  Field:    \"{field}\"",
            f"  Status:   NOT FOUND",
            "",
            "  Available fields:",
        ]

        if available_fields:
            for i, name in enumerate(available_fields):
                prefix = "    +-" if i == len(available_fields) - 1 else "    |-"
                lines.append(f"{prefix} {name}")
        else:
            lines.append("    (none)")

        lines.append("")
        lines.append("  -> Workflow stopped. Provide field or set default_value.")
        lines.append("=" * 50)

        print("\n".join(lines))
