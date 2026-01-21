"""WebhookSend node - Send workflow outputs to callback URL."""

import io
import json
import logging
import struct
from datetime import datetime
from typing import Optional, Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ..core.types import WebhookContext, WebhookResult, OutputInfo
from ..core.client import get_webhook_client
from ..utils.helpers import format_size, run_async

logger = logging.getLogger(__name__)


class WebhookSend:
    """
    Send workflow outputs to the callback URL as multipart/form-data.

    Supports 5 customizable wildcard fields. Each field accepts any
    ComfyUI type (IMAGE, AUDIO, VIDEO, MESH, LATENT, MASK, STRING, etc.)
    and automatically encodes it appropriately.

    Field Encoding:
    - IMAGE: PNG file
    - AUDIO: FLAC file
    - VIDEO: MP4 file
    - MESH: GLB file
    - LATENT: Safetensors file
    - MASK: PNG file (grayscale)
    - STRING: Text part
    - INT/FLOAT/BOOLEAN: Text part

    Always includes a metadata JSON field with request info.
    Payload is always gzip compressed (Content-Encoding: gzip).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "webhook_context": ("WEBHOOK_CONTEXT",),
            },
            "optional": {
                # Field 1
                "field_1": ("*", {"tooltip": "First output (any type)"}),
                "field_1_name": ("STRING", {
                    "default": "output_1",
                    "tooltip": "Field name for first output"
                }),
                # Field 2
                "field_2": ("*", {"tooltip": "Second output (any type)"}),
                "field_2_name": ("STRING", {
                    "default": "output_2",
                    "tooltip": "Field name for second output"
                }),
                # Field 3
                "field_3": ("*", {"tooltip": "Third output (any type)"}),
                "field_3_name": ("STRING", {
                    "default": "output_3",
                    "tooltip": "Field name for third output"
                }),
                # Field 4
                "field_4": ("*", {"tooltip": "Fourth output (any type)"}),
                "field_4_name": ("STRING", {
                    "default": "output_4",
                    "tooltip": "Field name for fourth output"
                }),
                # Field 5
                "field_5": ("*", {"tooltip": "Fifth output (any type)"}),
                "field_5_name": ("STRING", {
                    "default": "output_5",
                    "tooltip": "Field name for fifth output"
                }),
                # Debug
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable debug logging"
                }),
                # Batch outputs
                "batch_outputs": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Send all items in batched outputs (images/masks). Field names become prefixes with _0, _1, etc. suffixes."
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "send"
    CATEGORY = "webhook"
    DESCRIPTION = "Send workflow outputs to callback URL as gzip-compressed multipart/form-data. Supports 5 customizable wildcard fields."

    def send(
        self,
        webhook_context,
        field_1=None,
        field_1_name="output_1",
        field_2=None,
        field_2_name="output_2",
        field_3=None,
        field_3_name="output_3",
        field_4=None,
        field_4_name="output_4",
        field_5=None,
        field_5_name="output_5",
        debug=False,
        batch_outputs=False,
    ):
        if not isinstance(webhook_context, WebhookContext):
            logger.warning("WebhookSend: Invalid webhook context")
            return {}

        if not webhook_context.callback_url:
            logger.warning("WebhookSend: No callback URL configured")
            return {}

        # Collect non-None fields
        fields_to_send = []
        for field, name in [
            (field_1, field_1_name),
            (field_2, field_2_name),
            (field_3, field_3_name),
            (field_4, field_4_name),
            (field_5, field_5_name),
        ]:
            if field is not None:
                fields_to_send.append((name, field))

        try:
            result = run_async(
                self._send_async(webhook_context, fields_to_send, debug, batch_outputs)
            )
            return {}
        except Exception as e:
            logger.error(f"WebhookSend failed: {e}", exc_info=True)
            if debug:
                self._print_error(webhook_context, str(e))
            return {}

    async def _send_async(
        self,
        context: WebhookContext,
        fields_to_send: List[Tuple[str, Any]],
        debug: bool,
        batch_outputs: bool = False,
    ):
        """Async implementation of send."""
        files = []
        outputs = []
        start_time = datetime.now()

        # Process each field
        for name, value in fields_to_send:
            encoded = self._encode_value(name, value, batch_outputs)
            if encoded:
                # Check if we got a list of encoded items (batched) or single item
                if isinstance(encoded, list):
                    # Batched output - expand with _N suffixes
                    for i, (filename, data, mime_type, output_info) in enumerate(encoded):
                        batch_name = f"{name}_{i}"
                        files.append((batch_name, filename, data, mime_type))
                        outputs.append(output_info.to_dict())
                else:
                    # Single item
                    filename, data, mime_type, output_info = encoded
                    files.append((name, filename, data, mime_type))
                    outputs.append(output_info.to_dict())

        # Build metadata
        metadata = {
            "request_id": context.request_id,
            "prompt_id": context.prompt_id,
            "timestamp": datetime.now().isoformat(),
            "fields": outputs,
        }

        # Send request
        client = get_webhook_client()

        result = await client.send_multipart(
            url=context.callback_url,
            metadata=metadata,
            files=files,
            headers=context.get_auth_headers(),
            timeout=context.timeout,
            max_retries=context.max_retries,
            compress=True,  # Always use gzip compression
        )

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        if debug:
            self._print_debug(context, files, outputs, result, elapsed_ms)

        if not result.success:
            logger.error(f"WebhookSend failed: {result.error}")
        else:
            logger.info(
                f"WebhookSend: Sent {len(files)} fields to {context.callback_url}"
            )

    def _encode_value(
        self,
        name: str,
        value: Any,
        batch_outputs: bool = False,
    ) -> Optional[Union[Tuple[str, bytes, str, OutputInfo], List[Tuple[str, bytes, str, OutputInfo]]]]:
        """
        Encode value to appropriate format for multipart upload.

        Returns:
            tuple: (filename, data_bytes, mime_type, output_info) or None
            list: List of tuples when batch_outputs=True and value is batched
        """
        # IMAGE tensor
        if torch.is_tensor(value) and len(value.shape) == 4:
            # Batched images [B, H, W, C]
            batch_size = value.shape[0]

            if batch_outputs and batch_size > 1:
                # Return all images as separate items
                results = []
                for i in range(batch_size):
                    img_bytes = self._tensor_to_png(value[i])
                    filename = f"{name}_{i}.png"
                    output_info = OutputInfo(
                        type="IMAGE",
                        filename=filename,
                        mime_type="image/png",
                        file_size_bytes=len(img_bytes),
                        width=value.shape[2],
                        height=value.shape[1],
                        channels=value.shape[3] if len(value.shape) > 3 else 3,
                        dtype="float32",
                        format="png",
                        batch_index=i,
                        batch_size=batch_size,
                    )
                    results.append((filename, img_bytes, "image/png", output_info))
                return results
            else:
                # Send only first image (original behavior)
                data = self._tensor_to_png(value[0])
                filename = f"{name}.png"

                output_info = OutputInfo(
                    type="IMAGE",
                    filename=filename,
                    mime_type="image/png",
                    file_size_bytes=len(data),
                    width=value.shape[2],
                    height=value.shape[1],
                    channels=value.shape[3] if len(value.shape) > 3 else 3,
                    dtype="float32",
                    format="png",
                    batch_index=0,
                    batch_size=batch_size,
                )
                return (filename, data, "image/png", output_info)

        # MASK tensor [B, H, W] or [H, W] - check before IMAGE to avoid confusion
        # Masks are 2D or 3D with small batch dimension (no color channel)
        if torch.is_tensor(value) and len(value.shape) in (2, 3):
            # Check if this looks like a mask:
            # - 2D tensor [H, W] is definitely a mask
            # - 3D tensor [B, H, W] where B < 10 is a batched mask
            # - 3D tensor [H, W, C] where C is 3 or 4 is an image
            is_mask = False
            is_batched_mask = False
            batch_size = 1

            if len(value.shape) == 2:
                is_mask = True
            elif len(value.shape) == 3:
                # [B, H, W] mask vs [H, W, C] image
                # If last dim is 1, 3, or 4, it's likely an image
                # If first dim is small and last dim is large, it's likely a mask batch
                if value.shape[-1] not in (1, 3, 4) and value.shape[0] < 10:
                    is_mask = True
                    is_batched_mask = value.shape[0] > 1
                    batch_size = value.shape[0]
                elif value.shape[0] == 1 and value.shape[-1] > 10:
                    # [1, H, W] is a batched mask
                    is_mask = True
                    is_batched_mask = False
                    batch_size = 1

            if is_mask:
                if batch_outputs and is_batched_mask and batch_size > 1:
                    # Return all masks as separate items
                    results = []
                    for i in range(batch_size):
                        mask = value[i]
                        mask_bytes = self._mask_to_png(mask)
                        filename = f"{name}_{i}.png"
                        output_info = OutputInfo(
                            type="MASK",
                            filename=filename,
                            mime_type="image/png",
                            file_size_bytes=len(mask_bytes),
                            width=mask.shape[1],
                            height=mask.shape[0],
                            format="png",
                            batch_index=i,
                            batch_size=batch_size,
                        )
                        results.append((filename, mask_bytes, "image/png", output_info))
                    return results
                else:
                    # Send only first mask (original behavior)
                    mask = value[0] if len(value.shape) == 3 else value
                    data = self._mask_to_png(mask)
                    filename = f"{name}.png"

                    output_info = OutputInfo(
                        type="MASK",
                        filename=filename,
                        mime_type="image/png",
                        file_size_bytes=len(data),
                        width=mask.shape[1],
                        height=mask.shape[0],
                        format="png",
                        batch_index=0,
                        batch_size=batch_size,
                    )
                    return (filename, data, "image/png", output_info)

        # IMAGE tensor (single) [H, W, C]
        if torch.is_tensor(value) and len(value.shape) == 3:
            data = self._tensor_to_png(value)
            filename = f"{name}.png"

            output_info = OutputInfo(
                type="IMAGE",
                filename=filename,
                mime_type="image/png",
                file_size_bytes=len(data),
                width=value.shape[1],
                height=value.shape[0],
                channels=value.shape[2] if len(value.shape) > 2 else 3,
                dtype="float32",
                format="png",
            )
            return (filename, data, "image/png", output_info)

        # AUDIO dict
        if isinstance(value, dict) and "waveform" in value:
            data = self._audio_to_flac(value)
            if data:
                filename = f"{name}.flac"
                output_info = OutputInfo(
                    type="AUDIO",
                    filename=filename,
                    mime_type="audio/flac",
                    file_size_bytes=len(data),
                    sample_rate=value.get("sample_rate", 44100),
                    format="flac",
                )
                return (filename, data, "audio/flac", output_info)

        # VIDEO - ComfyUI VideoFromComponents (new API with get_components method)
        if hasattr(value, "get_components") and callable(value.get_components):
            try:
                components = value.get_components()
                images = getattr(components, "images", None)
                frame_rate = getattr(components, "frame_rate", None)
                audio = getattr(components, "audio", None)

                if images is not None:
                    fps = float(frame_rate) if frame_rate is not None else 24.0
                    video_dict = {"images": images, "fps": fps}
                    if audio is not None:
                        video_dict["audio"] = audio

                    data = self._video_to_mp4(video_dict)
                    if data:
                        filename = f"{name}.mp4"
                        output_info = OutputInfo(
                            type="VIDEO",
                            filename=filename,
                            mime_type="video/mp4",
                            file_size_bytes=len(data),
                            width=images.shape[2] if torch.is_tensor(images) else None,
                            height=images.shape[1] if torch.is_tensor(images) else None,
                            format="mp4",
                        )
                        return (filename, data, "video/mp4", output_info)
                    else:
                        logger.warning(f"WebhookSend: Failed to encode video for field '{name}'")
            except Exception as e:
                logger.error(f"WebhookSend: Error processing VideoFromComponents: {e}")

        # VIDEO - Legacy video objects with direct images/fps attributes
        if hasattr(value, "images") and hasattr(value, "fps"):
            video_dict = {
                "images": value.images,
                "fps": value.fps if isinstance(value.fps, (int, float)) else 24,
            }
            if hasattr(value, "audio") and value.audio is not None:
                video_dict["audio"] = value.audio
            data = self._video_to_mp4(video_dict)
            if data:
                filename = f"{name}.mp4"
                fps = video_dict["fps"]
                images = video_dict["images"]
                output_info = OutputInfo(
                    type="VIDEO",
                    filename=filename,
                    mime_type="video/mp4",
                    file_size_bytes=len(data),
                    width=images.shape[2] if torch.is_tensor(images) else None,
                    height=images.shape[1] if torch.is_tensor(images) else None,
                    format="mp4",
                )
                return (filename, data, "video/mp4", output_info)

        # VIDEO dict or tensor
        if isinstance(value, dict) and "images" in value:
            data = self._video_to_mp4(value)
            if data:
                filename = f"{name}.mp4"
                fps = value.get("fps", 24)
                images = value.get("images")
                output_info = OutputInfo(
                    type="VIDEO",
                    filename=filename,
                    mime_type="video/mp4",
                    file_size_bytes=len(data),
                    width=images.shape[2] if torch.is_tensor(images) else None,
                    height=images.shape[1] if torch.is_tensor(images) else None,
                    format="mp4",
                )
                return (filename, data, "video/mp4", output_info)

        # MESH dict
        if isinstance(value, dict) and ("vertices" in value or "faces" in value):
            data = self._mesh_to_glb(value)
            if data:
                filename = f"{name}.glb"
                vertices = value.get("vertices")
                faces = value.get("faces")
                output_info = OutputInfo(
                    type="MESH",
                    filename=filename,
                    mime_type="model/gltf-binary",
                    file_size_bytes=len(data),
                    format="glb",
                )
                return (filename, data, "model/gltf-binary", output_info)

        # LATENT dict
        if isinstance(value, dict) and "samples" in value:
            data = self._latent_to_safetensors(value)
            if data:
                filename = f"{name}.safetensors"
                samples = value.get("samples")
                output_info = OutputInfo(
                    type="LATENT",
                    filename=filename,
                    mime_type="application/octet-stream",
                    file_size_bytes=len(data),
                    format="safetensors",
                )
                return (filename, data, "application/octet-stream", output_info)

        # STRING
        if isinstance(value, str):
            data = value.encode("utf-8")
            filename = f"{name}.txt"
            output_info = OutputInfo(
                type="STRING",
                filename=filename,
                mime_type="text/plain",
                file_size_bytes=len(data),
                inline_content=value if len(value) < 1000 else None,
            )
            return (filename, data, "text/plain", output_info)

        # INT, FLOAT, BOOLEAN
        if isinstance(value, (int, float, bool)):
            str_value = str(value).lower() if isinstance(value, bool) else str(value)
            data = str_value.encode("utf-8")
            filename = f"{name}.txt"
            type_name = type(value).__name__.upper()
            output_info = OutputInfo(
                type=type_name,
                filename=filename,
                mime_type="text/plain",
                file_size_bytes=len(data),
                inline_content=str_value,
            )
            return (filename, data, "text/plain", output_info)

        # BYTES
        if isinstance(value, bytes):
            filename = f"{name}.bin"
            output_info = OutputInfo(
                type="BYTES",
                filename=filename,
                mime_type="application/octet-stream",
                file_size_bytes=len(value),
            )
            return (filename, value, "application/octet-stream", output_info)

        # Try JSON serialization for other types
        try:
            json_str = json.dumps(value, default=str)
            data = json_str.encode("utf-8")
            filename = f"{name}.json"
            output_info = OutputInfo(
                type="JSON",
                filename=filename,
                mime_type="application/json",
                file_size_bytes=len(data),
            )
            return (filename, data, "application/json", output_info)
        except Exception:
            pass

        logger.warning(f"WebhookSend: Unable to encode field '{name}' of type {type(value)}")
        return None

    def _tensor_to_png(self, tensor: torch.Tensor) -> bytes:
        """Convert image tensor [H, W, C] to PNG bytes."""
        np_image = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(np_image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG", compress_level=4)
        return buffer.getvalue()

    def _mask_to_png(self, tensor: torch.Tensor) -> bytes:
        """Convert mask tensor [H, W] to grayscale PNG bytes."""
        np_mask = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(np_mask, mode='L')

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _audio_to_flac(self, audio: dict) -> Optional[bytes]:
        """Convert audio dict to FLAC bytes."""
        try:
            import av
        except ImportError:
            logger.warning("av library not available for audio encoding")
            return None

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 44100)

        if waveform is None:
            return None

        if len(waveform.shape) == 3:
            waveform = waveform[0]

        try:
            buffer = io.BytesIO()
            container = av.open(buffer, mode="w", format="flac")

            layout = "mono" if waveform.shape[0] == 1 else "stereo"
            stream = container.add_stream("flac", rate=sample_rate, layout=layout)

            audio_np = waveform.cpu().numpy()

            frame = av.AudioFrame.from_ndarray(
                audio_np.reshape(1, -1).astype(np.float32),
                format="flt",
                layout=layout,
            )
            frame.sample_rate = sample_rate

            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

            container.close()
            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Failed to encode audio: {e}")
            return None

    def _video_to_mp4(self, video: dict) -> Optional[bytes]:
        """Convert video dict to MP4 bytes."""
        try:
            import av
        except ImportError:
            logger.warning("av library not available for video encoding")
            return None

        images = video.get("images")
        fps = video.get("fps", 24)

        if images is None:
            return None

        try:
            from fractions import Fraction
            buffer = io.BytesIO()
            container = av.open(buffer, mode="w", format="mp4")
            fps_fraction = Fraction(fps).limit_denominator(1000)
            stream = container.add_stream("h264", rate=fps_fraction)

            if len(images.shape) == 4:
                height, width = images.shape[1], images.shape[2]
            else:
                height, width = images.shape[0], images.shape[1]

            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"

            if len(images.shape) == 4:
                for i in range(images.shape[0]):
                    np_frame = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    frame = av.VideoFrame.from_ndarray(np_frame, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)

            for packet in stream.encode(None):
                container.mux(packet)

            container.close()
            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Failed to encode video: {e}")
            return None

    def _mesh_to_glb(self, mesh: dict) -> Optional[bytes]:
        """Convert mesh dict to GLB bytes."""
        try:
            vertices = mesh.get("vertices")
            faces = mesh.get("faces")

            if vertices is None or faces is None:
                return None

            if torch.is_tensor(vertices):
                vertices = vertices.cpu().numpy()
            if torch.is_tensor(faces):
                faces = faces.cpu().numpy()

            # Handle batch dimension
            if len(vertices.shape) == 3:
                vertices = vertices[0]
            if len(faces.shape) == 3:
                faces = faces[0]

            vertices = vertices.astype(np.float32)
            faces = faces.astype(np.uint32)

            # Calculate bounding box
            v_min = vertices.min(axis=0).tolist()
            v_max = vertices.max(axis=0).tolist()

            # Create buffer data
            vertex_data = vertices.tobytes()
            index_data = faces.flatten().tobytes()

            def pad4(data):
                remainder = len(data) % 4
                if remainder:
                    data += b'\x00' * (4 - remainder)
                return data

            vertex_data = pad4(vertex_data)
            index_data = pad4(index_data)
            buffer_data = vertex_data + index_data

            # Create glTF JSON
            gltf = {
                "asset": {"version": "2.0", "generator": "ComfyUI-Webhook"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}],
                "meshes": [{
                    "primitives": [{
                        "attributes": {"POSITION": 0},
                        "indices": 1
                    }]
                }],
                "accessors": [
                    {
                        "bufferView": 0,
                        "componentType": 5126,
                        "count": len(vertices),
                        "type": "VEC3",
                        "min": v_min,
                        "max": v_max
                    },
                    {
                        "bufferView": 1,
                        "componentType": 5125,
                        "count": faces.size,
                        "type": "SCALAR"
                    }
                ],
                "bufferViews": [
                    {
                        "buffer": 0,
                        "byteOffset": 0,
                        "byteLength": len(vertex_data)
                    },
                    {
                        "buffer": 0,
                        "byteOffset": len(vertex_data),
                        "byteLength": len(index_data)
                    }
                ],
                "buffers": [{"byteLength": len(buffer_data)}]
            }

            json_data = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
            json_data = pad4(json_data)

            # Build GLB
            glb = io.BytesIO()
            glb.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(json_data) + 8 + len(buffer_data)))
            glb.write(struct.pack('<II', len(json_data), 0x4E4F534A))
            glb.write(json_data)
            glb.write(struct.pack('<II', len(buffer_data), 0x004E4942))
            glb.write(buffer_data)

            return glb.getvalue()

        except Exception as e:
            logger.error(f"Failed to encode mesh: {e}")
            return None

    def _latent_to_safetensors(self, latent: dict) -> Optional[bytes]:
        try:
            from safetensors.torch import save
            samples = latent.get("samples")
            if samples is None:
                return None
            return save({"samples": samples})
        except Exception as e:
            logger.error(f"Failed to encode latent: {e}")
            return None

    def _print_debug(
        self,
        context: WebhookContext,
        files: List[Tuple[str, str, bytes, str]],
        outputs: List[Dict],
        result: WebhookResult,
        elapsed_ms: float
    ):
        """Print debug information about the outgoing request."""
        lines = [
            "",
            "[Webhook Send] Outgoing Request",
            "=" * 50,
            f"  Target URL:     {context.callback_url}",
            f"  Method:         POST",
            f"  Content-Type:   multipart/form-data",
            f"  Auth Header:    {context.censor_auth()}",
            f"  Request ID:     {context.request_id}",
            "",
            f"  Fields ({len(files)}):",
        ]

        for i, (name, filename, data, mime_type) in enumerate(files):
            prefix = "    +-" if i == len(files) - 1 else "    |-"

            # Find output info for additional details
            output = outputs[i] if i < len(outputs) else {}
            type_name = output.get("type", "UNKNOWN")

            if type_name == "IMAGE":
                w = output.get("width", "?")
                h = output.get("height", "?")
                lines.append(f"{prefix} {name:20} ({type_name}) {w}x{h}, PNG, {format_size(len(data))}")
            elif type_name == "MASK":
                w = output.get("width", "?")
                h = output.get("height", "?")
                lines.append(f"{prefix} {name:20} ({type_name}) {w}x{h}, PNG, {format_size(len(data))}")
            elif type_name == "STRING":
                content = output.get("inline_content")
                if content and len(content) < 30:
                    lines.append(f"{prefix} {name:20} ({type_name}) \"{content}\"")
                else:
                    lines.append(f"{prefix} {name:20} ({type_name}) {format_size(len(data))}")
            else:
                lines.append(f"{prefix} {name:20} ({type_name}) {format_size(len(data))}")

        lines.append("")
        lines.append("  Response:")
        if result.success:
            lines.append(f"    Status:   {result.status} OK")
        else:
            lines.append(f"    Status:   {result.status or 'N/A'} {result.error or 'Failed'}")

        lines.append(f"    Time:     {elapsed_ms:.0f} ms")

        if result.response_body:
            body = result.response_body
            if len(body) > 100:
                body = body[:97] + "..."
            lines.append(f"    Body:     {body}")

        if result.retries > 0:
            lines.append(f"    Retries:  {result.retries}")

        lines.append("=" * 50)
        print("\n".join(lines))

    def _print_error(self, context: WebhookContext, error: str):
        """Print error information."""
        lines = [
            "",
            "[Webhook Send] Request Failed",
            "=" * 50,
            f"  Target URL:     {context.callback_url}",
            f"  Request ID:     {context.request_id}",
            "",
            f"  Error:          {error}",
            "=" * 50,
        ]
        print("\n".join(lines))
