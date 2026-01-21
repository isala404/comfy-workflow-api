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
from ..utils.helpers import format_size, format_count, run_async

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
                self._send_async(webhook_context, fields_to_send, debug)
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
    ):
        """Async implementation of send."""
        files = []
        outputs = []
        start_time = datetime.now()

        # Process each field
        for name, value in fields_to_send:
            encoded = self._encode_value(name, value)
            if encoded:
                # _encode_value always returns a list of (filename, data, mime_type, output_info)
                for filename, data, mime_type, output_info in encoded:
                    # Extract field name from filename (e.g., "output_0.png" -> "output_0")
                    field_name = filename.rsplit(".", 1)[0]
                    files.append((field_name, filename, data, mime_type))
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
    ) -> Optional[List[Tuple[str, bytes, str, OutputInfo]]]:
        """
        Encode value to appropriate format for multipart upload.

        Batched tensors (batch_size >= 1) automatically expand to field_0, field_1, etc.

        Returns:
            List of (filename, data_bytes, mime_type, output_info) tuples, or None if encoding fails
        """
        # IMAGE tensor [B, H, W, C]
        if torch.is_tensor(value) and len(value.shape) == 4:
            batch_size = value.shape[0]
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

        # MASK tensor [H, W] (single, unbatched)
        if torch.is_tensor(value) and len(value.shape) == 2:
            mask_bytes = self._mask_to_png(value)
            filename = f"{name}_0.png"
            output_info = OutputInfo(
                type="MASK",
                filename=filename,
                mime_type="image/png",
                file_size_bytes=len(mask_bytes),
                width=value.shape[1],
                height=value.shape[0],
                format="png",
                batch_index=0,
                batch_size=1,
            )
            return [(filename, mask_bytes, "image/png", output_info)]

        # MASK tensor [B, H, W] (batched) - distinguish from [H, W, C] image
        if torch.is_tensor(value) and len(value.shape) == 3:
            # [B, H, W] mask: last dim is NOT a color channel (1, 3, 4) and first dim is small
            # [H, W, C] image: last dim IS a color channel
            is_batched_mask = value.shape[-1] not in (1, 3, 4) and value.shape[0] < 10
            # Also handle [1, H, W] as a single batched mask
            if not is_batched_mask and value.shape[0] == 1 and value.shape[-1] > 10:
                is_batched_mask = True

            if is_batched_mask:
                batch_size = value.shape[0]
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

        # IMAGE tensor (single) [H, W, C]
        if torch.is_tensor(value) and len(value.shape) == 3:
            # Check for NORMAL_MAP first (3 channels, has negative values)
            if value.shape[-1] == 3:
                v_min = value.min().item()
                if v_min < 0:
                    return self._normal_map_to_result(name, value.unsqueeze(0))

            data = self._tensor_to_png(value)
            filename = f"{name}_0.png"
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
                batch_index=0,
                batch_size=1,
            )
            return [(filename, data, "image/png", output_info)]

        # AUDIO dict
        if isinstance(value, dict) and "waveform" in value:
            data = self._audio_to_flac(value)
            if data:
                filename = f"{name}_0.flac"
                output_info = OutputInfo(
                    type="AUDIO",
                    filename=filename,
                    mime_type="audio/flac",
                    file_size_bytes=len(data),
                    sample_rate=value.get("sample_rate", 44100),
                    format="flac",
                    batch_index=0,
                    batch_size=1,
                )
                return [(filename, data, "audio/flac", output_info)]

        # GAUSSIAN_SPLATTING dict (has positions + opacity/scales/sh_coefficients)
        if isinstance(value, dict) and "positions" in value:
            if "opacity" in value or "scales" in value or "sh_coefficients" in value:
                return self._gaussian_to_result(name, value)

        # POINT_CLOUD dict (has points but no gaussian properties)
        if isinstance(value, dict) and "points" in value and "opacity" not in value:
            return self._point_cloud_to_result(name, value)

        # CAMERA_POSES dict (has poses or cameras)
        if isinstance(value, dict) and ("poses" in value or "cameras" in value):
            return self._camera_poses_to_result(name, value)

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
                        filename = f"{name}_0.mp4"
                        output_info = OutputInfo(
                            type="VIDEO",
                            filename=filename,
                            mime_type="video/mp4",
                            file_size_bytes=len(data),
                            width=images.shape[2] if torch.is_tensor(images) else None,
                            height=images.shape[1] if torch.is_tensor(images) else None,
                            format="mp4",
                            batch_index=0,
                            batch_size=1,
                        )
                        return [(filename, data, "video/mp4", output_info)]
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
                filename = f"{name}_0.mp4"
                images = video_dict["images"]
                output_info = OutputInfo(
                    type="VIDEO",
                    filename=filename,
                    mime_type="video/mp4",
                    file_size_bytes=len(data),
                    width=images.shape[2] if torch.is_tensor(images) else None,
                    height=images.shape[1] if torch.is_tensor(images) else None,
                    format="mp4",
                    batch_index=0,
                    batch_size=1,
                )
                return [(filename, data, "video/mp4", output_info)]

        # VIDEO dict or tensor
        if isinstance(value, dict) and "images" in value:
            data = self._video_to_mp4(value)
            if data:
                filename = f"{name}_0.mp4"
                images = value.get("images")
                output_info = OutputInfo(
                    type="VIDEO",
                    filename=filename,
                    mime_type="video/mp4",
                    file_size_bytes=len(data),
                    width=images.shape[2] if torch.is_tensor(images) else None,
                    height=images.shape[1] if torch.is_tensor(images) else None,
                    format="mp4",
                    batch_index=0,
                    batch_size=1,
                )
                return [(filename, data, "video/mp4", output_info)]

        # MESH dict
        if isinstance(value, dict) and ("vertices" in value or "faces" in value):
            data = self._mesh_to_glb(value)
            if data:
                filename = f"{name}_0.glb"
                output_info = OutputInfo(
                    type="MESH",
                    filename=filename,
                    mime_type="model/gltf-binary",
                    file_size_bytes=len(data),
                    format="glb",
                    batch_index=0,
                    batch_size=1,
                )
                return [(filename, data, "model/gltf-binary", output_info)]

        # LATENT dict
        if isinstance(value, dict) and "samples" in value:
            data = self._latent_to_safetensors(value)
            if data:
                filename = f"{name}_0.safetensors"
                output_info = OutputInfo(
                    type="LATENT",
                    filename=filename,
                    mime_type="application/octet-stream",
                    file_size_bytes=len(data),
                    format="safetensors",
                    batch_index=0,
                    batch_size=1,
                )
                return [(filename, data, "application/octet-stream", output_info)]

        # STRING
        if isinstance(value, str):
            data = value.encode("utf-8")
            filename = f"{name}_0.txt"
            output_info = OutputInfo(
                type="STRING",
                filename=filename,
                mime_type="text/plain",
                file_size_bytes=len(data),
                inline_content=value if len(value) < 1000 else None,
                batch_index=0,
                batch_size=1,
            )
            return [(filename, data, "text/plain", output_info)]

        # INT, FLOAT, BOOLEAN
        if isinstance(value, (int, float, bool)):
            str_value = str(value).lower() if isinstance(value, bool) else str(value)
            data = str_value.encode("utf-8")
            filename = f"{name}_0.txt"
            type_name = type(value).__name__.upper()
            output_info = OutputInfo(
                type=type_name,
                filename=filename,
                mime_type="text/plain",
                file_size_bytes=len(data),
                inline_content=str_value,
                batch_index=0,
                batch_size=1,
            )
            return [(filename, data, "text/plain", output_info)]

        # BYTES
        if isinstance(value, bytes):
            filename = f"{name}_0.bin"
            output_info = OutputInfo(
                type="BYTES",
                filename=filename,
                mime_type="application/octet-stream",
                file_size_bytes=len(value),
                batch_index=0,
                batch_size=1,
            )
            return [(filename, value, "application/octet-stream", output_info)]

        # Try JSON serialization for other types
        try:
            json_str = json.dumps(value, default=str)
            data = json_str.encode("utf-8")
            filename = f"{name}_0.json"
            output_info = OutputInfo(
                type="JSON",
                filename=filename,
                mime_type="application/json",
                file_size_bytes=len(data),
                batch_index=0,
                batch_size=1,
            )
            return [(filename, data, "application/json", output_info)]
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

    def _encode_ply_binary(
        self,
        positions: np.ndarray,
        properties: List[Tuple[str, str, np.ndarray]],
    ) -> bytes:
        """Encode point data to binary PLY format."""
        point_count = len(positions)

        # Build header
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {point_count}",
            "property float x",
            "property float y",
            "property float z",
        ]

        for prop_name, prop_type, _ in properties:
            header_lines.append(f"property {prop_type} {prop_name}")

        header_lines.append("end_header\n")
        header = "\n".join(header_lines).encode("ascii")

        # Build binary data
        buffer = io.BytesIO()
        buffer.write(header)

        positions = positions.astype(np.float32)
        for i in range(point_count):
            buffer.write(positions[i].tobytes())
            for _, _, data in properties:
                buffer.write(data[i].tobytes())

        return buffer.getvalue()

    def _gaussian_to_result(
        self,
        name: str,
        value: dict,
    ) -> List[Tuple[str, bytes, str, OutputInfo]]:
        """Convert Gaussian Splatting dict to PLY result."""
        try:
            positions = value.get("positions")
            if positions is None:
                return []

            if torch.is_tensor(positions):
                positions = positions.cpu().numpy()

            if len(positions.shape) == 3:
                positions = positions[0]

            point_count = len(positions)
            properties = []

            # Collect optional properties
            if "scales" in value:
                scales = value["scales"]
                if torch.is_tensor(scales):
                    scales = scales.cpu().numpy()
                if len(scales.shape) == 3:
                    scales = scales[0]
                for i in range(scales.shape[-1]):
                    properties.append((f"scale_{i}", "float", scales[:, i].astype(np.float32)))

            if "rotations" in value:
                rotations = value["rotations"]
                if torch.is_tensor(rotations):
                    rotations = rotations.cpu().numpy()
                if len(rotations.shape) == 3:
                    rotations = rotations[0]
                for i in range(rotations.shape[-1]):
                    properties.append((f"rot_{i}", "float", rotations[:, i].astype(np.float32)))

            if "opacity" in value:
                opacity = value["opacity"]
                if torch.is_tensor(opacity):
                    opacity = opacity.cpu().numpy()
                if len(opacity.shape) == 2:
                    opacity = opacity[0]
                opacity = opacity.flatten().astype(np.float32)
                properties.append(("opacity", "float", opacity))

            has_sh = "sh_coefficients" in value
            if has_sh:
                sh = value["sh_coefficients"]
                if torch.is_tensor(sh):
                    sh = sh.cpu().numpy()
                if len(sh.shape) == 4:
                    sh = sh[0]
                # Flatten SH coefficients
                sh_flat = sh.reshape(point_count, -1)
                for i in range(sh_flat.shape[-1]):
                    properties.append((f"f_dc_{i}" if i < 3 else f"f_rest_{i-3}", "float", sh_flat[:, i].astype(np.float32)))

            data = self._encode_ply_binary(positions, properties)
            filename = f"{name}_0.ply"

            output_info = OutputInfo(
                type="GAUSSIAN_SPLATTING",
                filename=filename,
                mime_type="application/x-ply",
                file_size_bytes=len(data),
                format="ply",
                point_count=point_count,
                has_sh_coefficients=has_sh,
                batch_index=0,
                batch_size=1,
            )
            return [(filename, data, "application/x-ply", output_info)]

        except Exception as e:
            logger.error(f"Failed to encode gaussian splatting: {e}")
            return []

    def _point_cloud_to_result(
        self,
        name: str,
        value: dict,
    ) -> List[Tuple[str, bytes, str, OutputInfo]]:
        """Convert Point Cloud dict to PLY result."""
        try:
            points = value.get("points")
            if points is None:
                return []

            if torch.is_tensor(points):
                points = points.cpu().numpy()

            if len(points.shape) == 3:
                points = points[0]

            point_count = len(points)
            properties = []

            # Colors
            if "colors" in value:
                colors = value["colors"]
                if torch.is_tensor(colors):
                    colors = colors.cpu().numpy()
                if len(colors.shape) == 3:
                    colors = colors[0]
                # Convert to uint8 if needed
                if colors.max() <= 1.0:
                    colors = (colors * 255).astype(np.uint8)
                else:
                    colors = colors.astype(np.uint8)
                properties.append(("red", "uchar", colors[:, 0]))
                properties.append(("green", "uchar", colors[:, 1]))
                properties.append(("blue", "uchar", colors[:, 2]))

            # Normals
            if "normals" in value:
                normals = value["normals"]
                if torch.is_tensor(normals):
                    normals = normals.cpu().numpy()
                if len(normals.shape) == 3:
                    normals = normals[0]
                properties.append(("nx", "float", normals[:, 0].astype(np.float32)))
                properties.append(("ny", "float", normals[:, 1].astype(np.float32)))
                properties.append(("nz", "float", normals[:, 2].astype(np.float32)))

            data = self._encode_ply_binary(points, properties)
            filename = f"{name}_0.ply"

            output_info = OutputInfo(
                type="POINT_CLOUD",
                filename=filename,
                mime_type="application/x-ply",
                file_size_bytes=len(data),
                format="ply",
                point_count=point_count,
                batch_index=0,
                batch_size=1,
            )
            return [(filename, data, "application/x-ply", output_info)]

        except Exception as e:
            logger.error(f"Failed to encode point cloud: {e}")
            return []

    def _depth_map_to_result(
        self,
        name: str,
        value: torch.Tensor,
    ) -> List[Tuple[str, bytes, str, OutputInfo]]:
        """Convert depth map tensor to EXR or PNG16 result."""
        try:
            if len(value.shape) == 3:
                batch_size = value.shape[0]
            else:
                value = value.unsqueeze(0)
                batch_size = 1

            results = []
            for i in range(batch_size):
                depth = value[i].cpu().numpy()
                depth_min = float(depth.min())
                depth_max = float(depth.max())

                # Try EXR first (preferred for metric depth)
                try:
                    import OpenEXR
                    import Imath
                    import tempfile
                    import os

                    height, width = depth.shape
                    header = OpenEXR.Header(width, height)
                    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
                    # Single Y channel for depth (not RGB)
                    header["channels"] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

                    exr_data = depth.astype(np.float32).tobytes()

                    # OpenEXR requires file path, use temp with proper cleanup
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
                            temp_path = f.name

                        out = OpenEXR.OutputFile(temp_path, header)
                        out.writePixels({"Y": exr_data})
                        out.close()

                        with open(temp_path, "rb") as f:
                            data = f.read()
                    finally:
                        if temp_path and os.path.exists(temp_path):
                            os.unlink(temp_path)

                    filename = f"{name}_{i}.exr"
                    mime_type = "image/x-exr"
                    fmt = "exr"

                except ImportError:
                    # Fallback to 16-bit PNG
                    depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
                    depth_16bit = (depth_normalized * 65535).astype(np.uint16)
                    pil_image = Image.fromarray(depth_16bit, mode="I;16")

                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    data = buffer.getvalue()

                    filename = f"{name}_{i}.png"
                    mime_type = "image/png"
                    fmt = "png16"

                output_info = OutputInfo(
                    type="DEPTH_MAP",
                    filename=filename,
                    mime_type=mime_type,
                    file_size_bytes=len(data),
                    format=fmt,
                    width=depth.shape[1],
                    height=depth.shape[0],
                    depth_min=depth_min,
                    depth_max=depth_max,
                    batch_index=i,
                    batch_size=batch_size,
                )
                results.append((filename, data, mime_type, output_info))

            return results

        except Exception as e:
            logger.error(f"Failed to encode depth map: {e}")
            return []

    def _normal_map_to_result(
        self,
        name: str,
        value: torch.Tensor,
    ) -> List[Tuple[str, bytes, str, OutputInfo]]:
        """Convert normal map tensor [B, H, W, 3] to PNG result."""
        try:
            if len(value.shape) == 3:
                value = value.unsqueeze(0)

            batch_size = value.shape[0]
            results = []

            for i in range(batch_size):
                normals = value[i].cpu().numpy()

                # Convert from [-1, 1] to [0, 255]
                normals_uint8 = ((normals + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
                pil_image = Image.fromarray(normals_uint8)

                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                data = buffer.getvalue()

                filename = f"{name}_{i}.png"
                output_info = OutputInfo(
                    type="NORMAL_MAP",
                    filename=filename,
                    mime_type="image/png",
                    file_size_bytes=len(data),
                    format="png",
                    width=normals.shape[1],
                    height=normals.shape[0],
                    channels=3,
                    batch_index=i,
                    batch_size=batch_size,
                )
                results.append((filename, data, "image/png", output_info))

            return results

        except Exception as e:
            logger.error(f"Failed to encode normal map: {e}")
            return []

    def _camera_poses_to_result(
        self,
        name: str,
        value: dict,
    ) -> List[Tuple[str, bytes, str, OutputInfo]]:
        """Convert camera poses dict to JSON result."""
        try:
            poses = value.get("poses") or value.get("cameras")
            if poses is None:
                return []

            if torch.is_tensor(poses):
                poses = poses.cpu().numpy().tolist()
            elif isinstance(poses, np.ndarray):
                poses = poses.tolist()

            camera_count = len(poses) if isinstance(poses, list) else 1

            output_dict = {"poses": poses}

            # Include intrinsics if present
            has_intrinsics = False
            if "intrinsics" in value:
                intrinsics = value["intrinsics"]
                if torch.is_tensor(intrinsics):
                    intrinsics = intrinsics.cpu().numpy().tolist()
                elif isinstance(intrinsics, np.ndarray):
                    intrinsics = intrinsics.tolist()
                output_dict["intrinsics"] = intrinsics
                has_intrinsics = True

            json_str = json.dumps(output_dict)
            data = json_str.encode("utf-8")
            filename = f"{name}_0.json"

            output_info = OutputInfo(
                type="CAMERA_POSES",
                filename=filename,
                mime_type="application/json",
                file_size_bytes=len(data),
                format="json",
                camera_count=camera_count,
                has_intrinsics=has_intrinsics,
                batch_index=0,
                batch_size=1,
            )
            return [(filename, data, "application/json", output_info)]

        except Exception as e:
            logger.error(f"Failed to encode camera poses: {e}")
            return []

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
            elif type_name == "NORMAL_MAP":
                w = output.get("width", "?")
                h = output.get("height", "?")
                lines.append(f"{prefix} {name:20} ({type_name}) {w}x{h}, PNG, {format_size(len(data))}")
            elif type_name == "DEPTH_MAP":
                w = output.get("width", "?")
                h = output.get("height", "?")
                fmt = output.get("format", "?")
                d_min = output.get("depth_min", "?")
                d_max = output.get("depth_max", "?")
                lines.append(f"{prefix} {name:20} ({type_name}) {w}x{h}, {fmt}, range=[{d_min:.2f}, {d_max:.2f}], {format_size(len(data))}")
            elif type_name == "GAUSSIAN_SPLATTING":
                pts = format_count(output.get("point_count", 0))
                has_sh = "SH" if output.get("has_sh_coefficients") else "no-SH"
                lines.append(f"{prefix} {name:20} ({type_name}) {pts} points, {has_sh}, PLY, {format_size(len(data))}")
            elif type_name == "POINT_CLOUD":
                pts = format_count(output.get("point_count", 0))
                lines.append(f"{prefix} {name:20} ({type_name}) {pts} points, PLY, {format_size(len(data))}")
            elif type_name == "CAMERA_POSES":
                cams = output.get("camera_count", "?")
                has_k = "K" if output.get("has_intrinsics") else "no-K"
                lines.append(f"{prefix} {name:20} ({type_name}) {cams} cameras, {has_k}, JSON, {format_size(len(data))}")
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
