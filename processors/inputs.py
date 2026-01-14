"""Input processing for webhook data."""

import asyncio
import base64
import io
import json
import logging
import re
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageOps

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

logger = logging.getLogger(__name__)

DATA_URL_PATTERN = re.compile(r'^data:([^;,]+)?(;base64)?,(.*)$', re.DOTALL)
HTTP_URL_PATTERN = re.compile(r'^https?://', re.IGNORECASE)


class WebhookInputProcessor:
    """Processes webhook input data into ComfyUI-compatible formats."""

    def __init__(self, download_timeout: float = 30.0):
        self.download_timeout = download_timeout

    async def process_inputs(self, webhook_inputs: dict) -> dict:
        """
        Process all inputs and return ComfyUI-compatible values.

        Args:
            webhook_inputs: Dictionary of input name -> value

        Returns:
            Dictionary of input name -> processed value
        """
        processed = {}

        for key, value in webhook_inputs.items():
            try:
                processed[key] = await self._process_value(value)
            except Exception as e:
                logger.error(f"Failed to process input '{key}': {e}")
                # Keep original value on error
                processed[key] = value

        return processed

    async def _process_value(self, value: Any) -> Any:
        """Process a single value based on its type."""
        if isinstance(value, str):
            return await self._process_string(value)
        elif isinstance(value, list):
            return await self._process_array(value)
        elif isinstance(value, dict):
            # JSON data - serialize to string
            return json.dumps(value)
        else:
            # Numbers, booleans - pass through
            return value

    async def _process_string(self, value: str) -> Any:
        """Process a string value, detecting if it's base64/URL data."""
        if self._is_base64_image(value):
            return await self._decode_base64_image(value)
        elif self._is_base64_audio(value):
            return await self._decode_base64_audio(value)
        elif self._is_url(value):
            return await self._download_and_decode(value)
        else:
            # Plain string
            return value

    async def _process_array(self, items: list) -> Union[torch.Tensor, list]:
        """
        Process array of items.

        If all items are images, returns a batched tensor [N, H, W, C].
        Otherwise returns a list of processed items.
        """
        if not items:
            return []

        # Check if all items are image data
        if all(self._is_image_data(item) for item in items):
            tensors = []
            for item in items:
                if isinstance(item, str):
                    if self._is_base64_image(item):
                        tensor = await self._decode_base64_image(item)
                    elif self._is_url(item):
                        tensor = await self._download_and_decode(item)
                    else:
                        continue
                    tensors.append(tensor)

            if tensors:
                # Stack into batch tensor [N, H, W, C]
                return self._batch_images(tensors)
            return []

        # Otherwise process each item individually
        return [await self._process_value(item) for item in items]

    def _batch_images(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Batch multiple image tensors, handling different sizes.

        Args:
            tensors: List of [1, H, W, C] tensors

        Returns:
            Batched tensor [N, H, W, C]
        """
        if not tensors:
            return torch.zeros(0, 64, 64, 3)

        # Check if all images have the same size
        shapes = [t.shape[1:3] for t in tensors]
        if all(s == shapes[0] for s in shapes):
            # All same size, simple cat
            return torch.cat(tensors, dim=0)

        # Different sizes - pad to largest
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)

        padded = []
        for t in tensors:
            h, w = t.shape[1], t.shape[2]
            if h < max_h or w < max_w:
                # Center pad
                pad_h_before = (max_h - h) // 2
                pad_h_after = max_h - h - pad_h_before
                pad_w_before = (max_w - w) // 2
                pad_w_after = max_w - w - pad_w_before
                # F.pad expects (left, right, top, bottom) for 4D tensor
                t = torch.nn.functional.pad(
                    t.permute(0, 3, 1, 2),  # [B, C, H, W]
                    (pad_w_before, pad_w_after, pad_h_before, pad_h_after),
                    mode='constant',
                    value=0
                ).permute(0, 2, 3, 1)  # Back to [B, H, W, C]
            padded.append(t)

        return torch.cat(padded, dim=0)

    # Detection methods

    def _is_base64_image(self, value: str) -> bool:
        """Check if string is a base64-encoded image."""
        if value.startswith("data:image/"):
            return True
        # Check for raw base64 that might be an image (heuristic)
        if len(value) > 100 and re.match(r'^[A-Za-z0-9+/]+=*$', value):
            # Try to detect PNG/JPEG magic bytes
            try:
                decoded = base64.b64decode(value[:32])
                # PNG magic: 89 50 4E 47
                if decoded.startswith(b'\x89PNG'):
                    return True
                # JPEG magic: FF D8 FF
                if decoded.startswith(b'\xff\xd8\xff'):
                    return True
                # WebP magic: RIFF....WEBP
                if decoded.startswith(b'RIFF') and b'WEBP' in decoded[:16]:
                    return True
            except:
                pass
        return False

    def _is_base64_audio(self, value: str) -> bool:
        """Check if string is a base64-encoded audio."""
        return value.startswith("data:audio/")

    def _is_url(self, value: str) -> bool:
        """Check if string is an HTTP(S) URL."""
        return bool(HTTP_URL_PATTERN.match(value))

    def _is_image_data(self, value: Any) -> bool:
        """Check if value is image data (base64 or URL)."""
        if not isinstance(value, str):
            return False
        return self._is_base64_image(value) or (
            self._is_url(value) and
            any(ext in value.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'])
        )

    # Decoding methods

    async def _decode_base64_image(self, data: str) -> torch.Tensor:
        """
        Decode base64 image to tensor.

        Supports:
        - data:image/png;base64,...
        - data:image/jpeg;base64,...
        - data:image/webp;base64,...
        - Raw base64 (no prefix)

        Returns:
            Tensor [1, H, W, C] with values 0-1
        """
        # Strip data URL prefix if present
        if data.startswith("data:"):
            match = DATA_URL_PATTERN.match(data)
            if match:
                data = match.group(3)
            else:
                _, data = data.split(",", 1)

        # Decode base64
        image_bytes = base64.b64decode(data)

        # Load with PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)

        # Convert to RGB if needed
        if pil_image.mode == "I":
            # 16-bit images
            pil_image = pil_image.point(lambda i: i * (1 / 255))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to tensor [1, H, W, C] with values 0-1
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image).unsqueeze(0)

        return tensor

    async def _decode_base64_audio(self, data: str) -> dict:
        """
        Decode base64 audio to ComfyUI audio format.

        Returns:
            {"waveform": tensor [1, channels, samples], "sample_rate": int}
        """
        if torchaudio is None:
            raise ImportError("torchaudio is required for audio processing")

        # Strip data URL prefix if present
        if data.startswith("data:"):
            _, data = data.split(",", 1)

        audio_bytes = base64.b64decode(data)

        # Use torchaudio to load
        buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(buffer)

        # Resample to 44100 if needed (common ComfyUI sample rate)
        target_rate = 44100
        if sample_rate != target_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
            waveform = resampler(waveform)
            sample_rate = target_rate

        return {
            "waveform": waveform.unsqueeze(0),  # [1, channels, samples]
            "sample_rate": sample_rate
        }

    async def _download_and_decode(self, url: str) -> Any:
        """
        Download content from URL and decode based on content type.

        Returns:
            Decoded content (tensor for images, dict for audio, etc.)
        """
        if aiohttp is None:
            raise ImportError("aiohttp is required for URL downloads")

        timeout = aiohttp.ClientTimeout(total=self.download_timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                data = await response.read()

        # Detect type from content-type header or URL
        if "image" in content_type or self._looks_like_image_url(url):
            return await self._decode_image_bytes(data)
        elif "audio" in content_type or self._looks_like_audio_url(url):
            return await self._decode_audio_bytes(data)
        else:
            # Return as string
            return data.decode("utf-8", errors="replace")

    async def _decode_image_bytes(self, data: bytes) -> torch.Tensor:
        """Decode image bytes to tensor."""
        pil_image = Image.open(io.BytesIO(data))
        pil_image = ImageOps.exif_transpose(pil_image)

        if pil_image.mode == "I":
            pil_image = pil_image.point(lambda i: i * (1 / 255))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        return tensor

    async def _decode_audio_bytes(self, data: bytes) -> dict:
        """Decode audio bytes to ComfyUI format."""
        if torchaudio is None:
            raise ImportError("torchaudio is required for audio processing")

        buffer = io.BytesIO(data)
        waveform, sample_rate = torchaudio.load(buffer)

        target_rate = 44100
        if sample_rate != target_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
            waveform = resampler(waveform)
            sample_rate = target_rate

        return {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": sample_rate
        }

    def _looks_like_image_url(self, url: str) -> bool:
        """Check if URL looks like an image based on extension."""
        lower = url.lower()
        return any(ext in lower for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'])

    def _looks_like_audio_url(self, url: str) -> bool:
        """Check if URL looks like audio based on extension."""
        lower = url.lower()
        return any(ext in lower for ext in ['.mp3', '.wav', '.flac', '.ogg', '.opus', '.m4a'])

    # Specialized extraction methods for nodes

    async def process_image(self, value: Any) -> torch.Tensor:
        """
        Process a single image input.

        Args:
            value: Base64 string, URL, or already a tensor

        Returns:
            Tensor [1, H, W, C]
        """
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, str):
            if self._is_base64_image(value):
                return await self._decode_base64_image(value)
            elif self._is_url(value):
                return await self._download_and_decode(value)
        # Return empty tensor for invalid input
        return torch.zeros(1, 64, 64, 3)

    async def process_audio(self, value: Any) -> dict:
        """
        Process a single audio input.

        Args:
            value: Base64 string, URL, or already an audio dict

        Returns:
            {"waveform": tensor, "sample_rate": int}
        """
        if isinstance(value, dict) and "waveform" in value:
            return value
        if isinstance(value, str):
            if self._is_base64_audio(value):
                return await self._decode_base64_audio(value)
            elif self._is_url(value):
                return await self._download_and_decode(value)
        # Return empty audio for invalid input
        return {"waveform": torch.zeros(1, 1, 44100), "sample_rate": 44100}

    async def process_mask(self, value: Any) -> torch.Tensor:
        """
        Process a mask input (grayscale image).

        Args:
            value: Base64 string or URL of grayscale image

        Returns:
            Tensor [1, H, W] for MASK type
        """
        if isinstance(value, torch.Tensor):
            if len(value.shape) == 3:
                return value
            elif len(value.shape) == 4:
                # Convert [B, H, W, C] to [B, H, W] using first channel
                return value[:, :, :, 0]

        if isinstance(value, str):
            # Decode image
            if self._is_base64_image(value):
                tensor = await self._decode_base64_image(value)
            elif self._is_url(value):
                tensor = await self._download_and_decode(value)
            else:
                return torch.zeros(1, 64, 64)

            # Convert to grayscale mask
            if len(tensor.shape) == 4:
                # Average RGB channels
                mask = tensor.mean(dim=-1)
            else:
                mask = tensor
            return mask

        return torch.zeros(1, 64, 64)

    async def process_video(self, value: Any) -> Any:
        """
        Process video input.

        Note: Video processing is more complex and may require
        additional dependencies like decord or av.

        Args:
            value: Base64 string or URL

        Returns:
            Video data (format depends on ComfyUI video nodes)
        """
        # Basic implementation - return None for now
        # Full video support would require decord or av
        logger.warning("Video input processing not fully implemented")
        return None

    async def process_latent(self, value: Any) -> dict:
        """
        Process latent input (safetensors or torch file).

        Args:
            value: Base64 string (safetensors/pt) or URL

        Returns:
            {"samples": tensor, ...} latent dict
        """
        if isinstance(value, dict) and "samples" in value:
            return value

        default_latent = {"samples": torch.zeros(1, 4, 64, 64)}

        if isinstance(value, str):
            try:
                # Get data bytes
                if value.startswith("data:"):
                    _, data_str = value.split(",", 1)
                    data = base64.b64decode(data_str)
                elif self._is_url(value):
                    if aiohttp is None:
                        return default_latent
                    timeout = aiohttp.ClientTimeout(total=self.download_timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(value) as response:
                            response.raise_for_status()
                            data = await response.read()
                else:
                    # Raw base64
                    data = base64.b64decode(value)

                # Try safetensors first
                try:
                    from safetensors.torch import load
                    buffer = io.BytesIO(data)
                    tensors = load(buffer.getvalue())
                    if "samples" in tensors:
                        return {"samples": tensors["samples"]}
                    # Return first tensor as samples
                    for key, tensor in tensors.items():
                        return {"samples": tensor}
                except ImportError:
                    pass
                except Exception:
                    pass

                # Try torch load
                try:
                    buffer = io.BytesIO(data)
                    loaded = torch.load(buffer, weights_only=True)
                    if isinstance(loaded, dict) and "samples" in loaded:
                        return loaded
                    if isinstance(loaded, torch.Tensor):
                        return {"samples": loaded}
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"Failed to process latent: {e}")

        return default_latent

    async def process_mesh(self, value: Any) -> dict:
        """
        Process 3D mesh input (GLB, OBJ, STL).

        Args:
            value: Base64 string or URL

        Returns:
            {"vertices": tensor, "faces": tensor} mesh dict
        """
        default_mesh = {
            "vertices": torch.zeros(1, 3, 3),
            "faces": torch.zeros(1, 1, 3, dtype=torch.long)
        }

        if isinstance(value, dict) and "vertices" in value:
            return value

        if isinstance(value, str):
            try:
                # Get data bytes
                if value.startswith("data:"):
                    _, data_str = value.split(",", 1)
                    data = base64.b64decode(data_str)
                elif self._is_url(value):
                    if aiohttp is None:
                        return default_mesh
                    timeout = aiohttp.ClientTimeout(total=self.download_timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(value) as response:
                            response.raise_for_status()
                            data = await response.read()
                else:
                    # Raw base64
                    data = base64.b64decode(value)

                # Detect format and parse
                return self._parse_mesh_bytes(data, value)

            except Exception as e:
                logger.error(f"Failed to process mesh: {e}")

        return default_mesh

    def _parse_mesh_bytes(self, data: bytes, hint: str = "") -> dict:
        """Parse mesh from bytes based on format detection."""
        default_mesh = {
            "vertices": torch.zeros(1, 3, 3),
            "faces": torch.zeros(1, 1, 3, dtype=torch.long)
        }

        # Detect format
        if data.startswith(b'glTF'):
            return self._parse_glb(data)
        elif data.startswith(b'solid ') or data[:80].count(b'\x00') < 10:
            # Text STL or ASCII format
            if b'v ' in data[:1000]:
                return self._parse_obj(data)
        elif len(data) > 84:
            # Binary STL check
            try:
                import struct
                num_triangles = struct.unpack('<I', data[80:84])[0]
                expected_size = 84 + num_triangles * 50
                if len(data) == expected_size:
                    return self._parse_stl_binary(data)
            except:
                pass

        # Try OBJ as fallback for text formats
        if b'v ' in data[:1000]:
            return self._parse_obj(data)

        return default_mesh

    def _parse_glb(self, data: bytes) -> dict:
        """Parse GLB (binary glTF) file."""
        import struct
        import json as json_mod

        default_mesh = {
            "vertices": torch.zeros(1, 3, 3),
            "faces": torch.zeros(1, 1, 3, dtype=torch.long)
        }

        try:
            # GLB header: magic (4), version (4), length (4)
            if len(data) < 12:
                return default_mesh

            magic, version, length = struct.unpack('<4sII', data[:12])
            if magic != b'glTF':
                return default_mesh

            # Read JSON chunk
            chunk_length, chunk_type = struct.unpack('<II', data[12:20])
            json_data = json_mod.loads(data[20:20+chunk_length])

            # Read BIN chunk
            bin_offset = 20 + chunk_length
            if bin_offset + 8 > len(data):
                return default_mesh

            bin_length, bin_type = struct.unpack('<II', data[bin_offset:bin_offset+8])
            bin_data = data[bin_offset+8:bin_offset+8+bin_length]

            # Extract mesh from first primitive
            meshes = json_data.get("meshes", [])
            if not meshes:
                return default_mesh

            primitive = meshes[0].get("primitives", [{}])[0]
            accessors = json_data.get("accessors", [])
            buffer_views = json_data.get("bufferViews", [])

            # Get vertices
            pos_accessor_idx = primitive.get("attributes", {}).get("POSITION")
            if pos_accessor_idx is None:
                return default_mesh

            pos_accessor = accessors[pos_accessor_idx]
            pos_view = buffer_views[pos_accessor["bufferView"]]
            pos_offset = pos_view.get("byteOffset", 0)
            vertex_count = pos_accessor["count"]

            vertices_flat = np.frombuffer(
                bin_data[pos_offset:pos_offset + vertex_count * 12],
                dtype=np.float32
            )
            vertices = vertices_flat.reshape(-1, 3)

            # Get indices
            indices_accessor_idx = primitive.get("indices")
            if indices_accessor_idx is not None:
                idx_accessor = accessors[indices_accessor_idx]
                idx_view = buffer_views[idx_accessor["bufferView"]]
                idx_offset = idx_view.get("byteOffset", 0)
                idx_count = idx_accessor["count"]

                # Handle different component types
                comp_type = idx_accessor.get("componentType", 5125)
                if comp_type == 5123:  # UNSIGNED_SHORT
                    faces_flat = np.frombuffer(
                        bin_data[idx_offset:idx_offset + idx_count * 2],
                        dtype=np.uint16
                    )
                else:  # UNSIGNED_INT
                    faces_flat = np.frombuffer(
                        bin_data[idx_offset:idx_offset + idx_count * 4],
                        dtype=np.uint32
                    )
                faces = faces_flat.reshape(-1, 3)
            else:
                # Generate faces from vertices
                faces = np.arange(len(vertices)).reshape(-1, 3)

            return {
                "vertices": torch.from_numpy(vertices).unsqueeze(0).float(),
                "faces": torch.from_numpy(faces.astype(np.int64)).unsqueeze(0)
            }

        except Exception as e:
            logger.error(f"Failed to parse GLB: {e}")
            return default_mesh

    def _parse_obj(self, data: bytes) -> dict:
        """Parse OBJ file."""
        default_mesh = {
            "vertices": torch.zeros(1, 3, 3),
            "faces": torch.zeros(1, 1, 3, dtype=torch.long)
        }

        try:
            text = data.decode('utf-8', errors='replace')
            vertices = []
            faces = []

            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()[1:4]
                    vertices.append([float(p) for p in parts])
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    # Handle v/vt/vn format
                    face_indices = []
                    for p in parts[:3]:  # Only first 3 vertices for triangles
                        idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                        face_indices.append(idx)
                    if len(face_indices) == 3:
                        faces.append(face_indices)

            if vertices and faces:
                return {
                    "vertices": torch.tensor(vertices).unsqueeze(0).float(),
                    "faces": torch.tensor(faces, dtype=torch.long).unsqueeze(0)
                }

        except Exception as e:
            logger.error(f"Failed to parse OBJ: {e}")

        return default_mesh

    def _parse_stl_binary(self, data: bytes) -> dict:
        """Parse binary STL file."""
        import struct

        default_mesh = {
            "vertices": torch.zeros(1, 3, 3),
            "faces": torch.zeros(1, 1, 3, dtype=torch.long)
        }

        try:
            num_triangles = struct.unpack('<I', data[80:84])[0]
            vertices = []
            faces = []

            offset = 84
            for i in range(num_triangles):
                # Skip normal (12 bytes)
                offset += 12
                # Read 3 vertices
                for j in range(3):
                    v = struct.unpack('<fff', data[offset:offset+12])
                    vertices.append(list(v))
                    offset += 12
                # Skip attribute byte count
                offset += 2

                # Create face indices
                base_idx = i * 3
                faces.append([base_idx, base_idx + 1, base_idx + 2])

            if vertices and faces:
                return {
                    "vertices": torch.tensor(vertices).unsqueeze(0).float(),
                    "faces": torch.tensor(faces, dtype=torch.long).unsqueeze(0)
                }

        except Exception as e:
            logger.error(f"Failed to parse STL: {e}")

        return default_mesh
