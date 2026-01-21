Tooling
- Stack: Python 3, torch, PIL, numpy, aiohttp
- Node framework: ComfyUI custom nodes
- Test: python run_tests.py
- Dependencies: av (video/audio), safetensors (latent), OpenEXR (optional depth)

Structure
- core/: types, client, base functionality
- nodes/: ComfyUI node definitions (send.py, transformer.py)
- processors/: input processing (inputs.py)
- utils/: helpers, mime types

Patterns
- Wildcard (*) types for flexible inputs/outputs
- OutputInfo dataclass for metadata
- Async HTTP via aiohttp with sync wrapper (run_async)
- Binary format encoding: PLY for 3D, PNG for images, FLAC for audio

Type Detection Order (send.py _encode_value)
- IMAGE tensor [B,H,W,C] first
- MASK tensor [B,H,W] before general tensors
- NORMAL_MAP: 3-channel with negative values
- DEPTH_MAP: float tensor outside [0,1] range
- GAUSSIAN_SPLATTING: dict with positions + (opacity|scales|sh_coefficients)
- POINT_CLOUD: dict with points, no gaussian props
- CAMERA_POSES: dict with poses or cameras key
- Then VIDEO, MESH, LATENT, primitives, JSON fallback

Output Naming
- All outputs use _N suffix (e.g., output_0.png, output_1.png)
- Consistent even for single items (batch_size=1)
