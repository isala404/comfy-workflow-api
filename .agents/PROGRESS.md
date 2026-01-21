Extended type support and auto-batch handling
- Added 3D types: GAUSSIAN_SPLATTING, POINT_CLOUD, DEPTH_MAP, NORMAL_MAP, CAMERA_POSES
- Removed batch_outputs parameter from WebhookSend
- DECISION: Always use _N suffix for all outputs for consistency (user request)
- PLY format for Gaussian and Point Cloud output (binary little-endian)
- Depth maps: EXR preferred, PNG16 fallback when OpenEXR not installed
- Normal maps: convert [-1,1] to [0,255] PNG

Files modified:
- core/types.py: Added OutputInfo fields (point_count, has_sh_coefficients, depth_min/max, camera_count, has_intrinsics)
- utils/mime.py: Added .ply (application/x-ply), .exr (image/x-exr), .spz
- utils/helpers.py: Added format_count() for large numbers
- nodes/send.py: New encoders for 3D types, auto-batch, updated debug output
- nodes/transformer.py: Dict detection for new 3D types
- processors/inputs.py: PLY and EXR parsing (binary/ASCII PLY, Gaussian detection)

Detection logic:
- GAUSSIAN_SPLATTING vs POINT_CLOUD: presence of opacity/scales/sh_coefficients
- DEPTH_MAP vs regular float tensor: values outside [0,1]
- NORMAL_MAP: 3-channel tensor with negative values

Code review fixes (post-review):
- Fixed mask loop: separated 2D single mask from 3D batched mask handling
- Fixed EXR encoding: single Y channel instead of duplicated RGB, proper temp file cleanup with finally
- Fixed PLY parsing: CRLF handling, vertex count validation against data size
- Updated _encode_value return type annotation to reflect always-list return
