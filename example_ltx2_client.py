#!/usr/bin/env python3
"""
Example client for comfy-workflow-api (LTX-2 Text-to-Video).

Submits a prompt to ComfyUI and receives generated video via webhook.

Usage:
    python example_ltx2_client.py <prompt>

Examples:
    python example_ltx2_client.py "a cat walking on the beach at sunset"
    python example_ltx2_client.py "timelapse of clouds moving over mountains"

Requirements:
    pip install requests

Make sure ComfyUI is running with LTX-2 models loaded.
"""

import gzip
import json
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import requests

# Configuration (override via env vars)
import os

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8188")
CALLBACK_URL = os.environ.get("CALLBACK_URL", "http://localhost:5001/webhook")
CALLBACK_PORT = 5001
OUTPUT_DIR = Path("./outputs")

# LTX-2 Text-to-Video workflow with spatial upscaling
WORKFLOW = {
    "98": {
        "inputs": {
            "field": "prompt",
            "debug": True,
            "webhook_context": ["99", 0],
        },
        "class_type": "WebhookTransformer",
        "_meta": {"title": "Workflow API Transformer"},
    },
    "99": {
        "inputs": {
            "default_callback_url": "",
            "default_timeout": 60,
            "default_max_retries": 3,
            "debug": True,
        },
        "class_type": "WebhookReceiver",
        "_meta": {"title": "Workflow API Receiver"},
    },
    "100": {
        "inputs": {
            "field_1_name": "video",
            "field_2_name": "output_2",
            "field_3_name": "output_3",
            "field_4_name": "output_4",
            "field_5_name": "output_5",
            "debug": False,
            "batch_outputs": False,
            "webhook_context": ["99", 0],
            "field_1": ["92:97", 0],
        },
        "class_type": "WebhookSend",
        "_meta": {"title": "Workflow API Send"},
    },
    "92:9": {
        "inputs": {
            "steps": 20,
            "max_shift": 2.05,
            "base_shift": 0.95,
            "stretch": True,
            "terminal": 0.1,
            "latent": ["92:56", 0],
        },
        "class_type": "LTXVScheduler",
        "_meta": {"title": "LTXVScheduler"},
    },
    "92:60": {
        "inputs": {
            "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
            "ckpt_name": "ltx-2-19b-dev-fp8.safetensors",
            "device": "default",
        },
        "class_type": "LTXAVTextEncoderLoader",
        "_meta": {"title": "LTXV Audio Text Encoder Loader"},
    },
    "92:73": {
        "inputs": {"sigmas": "0.909375, 0.725, 0.421875, 0.0"},
        "class_type": "ManualSigmas",
        "_meta": {"title": "ManualSigmas"},
    },
    "92:76": {
        "inputs": {"model_name": "ltx-2-spatial-upscaler-x2-1.0.safetensors"},
        "class_type": "LatentUpscaleModelLoader",
        "_meta": {"title": "Load Latent Upscale Model"},
    },
    "92:81": {
        "inputs": {
            "positive": ["92:22", 0],
            "negative": ["92:22", 1],
            "latent": ["92:80", 0],
        },
        "class_type": "LTXVCropGuides",
        "_meta": {"title": "LTXVCropGuides"},
    },
    "92:82": {
        "inputs": {
            "cfg": 1,
            "model": ["92:68", 0],
            "positive": ["92:81", 0],
            "negative": ["92:81", 1],
        },
        "class_type": "CFGGuider",
        "_meta": {"title": "CFGGuider"},
    },
    "92:90": {
        "inputs": {
            "upscale_method": "lanczos",
            "scale_by": 0.5,
            "image": ["92:89", 0],
        },
        "class_type": "ImageScaleBy",
        "_meta": {"title": "Upscale Image By"},
    },
    "92:91": {
        "inputs": {"image": ["92:90", 0]},
        "class_type": "GetImageSize",
        "_meta": {"title": "Get Image Size"},
    },
    "92:51": {
        "inputs": {
            "frames_number": ["92:62", 0],
            "frame_rate": ["92:99", 0],
            "batch_size": 1,
            "audio_vae": ["92:48", 0],
        },
        "class_type": "LTXVEmptyLatentAudio",
        "_meta": {"title": "LTXV Empty Latent Audio"},
    },
    "92:22": {
        "inputs": {
            "frame_rate": ["92:102", 0],
            "positive": ["92:3", 0],
            "negative": ["92:4", 0],
        },
        "class_type": "LTXVConditioning",
        "_meta": {"title": "LTXVConditioning"},
    },
    "92:43": {
        "inputs": {
            "width": ["92:91", 0],
            "height": ["92:91", 1],
            "length": ["92:62", 0],
            "batch_size": 1,
        },
        "class_type": "EmptyLTXVLatentVideo",
        "_meta": {"title": "EmptyLTXVLatentVideo"},
    },
    "92:56": {
        "inputs": {
            "video_latent": ["92:43", 0],
            "audio_latent": ["92:51", 0],
        },
        "class_type": "LTXVConcatAVLatent",
        "_meta": {"title": "LTXVConcatAVLatent"},
    },
    "92:4": {
        "inputs": {
            "text": "blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles",
            "clip": ["92:60", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "92:89": {
        "inputs": {"width": 1280, "height": 720, "batch_size": 1, "color": 0},
        "class_type": "EmptyImage",
        "_meta": {"title": "EmptyImage"},
    },
    "92:62": {
        "inputs": {"value": 121},
        "class_type": "PrimitiveInt",
        "_meta": {"title": "Length"},
    },
    "92:41": {
        "inputs": {
            "noise": ["92:11", 0],
            "guider": ["92:47", 0],
            "sampler": ["92:8", 0],
            "sigmas": ["92:9", 0],
            "latent_image": ["92:56", 0],
        },
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "SamplerCustomAdvanced"},
    },
    "92:67": {
        "inputs": {"noise_seed": 0},
        "class_type": "RandomNoise",
        "_meta": {"title": "RandomNoise"},
    },
    "92:11": {
        "inputs": {"noise_seed": 10},
        "class_type": "RandomNoise",
        "_meta": {"title": "RandomNoise"},
    },
    "92:80": {
        "inputs": {"av_latent": ["92:41", 0]},
        "class_type": "LTXVSeparateAVLatent",
        "_meta": {"title": "LTXVSeparateAVLatent"},
    },
    "92:83": {
        "inputs": {
            "video_latent": ["92:84", 0],
            "audio_latent": ["92:80", 1],
        },
        "class_type": "LTXVConcatAVLatent",
        "_meta": {"title": "LTXVConcatAVLatent"},
    },
    "92:84": {
        "inputs": {
            "samples": ["92:81", 2],
            "upscale_model": ["92:76", 0],
            "vae": ["92:1", 2],
        },
        "class_type": "LTXVLatentUpsampler",
        "_meta": {"title": "spatial"},
    },
    "92:70": {
        "inputs": {
            "noise": ["92:67", 0],
            "guider": ["92:82", 0],
            "sampler": ["92:66", 0],
            "sigmas": ["92:73", 0],
            "latent_image": ["92:83", 0],
        },
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "SamplerCustomAdvanced"},
    },
    "92:3": {
        "inputs": {"text": ["98", 0], "clip": ["92:60", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "92:97": {
        "inputs": {
            "fps": ["92:102", 0],
            "images": ["92:98", 0],
            "audio": ["92:96", 0],
        },
        "class_type": "CreateVideo",
        "_meta": {"title": "Create Video"},
    },
    "92:48": {
        "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"},
        "class_type": "LTXVAudioVAELoader",
        "_meta": {"title": "LTXV Audio VAE Loader"},
    },
    "92:94": {
        "inputs": {"av_latent": ["92:70", 1]},
        "class_type": "LTXVSeparateAVLatent",
        "_meta": {"title": "LTXVSeparateAVLatent"},
    },
    "92:98": {
        "inputs": {
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 4096,
            "temporal_overlap": 8,
            "samples": ["92:94", 0],
            "vae": ["92:1", 2],
        },
        "class_type": "VAEDecodeTiled",
        "_meta": {"title": "VAE Decode (Tiled)"},
    },
    "92:96": {
        "inputs": {"samples": ["92:94", 1], "audio_vae": ["92:48", 0]},
        "class_type": "LTXVAudioVAEDecode",
        "_meta": {"title": "LTXV Audio VAE Decode"},
    },
    "92:47": {
        "inputs": {
            "cfg": 4,
            "model": ["92:1", 0],
            "positive": ["92:22", 0],
            "negative": ["92:22", 1],
        },
        "class_type": "CFGGuider",
        "_meta": {"title": "CFGGuider"},
    },
    "92:102": {
        "inputs": {"value": 24},
        "class_type": "PrimitiveFloat",
        "_meta": {"title": "Frame Rate(float)"},
    },
    "92:99": {
        "inputs": {"value": 24},
        "class_type": "PrimitiveInt",
        "_meta": {"title": "Frame Rate(int)"},
    },
    "92:68": {
        "inputs": {
            "lora_name": "ltx-2-19b-distilled-lora-384.safetensors",
            "strength_model": 1,
            "model": ["92:1", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "LoraLoaderModelOnly"},
    },
    "92:8": {
        "inputs": {"sampler_name": "euler_ancestral"},
        "class_type": "KSamplerSelect",
        "_meta": {"title": "KSamplerSelect"},
    },
    "92:66": {
        "inputs": {"sampler_name": "euler_ancestral"},
        "class_type": "KSamplerSelect",
        "_meta": {"title": "KSamplerSelect"},
    },
    "92:1": {
        "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"},
    },
}


class WebhookHandler(BaseHTTPRequestHandler):
    """Handles webhook callbacks from ComfyUI."""

    files_received = []
    done = threading.Event()

    def log_message(self, *args):
        pass

    def do_POST(self):
        content_type = self.headers.get("Content-Type", "")
        content_encoding = self.headers.get("Content-Encoding", "")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if "gzip" in content_encoding:
            body = gzip.decompress(body)

        if "application/json" in content_type:
            data = json.loads(body)
            event = data.get("event")

            if event == "workflow.progress":
                progress = data.get("progress", {})
                print(f"  Progress: {progress.get('value')}/{progress.get('max')}")
            elif event == "workflow.completed":
                print(f"  Completed in {data.get('execution_time_ms', 'N/A')}ms")
            elif event == "workflow.error":
                print(f"  Error: {data.get('error')}")
                WebhookHandler.done.set()

        elif "multipart" in content_type:
            boundary = content_type.split("boundary=")[1]
            parts = body.split(f"--{boundary}".encode())

            OUTPUT_DIR.mkdir(exist_ok=True)

            for part in parts:
                if b"filename=" in part and b"\r\n\r\n" in part:
                    header, content = part.split(b"\r\n\r\n", 1)
                    filename = header.decode().split('filename="')[1].split('"')[0]
                    content = content.rstrip(b"\r\n--")

                    path = OUTPUT_DIR / filename
                    path.write_bytes(content)
                    WebhookHandler.files_received.append(str(path))
                    print(f"  Saved: {path}")

            WebhookHandler.done.set()

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def main():
    if len(sys.argv) < 2:
        print("Usage: python example_ltx2_client.py <prompt>")
        print('Example: python example_ltx2_client.py "a cat walking on the beach"')
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])

    print(f"Prompt: {prompt}")
    print(f"ComfyUI: {COMFYUI_URL}")
    print(f"Callback: {CALLBACK_URL}")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print(f"Video: 1280x720, 121 frames @ 24fps (~5s)\n")

    server = HTTPServer(("0.0.0.0", CALLBACK_PORT), WebhookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print("Submitting workflow...")
    response = requests.post(
        f"{COMFYUI_URL}/api/webhook",
        files={
            "workflow": (None, json.dumps(WORKFLOW), "application/json"),
            "callback_url": (None, CALLBACK_URL),
            "prompt": (None, prompt),
        },
    )

    result = response.json()
    print(f"  Request ID: {result.get('request_id')}")
    print(f"  Prompt ID: {result.get('prompt_id')}\n")

    print("Processing (this may take a few minutes)...")
    WebhookHandler.done.wait(timeout=600)

    server.shutdown()

    if WebhookHandler.files_received:
        print(f"\nDone! Video saved to {OUTPUT_DIR.absolute()}")
    else:
        print("\nNo files received.")


if __name__ == "__main__":
    main()
