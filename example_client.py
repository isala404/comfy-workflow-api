#!/usr/bin/env python3
"""
Example client for comfy-workflow-api (Image-to-Image).

Submits an image + prompt to ComfyUI and receives the transformed image via webhook.

Usage:
    python example_client.py <image_path> [prompt]

Examples:
    python example_client.py input.png "make it look like a painting"
    python example_client.py photo.jpg "add snow to the scene"

Requirements:
    pip install requests

Make sure ComfyUI is running on http://localhost:8188
"""

import gzip
import json
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import requests

# Configuration
COMFYUI_URL = "http://localhost:8188"
CALLBACK_PORT = 5001
OUTPUT_DIR = Path("./outputs")
WORKFLOW_FILE = Path(__file__).parent / "example_workflow.json"


class WebhookHandler(BaseHTTPRequestHandler):
    """Handles webhook callbacks from ComfyUI."""

    files_received = []
    done = threading.Event()

    def log_message(self, *args):
        pass  # Suppress logging

    def do_POST(self):
        content_type = self.headers.get("Content-Type", "")
        content_encoding = self.headers.get("Content-Encoding", "")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Decompress gzip if needed (WebhookSend uses gzip compression)
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
            # Parse multipart and save files
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
        print("Usage: python example_client.py <image_path> [prompt]")
        print("Example: python example_client.py input.png \"make it look like a painting\"")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "enhance the image"

    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"Output: {OUTPUT_DIR.absolute()}\n")

    # Load workflow
    workflow = json.loads(WORKFLOW_FILE.read_text())

    # Load image
    image_data = image_path.read_bytes()
    image_mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

    # Start webhook server
    server = HTTPServer(("localhost", CALLBACK_PORT), WebhookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    callback_url = f"http://localhost:{CALLBACK_PORT}/webhook"

    # Submit workflow with image and prompt
    print("Submitting workflow...")
    response = requests.post(
        f"{COMFYUI_URL}/api/webhook",
        files={
            "workflow": (None, json.dumps(workflow), "application/json"),
            "callback_url": (None, callback_url),
            "prompt": (None, prompt),
            "image": (image_path.name, image_data, image_mime),
        }
    )

    result = response.json()
    print(f"  Request ID: {result.get('request_id')}")
    print(f"  Prompt ID: {result.get('prompt_id')}\n")

    # Wait for completion
    print("Processing...")
    WebhookHandler.done.wait(timeout=300)

    server.shutdown()

    if WebhookHandler.files_received:
        print(f"\nDone! Files saved to {OUTPUT_DIR.absolute()}")
    else:
        print("\nNo files received.")


if __name__ == "__main__":
    main()
