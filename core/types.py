"""Core data types for webhook functionality."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
import uuid


class WebhookEvent(Enum):
    """Types of webhook events."""
    STARTED = "workflow.started"
    PROGRESS = "workflow.progress"
    COMPLETED = "workflow.completed"
    ERROR = "workflow.error"
    INTERRUPTED = "workflow.interrupted"


@dataclass
class WebhookField:
    """Individual field from webhook request."""
    name: str
    value: Any
    is_file: bool = False
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None


@dataclass
class WebhookContext:
    """
    Context for a webhook request.

    Tracks the entire lifecycle of a webhook request from submission
    through execution and completion.
    """
    # Callback configuration
    callback_url: str = ""
    auth_header: Optional[str] = None
    auth_value: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3

    # Identity
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: Optional[str] = None

    # Parsed webhook fields
    fields: Dict[str, WebhookField] = field(default_factory=dict)

    # Raw inputs for fallback
    inputs: dict = field(default_factory=dict)

    # Progress settings
    send_progress: bool = True
    progress_interval_ms: int = 500

    # Request metadata
    remote_ip: Optional[str] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    user_agent: Optional[str] = None

    # Workflow settings
    include_workflow_in_response: bool = False

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def get_auth_headers(self) -> dict:
        """Get authentication headers for HTTP requests."""
        if self.auth_header and self.auth_value:
            return {self.auth_header: self.auth_value}
        return {}

    def censor_auth(self) -> str:
        """Get censored auth header for logging."""
        if self.auth_header and self.auth_value:
            value = self.auth_value
            if len(value) > 12:
                censored = value[:4] + "*" * (len(value) - 8) + value[-4:]
            else:
                censored = "*" * len(value)
            return f"{self.auth_header}: {censored}"
        return "(not set)"

    def has_field(self, name: str) -> bool:
        """Check if a field exists."""
        return name in self.fields or name in self.inputs

    def get_field(self, name: str) -> Optional[WebhookField]:
        """Get a WebhookField by name."""
        return self.fields.get(name)

    def get_field_value(self, name: str, default: Any = None) -> Any:
        """Get a field's value."""
        if name in self.fields:
            return self.fields[name].value
        if name in self.inputs:
            return self.inputs[name]
        return default

    def get_field_names(self) -> list:
        """Get all field names."""
        names = set(self.fields.keys())
        names.update(self.inputs.keys())
        return sorted(names)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "prompt_id": self.prompt_id,
            "callback_url": self.callback_url,
            "fields": list(self.fields.keys()),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class WebhookResult:
    """Result of a webhook HTTP request."""
    success: bool
    status: Optional[int] = None
    error: Optional[str] = None
    response_body: Optional[str] = None
    elapsed_ms: float = 0.0
    retries: int = 0


@dataclass
class OutputInfo:
    """Information about a collected output file."""
    # Required fields
    type: str = "UNKNOWN"
    filename: str = ""
    mime_type: str = "application/octet-stream"

    # File info
    file_size_bytes: Optional[int] = None

    # Source info (for automatic collection)
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    output_key: Optional[str] = None
    subfolder: Optional[str] = None
    folder_type: str = "output"

    # Type-specific metadata
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    dtype: Optional[str] = None
    format: Optional[str] = None

    # Audio/Video metadata
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    frame_count: Optional[int] = None
    fps: Optional[float] = None

    # Batch info
    batch_index: Optional[int] = None
    batch_size: Optional[int] = None

    # Inline content for small text
    inline_content: Optional[str] = None

    # 3D type metadata
    point_count: Optional[int] = None
    has_sh_coefficients: Optional[bool] = None
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    camera_count: Optional[int] = None
    has_intrinsics: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for metadata payload."""
        result = {
            "type": self.type,
            "filename": self.filename,
            "mime_type": self.mime_type,
        }

        # Add optional fields if present
        for key in ["file_size_bytes", "node_id", "node_type", "output_key",
                    "width", "height", "channels", "dtype", "format",
                    "duration_seconds", "sample_rate", "frame_count", "fps",
                    "batch_index", "batch_size", "inline_content",
                    "point_count", "has_sh_coefficients", "depth_min", "depth_max",
                    "camera_count", "has_intrinsics"]:
            val = getattr(self, key)
            if val is not None:
                result[key] = val

        return result
