"""Validation utilities for webhook functionality."""

import re
import ipaddress
import logging
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger(__name__)

# Regex for data URL detection
DATA_URL_PATTERN = re.compile(r'^data:([^;,]+)?(;base64)?,(.*)$', re.DOTALL)
URL_PATTERN = re.compile(r'^https?://')


def validate_url(
    url: str,
    require_https: bool = False,
    block_private_ips: bool = False,
) -> tuple[bool, Optional[str]]:
    """
    Validate a webhook URL for safety and correctness.

    Args:
        url: URL to validate
        require_https: If True, only allow HTTPS URLs
        block_private_ips: If True, block private/local IP addresses

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL is empty"

    try:
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid URL scheme: {parsed.scheme}"

        if require_https and parsed.scheme != "https":
            return False, "HTTPS is required"

        # Check host
        if not parsed.netloc:
            return False, "URL has no host"

        # Check for private IPs if requested
        if block_private_ips:
            hostname = parsed.hostname
            if hostname:
                # Check for localhost
                if hostname in ("localhost", "127.0.0.1", "::1"):
                    return False, "Localhost URLs are not allowed"

                # Check for private IP ranges
                try:
                    ip = ipaddress.ip_address(hostname)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        return False, "Private IP addresses are not allowed"
                except ValueError:
                    # Not an IP address, hostname is fine
                    pass

        return True, None

    except Exception as e:
        return False, f"URL parsing error: {str(e)}"


def sanitize_for_logging(data: dict, sensitive_keys: Optional[set] = None) -> dict:
    """
    Sanitize a dictionary for safe logging by redacting sensitive values.

    Args:
        data: Dictionary to sanitize
        sensitive_keys: Set of key names to redact (case-insensitive)

    Returns:
        Sanitized copy of the dictionary
    """
    if sensitive_keys is None:
        sensitive_keys = {
            "auth_value",
            "authorization",
            "api_key",
            "apikey",
            "token",
            "password",
            "secret",
            "credential",
            "bearer",
        }

    def redact_value(key: str, value):
        if isinstance(value, dict):
            return {k: redact_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [redact_value(key, item) for item in value]
        elif isinstance(value, str):
            # Check if key is sensitive
            if key.lower() in sensitive_keys:
                return "[REDACTED]"
            # Check if value looks like base64 data (long strings)
            if len(value) > 100 and is_base64_data(value):
                return f"[BASE64_DATA: {len(value)} chars]"
        return value

    return {k: redact_value(k, v) for k, v in data.items()}


def is_base64_data(value: str) -> bool:
    """
    Check if a string appears to be base64 encoded data.

    Args:
        value: String to check

    Returns:
        True if the string looks like base64 data
    """
    # Check for data URL prefix
    if value.startswith("data:"):
        return True

    # Check for base64 character pattern (rough check)
    # Base64 uses A-Z, a-z, 0-9, +, /, and = for padding
    if len(value) > 50 and re.match(r'^[A-Za-z0-9+/]+=*$', value):
        return True

    return False


def is_url(value: str) -> bool:
    """
    Check if a string is an HTTP(S) URL.

    Args:
        value: String to check

    Returns:
        True if the string is a valid URL
    """
    return bool(URL_PATTERN.match(value))


def is_data_url(value: str) -> bool:
    """
    Check if a string is a data URL.

    Args:
        value: String to check

    Returns:
        True if the string is a data URL
    """
    return bool(DATA_URL_PATTERN.match(value))


def parse_data_url(value: str) -> tuple[Optional[str], Optional[bytes]]:
    """
    Parse a data URL and extract the content.

    Args:
        value: Data URL string

    Returns:
        Tuple of (mime_type, decoded_bytes) or (None, None) if invalid
    """
    import base64

    match = DATA_URL_PATTERN.match(value)
    if not match:
        return None, None

    mime_type = match.group(1) or "application/octet-stream"
    is_base64 = match.group(2) is not None
    data = match.group(3)

    try:
        if is_base64:
            decoded = base64.b64decode(data)
        else:
            # URL-encoded data
            from urllib.parse import unquote
            decoded = unquote(data).encode()
        return mime_type, decoded
    except Exception as e:
        logger.warning(f"Failed to parse data URL: {e}")
        return None, None
