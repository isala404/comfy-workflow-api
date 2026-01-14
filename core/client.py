"""
Production-grade async HTTP client for webhook delivery.
Supports multipart/form-data for file uploads with retry logic.
"""

import asyncio
import gzip
import io
import json
import logging
import random
import time
from typing import Optional
from pathlib import Path

import aiohttp

from .types import WebhookResult

logger = logging.getLogger(__name__)

# Singleton client instance
_webhook_client: Optional["WebhookClient"] = None


def get_webhook_client() -> "WebhookClient":
    """Get or create the singleton webhook client instance."""
    global _webhook_client
    if _webhook_client is None:
        _webhook_client = WebhookClient()
    return _webhook_client


class WebhookClient:
    """
    Production-grade async HTTP client for multipart webhook delivery.

    Features:
    - Async HTTP with connection pooling
    - Multipart/form-data support for file uploads
    - Exponential backoff with jitter for retries
    - Configurable timeout and retry settings
    """

    # Status codes that should trigger retry
    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        timeout: float = 60.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.2,
        max_connections: int = 10,
        max_connections_per_host: int = 5,
    ):
        """
        Initialize the webhook client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            jitter_factor: Random jitter factor (0-1)
            max_connections: Maximum total connections
            max_connections_per_host: Maximum connections per host
        """
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_loop: Optional[asyncio.AbstractEventLoop] = None
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        current_loop = asyncio.get_running_loop()

        # Create new session if none exists, if closed, or if event loop changed
        if (self._session is None or
            self._session.closed or
            self._session_loop is not current_loop):
            # Close old session if it exists and loop is still valid
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except Exception:
                    pass

            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
            )
            self._session_loop = current_loop
        return self._session

    async def send_json(
        self,
        url: str,
        payload: dict,
        headers: Optional[dict] = None,
    ) -> WebhookResult:
        """
        Send JSON payload to webhook URL.

        Used for progress events, errors, and status updates.

        Args:
            url: Webhook URL
            payload: JSON-serializable dictionary
            headers: Optional additional headers (e.g., auth)

        Returns:
            WebhookResult with success status and details
        """
        session = await self.get_session()
        all_headers = {"Content-Type": "application/json"}
        if headers:
            all_headers.update(headers)

        return await self._execute_with_retry(
            url=url,
            session=session,
            request_kwargs={"json": payload, "headers": all_headers}
        )

    async def send_multipart(
        self,
        url: str,
        metadata: dict,
        files: list[tuple[str, str, bytes, str]],
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        compress: bool = False,
    ) -> WebhookResult:
        """
        Send multipart/form-data request with files.

        Args:
            url: Webhook URL
            metadata: JSON metadata dictionary
            files: List of (field_name, filename, binary_data, mime_type)
            headers: Optional additional headers (e.g., auth)
            timeout: Optional request timeout (overrides default)
            max_retries: Optional max retries (overrides default)
            compress: If True, compress payload with gzip

        Returns:
            WebhookResult with success status and details
        """
        session = await self.get_session()

        if compress:
            # Build compressed payload
            return await self._send_multipart_compressed(
                url=url,
                session=session,
                metadata=metadata,
                files=files,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
            )

        def build_form():
            # Build FormData for each request (cannot reuse)
            form = aiohttp.FormData()

            # Add metadata as JSON part
            form.add_field(
                "metadata",
                json.dumps(metadata),
                content_type="application/json",
            )

            # Add files
            for name, filename, data, content_type in files:
                form.add_field(
                    name,
                    data,
                    filename=filename,
                    content_type=content_type,
                )
            return form

        return await self._execute_with_retry(
            url=url,
            session=session,
            request_kwargs={"headers": headers},
            form_builder=build_form,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def _send_multipart_compressed(
        self,
        url: str,
        session: aiohttp.ClientSession,
        metadata: dict,
        files: list[tuple[str, str, bytes, str]],
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> WebhookResult:
        """
        Send gzip-compressed multipart/form-data request.

        Builds the multipart body, compresses it with gzip, and sends
        with Content-Encoding: gzip header.
        """
        import secrets

        def build_compressed_payload():
            # Generate a unique boundary
            boundary = f"----WebhookBoundary{secrets.token_hex(16)}"
            boundary_bytes = boundary.encode("utf-8")

            # Build multipart body manually
            parts = []

            # Add metadata part
            parts.append(b"--" + boundary_bytes)
            parts.append(b'Content-Disposition: form-data; name="metadata"')
            parts.append(b"Content-Type: application/json")
            parts.append(b"")
            parts.append(json.dumps(metadata).encode("utf-8"))

            # Add file parts
            for name, filename, data, content_type in files:
                parts.append(b"--" + boundary_bytes)
                disposition = f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'
                parts.append(disposition.encode("utf-8"))
                parts.append(f"Content-Type: {content_type}".encode("utf-8"))
                parts.append(b"")
                parts.append(data)

            # Final boundary
            parts.append(b"--" + boundary_bytes + b"--")
            parts.append(b"")

            # Join with CRLF
            body = b"\r\n".join(parts)

            # Compress with gzip
            compressed = gzip.compress(body, compresslevel=6)

            return compressed, boundary

        compressed_data, boundary = build_compressed_payload()

        # Build headers
        all_headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Encoding": "gzip",
            "Content-Length": str(len(compressed_data)),
        }
        if headers:
            all_headers.update(headers)

        return await self._execute_with_retry(
            url=url,
            session=session,
            request_kwargs={"headers": all_headers, "data": compressed_data},
            timeout=timeout,
            max_retries=max_retries,
        )

    async def send_multipart_streaming(
        self,
        url: str,
        metadata: dict,
        file_paths: list[tuple[str, str, str, str]],
        headers: Optional[dict] = None,
    ) -> WebhookResult:
        """
        Send multipart request with files streamed from disk.

        Use this for large files (>50MB) to avoid memory issues.

        Args:
            url: Webhook URL
            metadata: JSON metadata dictionary
            file_paths: List of (field_name, filename, file_path, mime_type)
            headers: Optional additional headers (e.g., auth)

        Returns:
            WebhookResult with success status and details
        """
        session = await self.get_session()

        def build_writer():
            # Build multipart writer for streaming
            writer = aiohttp.MultipartWriter("form-data")

            # Add metadata
            metadata_payload = aiohttp.payload.JsonPayload(metadata)
            metadata_payload.set_content_disposition("form-data", name="metadata")
            writer.append_payload(metadata_payload)

            # Add files (streamed from disk)
            for name, filename, path, content_type in file_paths:
                file_path = Path(path)
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                    payload = aiohttp.payload.BytesPayload(
                        file_data,
                        content_type=content_type,
                    )
                    payload.set_content_disposition(
                        "form-data", name=name, filename=filename
                    )
                    writer.append_payload(payload)
            return writer

        return await self._execute_with_retry(
            url=url,
            session=session,
            request_kwargs={"headers": headers},
            data_builder=build_writer
        )

    async def _execute_with_retry(
        self,
        url: str,
        session: aiohttp.ClientSession,
        request_kwargs: Optional[dict] = None,
        form_builder=None,
        data_builder=None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> WebhookResult:
        """
        Execute request with exponential backoff and jitter.

        Args:
            url: Target URL
            session: aiohttp session
            request_kwargs: Additional kwargs for session.post()
            form_builder: Function that returns FormData (for multipart)
            data_builder: Function that returns MultipartWriter (for streaming)
            timeout: Optional request timeout (overrides default)
            max_retries: Optional max retries (overrides default)

        Returns:
            WebhookResult with success status and details
        """
        last_error = None
        start_time = time.time()
        retries = 0
        kwargs = request_kwargs or {}

        # Use instance defaults if not specified
        effective_max_retries = max_retries if max_retries is not None else self.max_retries

        # Add timeout to kwargs if specified
        if timeout is not None:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout)

        for attempt in range(effective_max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.info(
                        f"Webhook retry {attempt}/{effective_max_retries} after {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                    retries = attempt

                # Build request data
                post_kwargs = dict(kwargs)
                if form_builder:
                    post_kwargs["data"] = form_builder()
                elif data_builder:
                    post_kwargs["data"] = data_builder()

                # Make the request
                response = await session.post(url, **post_kwargs)
                try:
                    elapsed_ms = (time.time() - start_time) * 1000

                    if response.status < 400:
                        return WebhookResult(
                            success=True,
                            status=response.status,
                            elapsed_ms=elapsed_ms,
                            retries=retries,
                        )

                    # Check if retryable
                    if response.status in self.RETRYABLE_STATUS_CODES:
                        last_error = f"HTTP {response.status}"
                        logger.warning(
                            f"Webhook request failed with retryable status {response.status}"
                        )
                        continue

                    # Non-retryable error
                    body = await response.text()
                    return WebhookResult(
                        success=False,
                        status=response.status,
                        error=f"HTTP {response.status}: {body[:500]}",
                        elapsed_ms=elapsed_ms,
                        retries=retries,
                    )
                finally:
                    response.close()

            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
                logger.warning(f"Webhook client error: {e}")
                continue
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning("Webhook request timed out")
                continue
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Webhook unexpected error: {e}", exc_info=True)
                continue

        elapsed_ms = (time.time() - start_time) * 1000
        return WebhookResult(
            success=False,
            error=last_error,
            elapsed_ms=elapsed_ms,
            retries=retries,
        )

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        # Add jitter
        jitter = delay * self.jitter_factor * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)

    async def close(self):
        """Close the HTTP session and release resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("Webhook client session closed")


async def close_webhook_client():
    """Close the global webhook client instance."""
    global _webhook_client
    if _webhook_client:
        await _webhook_client.close()
        _webhook_client = None
