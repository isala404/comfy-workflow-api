"""Shared helper functions."""

import asyncio
import concurrent.futures


def format_size(size_bytes: int) -> str:
    """Format bytes as human readable."""
    if size_bytes is None:
        return "unknown"
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_count(count: int) -> str:
    """Format large counts: 1234567 -> '1.2M'"""
    if count is None:
        return "unknown"
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"


def run_async(coro):
    """Run an async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
