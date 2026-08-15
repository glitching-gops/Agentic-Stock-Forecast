"""
Shared FastAPI dependencies — API key verification.
"""
import os
import secrets

from fastapi import Header, HTTPException


def verify_api_key(x_api_key: str = Header(...)) -> None:
    """
    Validates the admin API key in constant time.

    ``!=`` on strings short-circuits at the first differing byte, which leaks
    the key one character at a time to an attacker who can measure response
    latency (audit finding F15). ``secrets.compare_digest`` does not.
    """
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin API is not configured")

    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=403, detail="Invalid API key")
