"""SHA-256 helpers for content dedup."""
import hashlib


def sha256_hex(text: str) -> str:
    """Hex SHA-256 digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
