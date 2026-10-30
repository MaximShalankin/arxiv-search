"""Ingestion pipeline: PDF parse -> chunk -> embed -> write."""
from .chunker import Chunk, chunk_markdown
from .parser import ParseResult, ParserPool

__all__ = ["Chunk", "ParseResult", "ParserPool", "chunk_markdown"]
