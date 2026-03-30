"""Query cache package for stage 2.

This module exposes a small interface for semantic caching of query results.
"""

from .lru_cache import QueryCache  # noqa: F401

__all__ = ["QueryCache"]
