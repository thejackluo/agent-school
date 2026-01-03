"""
Entities Module - Entity Registry for cross-platform identity management

Enables instant name→identity lookups:
- "Jack Luo" → jack145945@gmail.com (Gmail)
- "Jack" → @jackluo (Notion)
"""

from .entity_registry import Entity, EntityRegistry

__all__ = [
    "Entity",
    "EntityRegistry",
]
