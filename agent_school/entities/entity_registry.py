"""
Entity Registry - Stores and looks up person/entity identities across platforms

Enables instant lookups like:
- "Jack Luo" → jack145945@gmail.com (Gmail)
- "Jack" → @jackluo (Notion)
- "Jack L" → U12345 (Slack)

Learns from browser explorations automatically.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class Entity:
    """
    A person or entity with identities across multiple platforms.
    
    Example:
        Entity(
            canonical_name="Jack Luo",
            aliases=["Jack", "jluo", "Jack L"],
            identities={
                "gmail": "jack145945@gmail.com",
                "notion": "@jackluo",
                "github": "thejackluo"
            }
        )
    """
    id: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    identities: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "identities": self.identities,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)
    
    def add_identity(self, platform: str, identity: str) -> None:
        """Add or update an identity for a platform."""
        self.identities[platform] = identity
        self.updated_at = datetime.now()
    
    def add_alias(self, alias: str) -> None:
        """Add an alias if not already present."""
        if alias.lower() not in [a.lower() for a in self.aliases]:
            self.aliases.append(alias)
            self.updated_at = datetime.now()
    
    def get_identity(self, platform: str) -> Optional[str]:
        """Get identity for a specific platform."""
        return self.identities.get(platform)
    
    def matches_name(self, query: str) -> float:
        """
        Check if this entity matches a name query.
        Returns a confidence score 0.0-1.0.
        """
        query_lower = query.lower().strip()
        
        # Exact match on canonical name
        if query_lower == self.canonical_name.lower():
            return 1.0
        
        # Exact match on alias
        for alias in self.aliases:
            if query_lower == alias.lower():
                return 0.95
        
        # Partial match on canonical name (first name, last name)
        name_parts = self.canonical_name.lower().split()
        if query_lower in name_parts:
            return 0.8
        
        # Partial match on aliases
        for alias in self.aliases:
            if query_lower in alias.lower() or alias.lower() in query_lower:
                return 0.6
        
        # Fuzzy contains (name contains query or vice versa)
        if query_lower in self.canonical_name.lower():
            return 0.4
        
        return 0.0


class EntityRegistry:
    """
    Registry for storing and looking up entities across platforms.
    
    Features:
    - Fuzzy name matching
    - Multi-platform identity storage
    - Automatic persistence to JSON
    - Learning from explorations
    """
    
    def __init__(self, base_dir: str = "workflows"):
        self.base_dir = Path(base_dir)
        self.entities_dir = self.base_dir / "entities"
        self.registry_path = self.entities_dir / "registry.json"
        self.entities: Dict[str, Entity] = {}
        
        # Load existing registry
        self._load()
    
    def _load(self) -> None:
        """Load entities from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                    for entity_data in data.get("entities", []):
                        entity = Entity.from_dict(entity_data)
                        self.entities[entity.id] = entity
            except Exception as e:
                print(f"[WARN] Failed to load entity registry: {e}")
    
    def _save(self) -> None:
        """Persist entities to disk."""
        self.entities_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            "entities": [e.to_dict() for e in self.entities.values()],
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def register(
        self,
        canonical_name: str,
        platform: str,
        identity: str,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """
        Register an entity or add identity to existing entity.
        
        Args:
            canonical_name: Full name (e.g., "Jack Luo")
            platform: Platform name (e.g., "gmail", "notion")
            identity: Platform-specific identity (e.g., "jack@gmail.com")
            aliases: Optional list of name aliases
            metadata: Optional additional metadata
        
        Returns:
            The created or updated Entity
        """
        # Check if entity already exists
        existing = self.find_by_name(canonical_name, min_confidence=0.9)
        
        if existing:
            entity = existing
            entity.add_identity(platform, identity)
            if aliases:
                for alias in aliases:
                    entity.add_alias(alias)
            if metadata:
                entity.metadata.update(metadata)
        else:
            # Create new entity
            entity = Entity(
                id=str(uuid.uuid4())[:8],
                canonical_name=canonical_name,
                aliases=aliases or [],
                identities={platform: identity},
                metadata=metadata or {},
            )
            self.entities[entity.id] = entity
        
        self._save()
        return entity
    
    def find_by_name(
        self,
        query: str,
        min_confidence: float = 0.4,
    ) -> Optional[Entity]:
        """
        Find an entity by name using fuzzy matching.
        
        Args:
            query: Name to search for (e.g., "Jack", "Jack Luo")
            min_confidence: Minimum confidence score (0.0-1.0)
        
        Returns:
            Best matching Entity or None
        """
        best_match: Optional[Entity] = None
        best_score = 0.0
        
        for entity in self.entities.values():
            score = entity.matches_name(query)
            if score > best_score and score >= min_confidence:
                best_score = score
                best_match = entity
        
        return best_match
    
    def find_by_identity(
        self,
        platform: str,
        identity: str,
    ) -> Optional[Entity]:
        """Find entity by platform identity."""
        for entity in self.entities.values():
            if entity.get_identity(platform) == identity:
                return entity
        return None
    
    def get_identity(
        self,
        query: str,
        platform: str,
        min_confidence: float = 0.4,
    ) -> Optional[str]:
        """
        Convenience method: find entity by name and return platform identity.
        
        Args:
            query: Name to search for
            platform: Platform to get identity for
            min_confidence: Minimum match confidence
        
        Returns:
            Platform identity or None
        """
        entity = self.find_by_name(query, min_confidence)
        if entity:
            return entity.get_identity(platform)
        return None
    
    def list_all(self) -> List[Entity]:
        """List all registered entities."""
        return list(self.entities.values())
    
    def delete(self, entity_id: str) -> bool:
        """Delete an entity by ID."""
        if entity_id in self.entities:
            del self.entities[entity_id]
            self._save()
            return True
        return False
