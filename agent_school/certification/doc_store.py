"""
Doc Store - Simple storage for ingested help center documentation

Stores LLM-friendly documentation indexed by domain for retrieval
during exploration and certification.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


@dataclass
class DocEntry:
    """A stored documentation entry"""
    domain: str
    title: str
    content: str  # LLM-friendly markdown
    source_url: str
    ingested_at: datetime
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "title": self.title,
            "content": self.content,
            "source_url": self.source_url,
            "ingested_at": self.ingested_at.isoformat(),
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocEntry":
        data = data.copy()
        data["ingested_at"] = datetime.fromisoformat(data["ingested_at"])
        return cls(**data)


class DocStore:
    """
    Simple JSON-based storage for documentation.
    
    Structure:
    docs/
    ├── mail_google_com/
    │   ├── index.json (list of doc entries)
    │   └── entries/
    │       ├── compose_email.json
    │       └── search_inbox.json
    └── docs_google_com/
        └── ...
    """
    
    def __init__(self, base_dir: str = "docs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _domain_path(self, domain: str) -> Path:
        """Convert domain to safe directory path"""
        safe_name = domain.replace(".", "_").replace("/", "_")
        return self.base_dir / safe_name
    
    def store(self, entry: DocEntry) -> str:
        """
        Store a documentation entry.
        
        Returns:
            Path to stored entry
        """
        domain_dir = self._domain_path(entry.domain)
        entries_dir = domain_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename from title
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in entry.title)
        safe_title = safe_title.replace(" ", "_").lower()[:50]
        
        entry_path = entries_dir / f"{safe_title}.json"
        with open(entry_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)
        
        # Update index
        self._update_index(entry.domain)
        
        return str(entry_path)
    
    def _update_index(self, domain: str) -> None:
        """Update the domain index file"""
        domain_dir = self._domain_path(domain)
        entries_dir = domain_dir / "entries"
        
        if not entries_dir.exists():
            return
        
        index = []
        for entry_file in entries_dir.glob("*.json"):
            with open(entry_file, "r") as f:
                data = json.load(f)
                index.append({
                    "title": data["title"],
                    "file": entry_file.name,
                    "tags": data.get("tags", []),
                })
        
        index_path = domain_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump({
                "domain": domain,
                "updated_at": datetime.now().isoformat(),
                "entries": index,
            }, f, indent=2)
    
    def retrieve(self, domain: str, query: Optional[str] = None) -> List[DocEntry]:
        """
        Retrieve documentation for a domain.
        
        Args:
            domain: Domain to retrieve docs for (e.g., "mail.google.com")
            query: Optional query to filter docs (simple keyword match for now)
            
        Returns:
            List of matching DocEntry objects
        """
        domain_dir = self._domain_path(domain)
        entries_dir = domain_dir / "entries"
        
        if not entries_dir.exists():
            return []
        
        entries = []
        for entry_file in entries_dir.glob("*.json"):
            with open(entry_file, "r") as f:
                data = json.load(f)
                entry = DocEntry.from_dict(data)
                
                # Simple keyword matching if query provided
                if query:
                    query_lower = query.lower()
                    if (query_lower in entry.title.lower() or 
                        query_lower in entry.content.lower() or
                        any(query_lower in tag.lower() for tag in entry.tags)):
                        entries.append(entry)
                else:
                    entries.append(entry)
        
        return entries
    
    def retrieve_context(self, domain: str, task: str, max_tokens: int = 2000) -> str:
        """
        Retrieve relevant documentation as context string.
        
        This is what gets injected into exploration prompts.
        
        Args:
            domain: Target domain
            task: Task description to find relevant docs
            max_tokens: Approximate max tokens to return
            
        Returns:
            Formatted context string for LLM
        """
        entries = self.retrieve(domain, query=task)
        
        if not entries:
            return f"No documentation available for {domain}."
        
        context_parts = [f"# Documentation for {domain}\n"]
        current_length = 0
        
        for entry in entries:
            entry_text = f"\n## {entry.title}\n{entry.content}\n"
            entry_length = len(entry_text.split())
            
            if current_length + entry_length > max_tokens:
                break
            
            context_parts.append(entry_text)
            current_length += entry_length
        
        return "".join(context_parts)
    
    def list_domains(self) -> List[str]:
        """List all domains with stored documentation"""
        domains = []
        for domain_dir in self.base_dir.iterdir():
            if domain_dir.is_dir():
                # Convert back to domain format
                domain = domain_dir.name.replace("_", ".")
                domains.append(domain)
        return domains
    
    def get_domain_summary(self, domain: str) -> Dict[str, Any]:
        """Get summary of documentation for a domain"""
        domain_dir = self._domain_path(domain)
        index_path = domain_dir / "index.json"
        
        if not index_path.exists():
            return {"domain": domain, "entries": 0, "last_updated": None}
        
        with open(index_path, "r") as f:
            index = json.load(f)
        
        return {
            "domain": domain,
            "entries": len(index.get("entries", [])),
            "last_updated": index.get("updated_at"),
            "titles": [e["title"] for e in index.get("entries", [])],
        }
