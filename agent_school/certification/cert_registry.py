"""
Certification Registry - Manages stored certifications

Provides:
- Certification discovery and lookup
- Task pattern matching
- Loading/saving certifications
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .certification import Certification


class CertificationRegistry:
    """
    Registry for stored certifications.
    
    Structure on disk:
    workflows/certifications/
    ├── mail_google_com/
    │   ├── compose_email/
    │   │   └── certification.json
    │   └── search_inbox/
    │       └── certification.json
    ├── docs_google_com/
    │   └── create_document/
    │       └── certification.json
    └── registry.json
    """
    
    def __init__(self, base_dir: str = "workflows"):
        self.base_dir = Path(base_dir)
        self.certs_dir = self.base_dir / "certifications"
        self.certs_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Certification] = {}
        self._load_registry()
    
    def _registry_path(self) -> Path:
        return self.certs_dir / "registry.json"
    
    def _load_registry(self) -> None:
        """Load registry index from disk"""
        registry_path = self._registry_path()
        if registry_path.exists():
            with open(registry_path, "r") as f:
                self._registry = json.load(f)
        else:
            self._registry = {"certifications": {}, "last_updated": None}
    
    def _save_registry(self) -> None:
        """Save registry index to disk"""
        self._registry["last_updated"] = datetime.now().isoformat()
        with open(self._registry_path(), "w") as f:
            json.dump(self._registry, f, indent=2)
    
    def register(self, cert: Certification) -> str:
        """
        Register a certification.
        
        Returns:
            Path to saved certification
        """
        # Save to disk
        cert_path = cert.save(self.base_dir)
        
        # Update registry index
        key = f"{cert.domain}/{cert.name}"
        self._registry["certifications"][key] = {
            "name": cert.name,
            "domain": cert.domain,
            "task_description": cert.task_description,
            "task_patterns": cert.task_patterns,
            "version": cert.version,
            "created_at": cert.created_at.isoformat(),
            "path": str(cert_path),
        }
        
        self._save_registry()
        self._cache[key] = cert
        
        return str(cert_path)
    
    def get(self, domain: str, name: str) -> Optional[Certification]:
        """Get a certification by domain and name"""
        key = f"{domain}/{name}"
        
        # Check cache
        if key in self._cache:
            return self._cache[key]
        
        # Load from disk
        if key in self._registry["certifications"]:
            cert_path = Path(self._registry["certifications"][key]["path"])
            if cert_path.exists():
                cert = Certification.load(cert_path)
                self._cache[key] = cert
                return cert
        
        return None
    
    def find_for_task(
        self,
        task: str,
        domain: Optional[str] = None,
    ) -> Optional[Certification]:
        """
        Find a certification that matches a task.
        
        Uses task patterns to match.
        """
        task_lower = task.lower()
        
        for key, entry in self._registry["certifications"].items():
            # Filter by domain if specified
            if domain and entry["domain"] != domain:
                continue
            
            # Check task patterns
            for pattern in entry.get("task_patterns", []):
                try:
                    if re.search(pattern, task_lower):
                        return self.get(entry["domain"], entry["name"])
                except re.error:
                    # Invalid regex, try literal match
                    if pattern.lower() in task_lower:
                        return self.get(entry["domain"], entry["name"])
        
        return None
    
    def list_certifications(
        self,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all certifications, optionally filtered by domain"""
        certs = []
        
        for key, entry in self._registry["certifications"].items():
            if domain and entry["domain"] != domain:
                continue
            certs.append(entry)
        
        return certs
    
    def list_domains(self) -> List[str]:
        """List all domains with certifications"""
        domains = set()
        for entry in self._registry["certifications"].values():
            domains.add(entry["domain"])
        return sorted(domains)
    
    def delete(self, domain: str, name: str) -> bool:
        """Delete a certification"""
        key = f"{domain}/{name}"
        
        if key not in self._registry["certifications"]:
            return False
        
        # Remove from disk
        entry = self._registry["certifications"][key]
        cert_path = Path(entry["path"])
        if cert_path.exists():
            cert_path.unlink()
            # Remove parent dir if empty
            if cert_path.parent.exists() and not list(cert_path.parent.iterdir()):
                cert_path.parent.rmdir()
        
        # Remove from registry
        del self._registry["certifications"][key]
        self._save_registry()
        
        # Remove from cache
        if key in self._cache:
            del self._cache[key]
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall certification statistics"""
        total = len(self._registry["certifications"])
        domains = self.list_domains()
        
        total_successes = 0
        total_failures = 0
        
        for entry in self._registry["certifications"].values():
            cert = self.get(entry["domain"], entry["name"])
            if cert:
                total_successes += cert.success_count
                total_failures += cert.failure_count
        
        return {
            "total_certifications": total,
            "domains": domains,
            "domain_count": len(domains),
            "total_executions": total_successes + total_failures,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "overall_success_rate": (
                total_successes / (total_successes + total_failures)
                if (total_successes + total_failures) > 0 else 0
            ),
        }
