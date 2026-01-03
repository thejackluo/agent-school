"""
Certification - Data models for cached workflow certifications

A Certification is a parameterized workflow script that can be executed
deterministically without LLM reasoning, achieving near-zero inference cost.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import json
from pathlib import Path


class ActionType(str, Enum):
    """Types of browser actions that can be certified"""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    WAIT = "wait"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    ASSERT = "assert"


class SelectorStrategy(str, Enum):
    """How to find elements - prefer semantic over brittle CSS"""
    TEXT = "text"           # By visible text content
    ARIA_LABEL = "aria"     # By accessibility label
    PLACEHOLDER = "placeholder"
    ROLE = "role"           # By ARIA role
    CSS = "css"             # Last resort - brittle
    XPATH = "xpath"         # Last resort - very brittle


@dataclass
class ParamSpec:
    """Specification for a certification parameter"""
    name: str
    description: str
    type: Literal["string", "int", "float", "bool", "list"]
    required: bool = True
    default: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "required": self.required,
            "default": self.default,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParamSpec":
        return cls(**data)


@dataclass
class CertStep:
    """A single step in a certification workflow"""
    id: int
    action: ActionType
    description: str
    
    # Selector for element (if applicable)
    selector_strategy: Optional[SelectorStrategy] = None
    selector_value: Optional[str] = None
    
    # Value to input/check (can contain {param} placeholders)
    value: Optional[str] = None
    
    # Wait conditions
    wait_for: Optional[str] = None
    timeout_ms: int = 10000
    
    # Error handling
    optional: bool = False
    fallback_step: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action.value,
            "description": self.description,
            "selector_strategy": self.selector_strategy.value if self.selector_strategy else None,
            "selector_value": self.selector_value,
            "value": self.value,
            "wait_for": self.wait_for,
            "timeout_ms": self.timeout_ms,
            "optional": self.optional,
            "fallback_step": self.fallback_step,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CertStep":
        data = data.copy()
        data["action"] = ActionType(data["action"])
        if data.get("selector_strategy"):
            data["selector_strategy"] = SelectorStrategy(data["selector_strategy"])
        return cls(**data)


@dataclass
class Certification:
    """
    A cached workflow certification.
    
    Contains all information needed to execute a browser workflow
    deterministically, without LLM reasoning.
    """
    name: str
    domain: str  # e.g., "mail.google.com", "docs.google.com"
    task_description: str
    task_patterns: List[str]  # Regex patterns to match user requests
    
    steps: List[CertStep]
    parameters: List[ParamSpec]
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    # Tracking
    success_count: int = 0
    failure_count: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    # Source information (for debugging/retraining)
    source_exploration_id: Optional[str] = None
    source_docs: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "task_description": self.task_description,
            "task_patterns": self.task_patterns,
            "steps": [s.to_dict() for s in self.steps],
            "parameters": [p.to_dict() for p in self.parameters],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "source_exploration_id": self.source_exploration_id,
            "source_docs": self.source_docs,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Certification":
        data = data.copy()
        data["steps"] = [CertStep.from_dict(s) for s in data["steps"]]
        data["parameters"] = [ParamSpec.from_dict(p) for p in data["parameters"]]
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("last_success"):
            data["last_success"] = datetime.fromisoformat(data["last_success"])
        if data.get("last_failure"):
            data["last_failure"] = datetime.fromisoformat(data["last_failure"])
        return cls(**data)
    
    def save(self, base_dir: Path) -> Path:
        """Save certification to disk"""
        cert_dir = base_dir / "certifications" / self.domain.replace(".", "_") / self.name
        cert_dir.mkdir(parents=True, exist_ok=True)
        
        cert_path = cert_dir / "certification.json"
        with open(cert_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return cert_path
    
    @classmethod
    def load(cls, cert_path: Path) -> "Certification":
        """Load certification from disk"""
        with open(cert_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def record_success(self) -> None:
        """Record a successful execution"""
        self.success_count += 1
        self.last_success = datetime.now()
        self.updated_at = datetime.now()
    
    def record_failure(self) -> None:
        """Record a failed execution"""
        self.failure_count += 1
        self.last_failure = datetime.now()
        self.updated_at = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def bind_parameters(self, params: Dict[str, Any]) -> List[CertStep]:
        """
        Create a copy of steps with parameters bound to actual values.
        
        Example:
            Step with value="Hello {message}" and params={"message": "World"}
            Returns step with value="Hello World"
        """
        bound_steps = []
        for step in self.steps:
            bound_step = CertStep(
                id=step.id,
                action=step.action,
                description=step.description,
                selector_strategy=step.selector_strategy,
                selector_value=step.selector_value,
                value=step.value.format(**params) if step.value else None,
                wait_for=step.wait_for,
                timeout_ms=step.timeout_ms,
                optional=step.optional,
                fallback_step=step.fallback_step,
            )
            bound_steps.append(bound_step)
        return bound_steps
