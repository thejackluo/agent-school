"""
Certification Module - Agent Certification System

This module enables agents to:
1. Learn workflows through autonomous exploration (ReAct)
2. Synthesize successful runs into cached certifications
3. Execute certifications deterministically
4. Self-heal when certifications break due to UI changes
"""

from .certification import Certification, CertStep, ParamSpec, ActionType, SelectorStrategy
from .explorer import ExplorationAgent, ExplorationResult
from .synthesizer import CertificationSynthesizer
from .certified_executor import CertifiedExecutor
from .drift_detector import DriftDetector
from .doc_ingester import DocIngester
from .doc_store import DocStore
from .action_logger import ActionLogger
from .cert_registry import CertificationRegistry

__all__ = [
    "Certification",
    "CertStep", 
    "ParamSpec",
    "ActionType",
    "SelectorStrategy",
    "ExplorationAgent",
    "ExplorationResult",
    "CertificationSynthesizer",
    "CertifiedExecutor",
    "DriftDetector",
    "DocIngester",
    "DocStore",
    "ActionLogger",
    "CertificationRegistry",
]

