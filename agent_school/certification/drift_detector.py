"""
Drift Detector - Detects and heals certification failures

When a certification fails due to UI changes (drift), this module:
1. Detects if the failure is due to drift vs user error
2. Triggers re-exploration to find the new path
3. Updates the certification with the new workflow
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .certification import Certification
from .explorer import ExplorationAgent, ExplorationResult
from .synthesizer import CertificationSynthesizer
from .certified_executor import ExecutionResult


class DriftError(Exception):
    """Exception indicating UI drift detected"""
    def __init__(self, message: str, step_id: int, original_error: str):
        super().__init__(message)
        self.step_id = step_id
        self.original_error = original_error


class DriftDetector:
    """
    Detects certification failures due to UI changes and triggers self-healing.
    
    Drift patterns:
    - ElementNotFound: UI element moved or was removed
    - Timeout waiting for selector: Element not appearing
    - Text changed: Button/label text was updated
    
    Self-healing process:
    1. Detect drift from error patterns
    2. Re-run exploration with updated docs
    3. Synthesize new certification
    4. Validate and replace old certification
    """
    
    # Error patterns that indicate drift
    DRIFT_PATTERNS = [
        r"element.*not found",
        r"timeout.*waiting.*selector",
        r"no element.*matches",
        r"locator.*resolved to.*elements",
        r"waiting for.*failed",
        r"target closed",
        r"navigation.*failed",
    ]
    
    # Error patterns that indicate user error (not drift)
    USER_ERROR_PATTERNS = [
        r"invalid.*param",
        r"missing.*required",
        r"authentication.*failed",
        r"permission.*denied",
        r"rate.*limit",
    ]
    
    def __init__(
        self,
        explorer: Optional[ExplorationAgent] = None,
        synthesizer: Optional[CertificationSynthesizer] = None,
    ):
        self.explorer = explorer or ExplorationAgent()
        self.synthesizer = synthesizer or CertificationSynthesizer()
    
    def detect_drift(
        self,
        result: ExecutionResult,
        cert: Certification,
    ) -> Tuple[bool, Optional[DriftError]]:
        """
        Analyze execution result to detect if failure was due to drift.
        
        Args:
            result: Result from failed certification execution
            cert: The certification that failed
            
        Returns:
            (is_drift, drift_error) tuple
        """
        if result.success:
            return False, None
        
        error = result.error_message or ""
        error_lower = error.lower()
        
        # Check for user error patterns first
        for pattern in self.USER_ERROR_PATTERNS:
            if re.search(pattern, error_lower):
                return False, None
        
        # Check for drift patterns
        for pattern in self.DRIFT_PATTERNS:
            if re.search(pattern, error_lower):
                return True, DriftError(
                    message=f"UI drift detected at step {result.error_step}",
                    step_id=result.error_step or 0,
                    original_error=error,
                )
        
        # Heuristic: if failed at a specific step with no clear user error,
        # and cert has been working before, likely drift
        if (result.error_step and 
            cert.success_count > 0 and
            cert.success_rate > 0.8):
            return True, DriftError(
                message=f"Suspected drift: step {result.error_step} failed after {cert.success_count} successes",
                step_id=result.error_step or 0,
                original_error=error,
            )
        
        return False, None
    
    async def heal(
        self,
        cert: Certification,
        drift_error: DriftError,
        params: Optional[Dict[str, Any]] = None,
    ) -> Certification:
        """
        Attempt to heal a certification by re-exploring.
        
        Args:
            cert: The broken certification
            drift_error: The drift error that was detected
            params: Optional params to use for re-exploration
            
        Returns:
            Updated certification (new version)
        """
        print(f"🔄 Self-healing certification '{cert.name}' (drift at step {drift_error.step_id})")
        
        # Re-explore the task
        result = await self.explorer.explore(
            task=cert.task_description,
            domain=cert.domain,
            max_steps=len(cert.steps) + 10,  # Allow some extra steps
        )
        
        if not result.success:
            raise RuntimeError(
                f"Self-healing failed: exploration unsuccessful. Error: {result.error_message}"
            )
        
        # Synthesize new certification
        new_cert = self.synthesizer.synthesize(
            action_log=result.action_log,
            task_patterns=cert.task_patterns,  # Keep same patterns
        )
        
        # Preserve metadata from old certification
        new_cert.name = cert.name
        new_cert.version = cert.version + 1
        new_cert.created_at = cert.created_at  # Keep original creation time
        new_cert.updated_at = datetime.now()
        
        # Note the healing in source info
        new_cert.source_exploration_id = result.exploration_id
        
        print(f"✅ Healed certification '{cert.name}' (v{cert.version} → v{new_cert.version})")
        
        return new_cert
    
    async def execute_with_healing(
        self,
        cert: Certification,
        params: Dict[str, Any],
        executor,  # CertifiedExecutor
        max_heal_attempts: int = 2,
    ) -> Tuple[ExecutionResult, Optional[Certification]]:
        """
        Execute a certification with automatic healing on drift.
        
        Args:
            cert: Certification to execute
            params: Parameters to pass
            executor: CertifiedExecutor instance
            max_heal_attempts: Maximum heal attempts before giving up
            
        Returns:
            (execution_result, updated_cert) - updated_cert is None if no healing needed
        """
        current_cert = cert
        heal_attempts = 0
        
        while True:
            result = await executor.execute(current_cert, params)
            
            if result.success:
                return result, current_cert if current_cert != cert else None
            
            # Check for drift
            is_drift, drift_error = self.detect_drift(result, current_cert)
            
            if not is_drift:
                # Not drift - return the failure
                return result, None
            
            if heal_attempts >= max_heal_attempts:
                print(f"❌ Max heal attempts ({max_heal_attempts}) reached")
                return result, None
            
            # Attempt healing
            try:
                current_cert = await self.heal(current_cert, drift_error, params)
                heal_attempts += 1
            except Exception as e:
                print(f"❌ Healing failed: {e}")
                return result, None
    
    def should_proactively_update(self, cert: Certification) -> bool:
        """
        Check if a certification should be proactively updated.
        
        Criteria:
        - High failure rate recently
        - Old and hasn't been validated
        - Known platform update
        """
        # High recent failure rate
        if cert.failure_count > 5 and cert.success_rate < 0.5:
            return True
        
        # Very old certification (30+ days without validation)
        if cert.last_success:
            days_since_success = (datetime.now() - cert.last_success).days
            if days_since_success > 30:
                return True
        
        return False
    
    def analyze_failure_trends(
        self,
        certs: List[Certification],
    ) -> Dict[str, Any]:
        """
        Analyze failure trends across certifications.
        
        Useful for identifying systemic issues (e.g., platform-wide UI update).
        """
        domain_failures = {}
        
        for cert in certs:
            domain = cert.domain
            if domain not in domain_failures:
                domain_failures[domain] = {
                    "total_certs": 0,
                    "failing_certs": 0,
                    "total_failures": 0,
                    "avg_success_rate": 0,
                }
            
            stats = domain_failures[domain]
            stats["total_certs"] += 1
            stats["total_failures"] += cert.failure_count
            
            if cert.success_rate < 0.8:
                stats["failing_certs"] += 1
        
        # Calculate averages
        for domain, stats in domain_failures.items():
            if stats["total_certs"] > 0:
                stats["failure_rate"] = stats["failing_certs"] / stats["total_certs"]
        
        return {
            "by_domain": domain_failures,
            "analyzed_at": datetime.now().isoformat(),
        }
