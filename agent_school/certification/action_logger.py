"""
Action Logger - Captures and structures browser actions from Browser Use

Extracts action history from Browser Use agent runs and converts them
into a structured format suitable for certification synthesis.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import json


@dataclass
class Action:
    """A single captured browser action"""
    timestamp: datetime
    action_type: str  # e.g., "click", "type", "navigate"
    description: str  # Human-readable description
    
    # Target element info
    element_text: Optional[str] = None
    element_role: Optional[str] = None
    element_selector: Optional[str] = None
    
    # Action details
    input_value: Optional[str] = None
    url: Optional[str] = None
    
    # Result
    success: bool = True
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "description": self.description,
            "element_text": self.element_text,
            "element_role": self.element_role,
            "element_selector": self.element_selector,
            "input_value": self.input_value,
            "url": self.url,
            "success": self.success,
            "error_message": self.error_message,
            "screenshot_path": self.screenshot_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ActionLog:
    """Complete log of actions from an exploration session"""
    exploration_id: str
    task: str
    domain: str
    start_time: datetime
    end_time: Optional[datetime]
    actions: List[Action]
    success: bool
    final_result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploration_id": self.exploration_id,
            "task": self.task,
            "domain": self.domain,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "actions": [a.to_dict() for a in self.actions],
            "success": self.success,
            "final_result": self.final_result,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionLog":
        data = data.copy()
        data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        data["actions"] = [Action.from_dict(a) for a in data["actions"]]
        return cls(**data)


class ActionLogger:
    """
    Extracts and structures actions from Browser Use agent history.
    
    Browser Use maintains an internal history of actions taken.
    This class converts that into our Action format for synthesis.
    """
    
    def extract_actions(self, browser_history: Any) -> List[Action]:
        """
        Extract actions from Browser Use agent history.
        
        Args:
            browser_history: The history object from agent.run()
            
        Returns:
            List of structured Action objects
        """
        actions = []
        
        # Browser Use history contains action results
        # We need to parse the history format
        if hasattr(browser_history, 'history'):
            for entry in browser_history.history:
                action = self._parse_history_entry(entry)
                if action:
                    actions.append(action)
        elif isinstance(browser_history, list):
            for entry in browser_history:
                action = self._parse_history_entry(entry)
                if action:
                    actions.append(action)
        
        return actions
    
    def _parse_history_entry(self, entry: Any) -> Optional[Action]:
        """Parse a single history entry into an Action"""
        try:
            # Handle different entry formats
            if hasattr(entry, 'model_output'):
                # Browser Use AgentHistoryStep
                return self._parse_agent_step(entry)
            elif isinstance(entry, dict):
                return self._parse_dict_entry(entry)
            else:
                return None
        except Exception as e:
            # Log but don't fail on parse errors
            print(f"Warning: Failed to parse history entry: {e}")
            return None
    
    def _parse_agent_step(self, step: Any) -> Optional[Action]:
        """Parse Browser Use AgentHistoryStep"""
        action_type = "unknown"
        description = ""
        element_text = None
        input_value = None
        url = None
        
        # Extract from model output
        if hasattr(step, 'model_output') and step.model_output:
            output = step.model_output
            if hasattr(output, 'action'):
                action_info = output.action
                action_type = getattr(action_info, 'name', 'unknown')
                
                # Extract action-specific details
                if hasattr(action_info, 'text'):
                    input_value = action_info.text
                if hasattr(action_info, 'url'):
                    url = action_info.url
                if hasattr(action_info, 'element'):
                    element_text = str(action_info.element)
        
        # Get description from result
        if hasattr(step, 'result') and step.result:
            if hasattr(step.result, 'extracted_content'):
                description = step.result.extracted_content or ""
        
        if not description:
            description = f"{action_type}: {input_value or element_text or url or 'element'}"
        
        return Action(
            timestamp=datetime.now(),  # Browser Use doesn't always have timestamps
            action_type=action_type,
            description=description,
            element_text=element_text,
            input_value=input_value,
            url=url,
            success=True,
        )
    
    def _parse_dict_entry(self, entry: Dict[str, Any]) -> Optional[Action]:
        """Parse dictionary-format entry"""
        return Action(
            timestamp=datetime.fromisoformat(entry.get("timestamp", datetime.now().isoformat())),
            action_type=entry.get("action_type", "unknown"),
            description=entry.get("description", ""),
            element_text=entry.get("element_text"),
            element_role=entry.get("element_role"),
            element_selector=entry.get("element_selector"),
            input_value=entry.get("input_value"),
            url=entry.get("url"),
            success=entry.get("success", True),
            error_message=entry.get("error_message"),
        )
    
    def create_action_log(
        self,
        exploration_id: str,
        task: str,
        domain: str,
        browser_history: Any,
        success: bool,
        final_result: Optional[str] = None,
    ) -> ActionLog:
        """
        Create a complete action log from an exploration session.
        
        Args:
            exploration_id: Unique ID for this exploration
            task: The task that was performed
            domain: Target domain (e.g., "mail.google.com")
            browser_history: Browser Use agent history
            success: Whether the task was completed successfully
            final_result: Optional final extracted result
            
        Returns:
            Complete ActionLog
        """
        actions = self.extract_actions(browser_history)
        
        start_time = actions[0].timestamp if actions else datetime.now()
        end_time = actions[-1].timestamp if actions else datetime.now()
        
        return ActionLog(
            exploration_id=exploration_id,
            task=task,
            domain=domain,
            start_time=start_time,
            end_time=end_time,
            actions=actions,
            success=success,
            final_result=final_result,
        )
    
    def save_log(self, log: ActionLog, base_dir: str) -> str:
        """Save action log to disk for later analysis"""
        from pathlib import Path
        
        log_dir = Path(base_dir) / "exploration_logs" / log.domain.replace(".", "_")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = log_dir / f"{log.exploration_id}.json"
        
        try:
            # Use custom encoder to handle non-serializable objects
            with open(log_path, "w") as f:
                json.dump(log.to_dict(), f, indent=2, default=self._json_safe_serialize)
        except Exception as e:
            print(f"[WARN] Failed to save action log: {e}")
            # Save a minimal log on failure
            with open(log_path, "w") as f:
                json.dump({
                    "exploration_id": log.exploration_id,
                    "task": log.task,
                    "domain": log.domain,
                    "success": log.success,
                    "error": f"Failed to serialize full log: {e}"
                }, f, indent=2)
        
        return str(log_path)
    
    def _json_safe_serialize(self, obj):
        """Safely serialize objects that aren't JSON-compatible"""
        if callable(obj):
            return f"<{type(obj).__name__}>"
        if hasattr(obj, '__dict__'):
            return str(obj)
        return f"<non-serializable: {type(obj).__name__}>"

