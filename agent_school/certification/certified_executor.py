"""
Certified Executor - Executes cached certifications deterministically

This is Phase C of the certification loop: taking a cached certification
and executing it step-by-step without LLM reasoning.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .certification import Certification, CertStep, ActionType, SelectorStrategy


@dataclass
class ExecutionResult:
    """Result of executing a certification"""
    certification_name: str
    success: bool
    steps_completed: int
    total_steps: int
    final_result: Optional[Any] = None
    error_message: Optional[str] = None
    error_step: Optional[int] = None
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "certification_name": self.certification_name,
            "success": self.success,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "final_result": self.final_result,
            "error_message": self.error_message,
            "error_step": self.error_step,
            "execution_time_ms": self.execution_time_ms,
        }


class CertifiedExecutor:
    """
    Executes cached certifications deterministically.
    
    This runs certification scripts step-by-step using Playwright
    (via Browser Use's browser) without any LLM calls.
    
    Cost: Near zero - just browser automation.
    """
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser = None
        self._page = None
    
    async def execute(
        self,
        cert: Certification,
        params: Dict[str, Any],
        timeout_seconds: int = 60,
    ) -> ExecutionResult:
        """
        Execute a certification with given parameters.
        
        Args:
            cert: The certification to execute
            params: Parameter values to bind (e.g., {"recipient": "user@example.com"})
            timeout_seconds: Overall execution timeout
            
        Returns:
            ExecutionResult with success status and any outputs
        """
        start_time = datetime.now()
        steps_completed = 0
        final_result = None
        error_message = None
        error_step = None
        
        try:
            # Bind parameters to steps
            bound_steps = cert.bind_parameters(params)
            
            # Initialize browser
            await self._init_browser()
            
            # Execute each step
            for step in bound_steps:
                try:
                    result = await asyncio.wait_for(
                        self._execute_step(step),
                        timeout=step.timeout_ms / 1000
                    )
                    steps_completed += 1
                    
                    # Capture extraction results
                    if step.action == ActionType.EXTRACT and result:
                        final_result = result
                        
                except asyncio.TimeoutError:
                    error_message = f"Step {step.id} timed out after {step.timeout_ms}ms"
                    error_step = step.id
                    if not step.optional:
                        break
                except Exception as e:
                    error_message = f"Step {step.id} failed: {str(e)}"
                    error_step = step.id
                    if not step.optional:
                        break
            
            success = steps_completed == len(bound_steps)
            
        except Exception as e:
            error_message = str(e)
            success = False
            
        finally:
            await self._close_browser()
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update certification stats
        if success:
            cert.record_success()
        else:
            cert.record_failure()
        
        return ExecutionResult(
            certification_name=cert.name,
            success=success,
            steps_completed=steps_completed,
            total_steps=len(cert.steps),
            final_result=final_result,
            error_message=error_message,
            error_step=error_step,
            execution_time_ms=execution_time,
        )
    
    async def _init_browser(self):
        """Initialize Playwright browser"""
        try:
            from playwright.async_api import async_playwright
            
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless
            )
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize browser: {e}")
    
    async def _close_browser(self):
        """Close browser and cleanup"""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if hasattr(self, '_playwright') and self._playwright:
                await self._playwright.stop()
        except:
            pass
    
    async def _execute_step(self, step: CertStep) -> Optional[Any]:
        """Execute a single certification step"""
        
        if step.action == ActionType.NAVIGATE:
            await self._page.goto(step.value or step.selector_value)
            return None
            
        elif step.action == ActionType.CLICK:
            element = await self._find_element(step)
            await element.click()
            return None
            
        elif step.action == ActionType.TYPE:
            element = await self._find_element(step)
            await element.fill(step.value or "")
            return None
            
        elif step.action == ActionType.SELECT:
            element = await self._find_element(step)
            await element.select_option(step.value)
            return None
            
        elif step.action == ActionType.WAIT:
            if step.selector_value:
                await self._page.wait_for_selector(
                    step.selector_value,
                    timeout=step.timeout_ms
                )
            else:
                await asyncio.sleep(step.timeout_ms / 1000)
            return None
            
        elif step.action == ActionType.SCROLL:
            await self._page.evaluate(f"window.scrollBy(0, {step.value or 300})")
            return None
            
        elif step.action == ActionType.EXTRACT:
            element = await self._find_element(step)
            return await element.text_content()
            
        elif step.action == ActionType.SCREENSHOT:
            path = step.value or "screenshot.png"
            await self._page.screenshot(path=path)
            return path
            
        elif step.action == ActionType.ASSERT:
            element = await self._find_element(step)
            text = await element.text_content()
            if step.value and step.value not in (text or ""):
                raise AssertionError(f"Expected '{step.value}' not found")
            return True
        
        return None
    
    async def _find_element(self, step: CertStep):
        """Find element using the step's selector strategy"""
        if not step.selector_value:
            raise ValueError(f"Step {step.id} has no selector")
        
        strategy = step.selector_strategy or SelectorStrategy.TEXT
        value = step.selector_value
        
        if strategy == SelectorStrategy.TEXT:
            # Find by visible text
            selector = f"text={value}"
        elif strategy == SelectorStrategy.ARIA_LABEL:
            selector = f"[aria-label='{value}']"
        elif strategy == SelectorStrategy.PLACEHOLDER:
            selector = f"[placeholder='{value}']"
        elif strategy == SelectorStrategy.ROLE:
            selector = f"role={value}"
        elif strategy == SelectorStrategy.CSS:
            selector = value
        elif strategy == SelectorStrategy.XPATH:
            selector = f"xpath={value}"
        else:
            selector = f"text={value}"
        
        # Wait for element and return
        await self._page.wait_for_selector(selector, timeout=step.timeout_ms)
        return self._page.locator(selector).first
    
    def bind_params_from_input(
        self,
        cert: Certification,
        user_input: str,
    ) -> Dict[str, Any]:
        """
        Extract parameters from natural language user input.
        
        This uses simple heuristics - for complex extraction,
        you might want to use an LLM.
        """
        params = {}
        
        # For now, basic extraction
        # TODO: Enhance with LLM-based extraction
        
        for param in cert.parameters:
            # Try to find quoted strings
            import re
            quoted = re.findall(r'"([^"]*)"', user_input)
            if quoted and param.name not in params:
                params[param.name] = quoted[0]
                quoted.pop(0)
            elif param.default is not None:
                params[param.name] = param.default
        
        return params
