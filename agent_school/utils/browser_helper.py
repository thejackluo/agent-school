"""
Browser Automation Helper

Provides utilities for Playwright-based web scraping workflows.
"""

from typing import Optional, Dict, Any, List
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
import time


class BrowserHelper:
    """
    Helper class for browser automation with Playwright.

    Provides common patterns for web scraping workflows:
    - Browser instance management
    - Navigation with retries
    - Element waiting and interaction
    - Data extraction
    """

    def __init__(
        self,
        headless: bool = True,
        slow_mo: int = 0,
        timeout: int = 30000
    ):
        """
        Initialize browser helper.

        Args:
            headless: Run browser in headless mode
            slow_mo: Slow down operations by N milliseconds
            timeout: Default timeout for operations (ms)
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.timeout = timeout

        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def start(self, browser_type: str = "chromium"):
        """
        Start the browser.

        Args:
            browser_type: Type of browser (chromium, firefox, webkit)
        """
        self.playwright = sync_playwright().start()

        if browser_type == "chromium":
            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo
            )
        elif browser_type == "firefox":
            self.browser = self.playwright.firefox.launch(
                headless=self.headless,
                slow_mo=self.slow_mo
            )
        elif browser_type == "webkit":
            self.browser = self.playwright.webkit.launch(
                headless=self.headless,
                slow_mo=self.slow_mo
            )
        else:
            raise ValueError(f"Unknown browser type: {browser_type}")

        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.timeout)

    def close(self):
        """Close the browser and cleanup resources."""
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> bool:
        """
        Navigate to a URL with retry logic.

        Args:
            url: URL to navigate to
            wait_until: Wait strategy (domcontentloaded, load, networkidle)

        Returns:
            True if navigation successful
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.page.goto(url, wait_until=wait_until, timeout=self.timeout)
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        return False

    def wait_for_selector(
        self,
        selector: str,
        state: str = "visible",
        timeout: Optional[int] = None
    ) -> bool:
        """
        Wait for an element to appear.

        Args:
            selector: CSS selector
            state: Element state to wait for (attached, visible, hidden)
            timeout: Custom timeout (uses default if None)

        Returns:
            True if element found
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        try:
            self.page.wait_for_selector(
                selector,
                state=state,
                timeout=timeout or self.timeout
            )
            return True
        except:
            return False

    def extract_text(self, selector: str) -> Optional[str]:
        """
        Extract text content from an element.

        Args:
            selector: CSS selector

        Returns:
            Text content or None if not found
        """
        if not self.page:
            return None

        try:
            element = self.page.query_selector(selector)
            if element:
                return element.inner_text()
        except:
            pass

        return None

    def extract_multiple(self, selector: str) -> List[str]:
        """
        Extract text from multiple elements.

        Args:
            selector: CSS selector

        Returns:
            List of text content
        """
        if not self.page:
            return []

        try:
            elements = self.page.query_selector_all(selector)
            return [elem.inner_text() for elem in elements if elem]
        except:
            return []

    def extract_data(self, extraction_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract data using a mapping of field names to selectors.

        Args:
            extraction_map: Dict mapping field names to CSS selectors

        Returns:
            Dict of extracted data

        Example:
            data = browser.extract_data({
                "title": "h1.event-title",
                "date": ".event-date",
                "location": ".event-location"
            })
        """
        results = {}

        for field_name, selector in extraction_map.items():
            results[field_name] = self.extract_text(selector)

        return results

    def fill_form(self, form_data: Dict[str, str]) -> bool:
        """
        Fill a form with data.

        Args:
            form_data: Dict mapping CSS selectors to values

        Returns:
            True if all fields filled successfully

        Example:
            browser.fill_form({
                "input[name='search']": "hip-hop party",
                "input[name='location']": "San Francisco"
            })
        """
        if not self.page:
            return False

        try:
            for selector, value in form_data.items():
                self.page.fill(selector, value)
            return True
        except:
            return False

    def click(self, selector: str, wait_for_navigation: bool = False) -> bool:
        """
        Click an element.

        Args:
            selector: CSS selector
            wait_for_navigation: Wait for navigation after click

        Returns:
            True if click successful
        """
        if not self.page:
            return False

        try:
            if wait_for_navigation:
                with self.page.expect_navigation():
                    self.page.click(selector)
            else:
                self.page.click(selector)
            return True
        except:
            return False

    def screenshot(self, path: str, full_page: bool = False) -> bool:
        """
        Take a screenshot.

        Args:
            path: Path to save screenshot
            full_page: Capture full scrollable page

        Returns:
            True if screenshot saved successfully
        """
        if not self.page:
            return False

        try:
            self.page.screenshot(path=path, full_page=full_page)
            return True
        except:
            return False

    def evaluate(self, script: str) -> Any:
        """
        Execute JavaScript on the page.

        Args:
            script: JavaScript code to execute

        Returns:
            Result of script execution
        """
        if not self.page:
            return None

        try:
            return self.page.evaluate(script)
        except:
            return None


def quick_scrape(
    url: str,
    extraction_map: Dict[str, str],
    headless: bool = True
) -> Dict[str, Any]:
    """
    Quick utility function to scrape data from a single page.

    Args:
        url: URL to scrape
        extraction_map: Dict mapping field names to CSS selectors
        headless: Run browser in headless mode

    Returns:
        Extracted data

    Example:
        data = quick_scrape(
            "https://lu.ma/sf-events",
            {
                "title": "h1",
                "description": ".description"
            }
        )
    """
    with BrowserHelper(headless=headless) as browser:
        browser.navigate(url)
        return browser.extract_data(extraction_map)
