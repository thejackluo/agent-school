"""
Luma Event Scraper - Deterministic Workflow

This is a Layer 1 deterministic workflow that extracts events from Luma.
- NO LLM calls inside
- Takes structured input
- Returns structured output
- Uses browser automation (Playwright)
- Deterministic: same input -> same output
"""

from playwright.sync_api import sync_playwright, Page, Browser
from typing import List, Dict, Optional
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def fetch_luma_events(
    location: str = "San Francisco",
    radius: int = 5,
    keywords: Optional[List[str]] = None,
    max_results: int = 20
) -> List[Dict[str, str]]:
    """
    Fetch events from Luma using browser automation.

    This is a deterministic workflow - it does NOT use LLMs.
    It performs pure web scraping with Playwright.

    Args:
        location: Location to search (e.g., "San Francisco", "New York")
        radius: Radius in miles (not directly supported by Luma, used for filtering)
        keywords: List of keywords to filter (e.g., ["hip-hop", "startup"])
        max_results: Maximum number of results to return

    Returns:
        List of event dictionaries with structure:
        [
            {
                "title": "Event Title",
                "date": "2025-01-15",
                "time": "7:00 PM",
                "location": "San Francisco, CA",
                "url": "https://lu.ma/event-slug",
                "description": "Event description...",
                "host": "Organizer Name"
            }
        ]
    """
    logger.info(f"Starting Luma scraper for {location} with keywords: {keywords}")

    events = []

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # Navigate to Luma explore page
            search_location = location.replace(" ", "+")
            url = f"https://lu.ma/explore?location={search_location}"

            logger.info(f"Navigating to {url}")
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for events to load
            page.wait_for_selector('[data-testid="event-card"], .event-card, article', timeout=10000)

            # Extract events
            events = _extract_events_from_page(page, keywords, max_results)

            logger.info(f"Successfully extracted {len(events)} events")

        except Exception as e:
            logger.error(f"Error scraping Luma: {str(e)}")
            raise
        finally:
            browser.close()

    return events


def _extract_events_from_page(page: Page, keywords: Optional[List[str]], max_results: int) -> List[Dict[str, str]]:
    """
    Extract event data from the loaded page.

    Args:
        page: Playwright page object
        keywords: Keywords to filter by
        max_results: Maximum results to return

    Returns:
        List of event dictionaries
    """
    events = []

    # Try multiple selectors as Luma's DOM structure may vary
    event_selectors = [
        'article[data-testid="event-card"]',
        'div.event-card',
        'article',
        '[class*="event"]'
    ]

    event_elements = None
    for selector in event_selectors:
        try:
            event_elements = page.query_selector_all(selector)
            if event_elements and len(event_elements) > 0:
                logger.info(f"Found {len(event_elements)} events using selector: {selector}")
                break
        except:
            continue

    if not event_elements:
        logger.warning("No event elements found on page")
        return []

    for element in event_elements[:max_results * 2]:  # Get more than needed for filtering
        try:
            event_data = _extract_event_data(element, page)

            if event_data:
                # Filter by keywords if provided
                if keywords:
                    if _matches_keywords(event_data, keywords):
                        events.append(event_data)
                else:
                    events.append(event_data)

                # Stop if we have enough results
                if len(events) >= max_results:
                    break

        except Exception as e:
            logger.debug(f"Failed to extract event: {str(e)}")
            continue

    return events


def _extract_event_data(element, page: Page) -> Optional[Dict[str, str]]:
    """
    Extract data from a single event element.

    Args:
        element: Playwright element handle
        page: Playwright page object

    Returns:
        Event dictionary or None if extraction fails
    """
    try:
        # Extract title
        title = None
        title_selectors = ['h2', 'h3', '[class*="title"]', 'a']
        for selector in title_selectors:
            title_el = element.query_selector(selector)
            if title_el:
                title = title_el.inner_text().strip()
                if title and len(title) > 3:
                    break

        if not title:
            return None

        # Extract URL
        url = None
        link_el = element.query_selector('a[href*="lu.ma"]')
        if link_el:
            href = link_el.get_attribute('href')
            if href:
                url = href if href.startswith('http') else f"https://lu.ma{href}"

        # Extract date/time
        date = "TBD"
        time = "TBD"
        date_selectors = ['[class*="date"]', '[class*="time"]', 'time']
        for selector in date_selectors:
            date_el = element.query_selector(selector)
            if date_el:
                date_text = date_el.inner_text().strip()
                if date_text:
                    # Parse date
                    parsed_date = _parse_date(date_text)
                    if parsed_date:
                        date = parsed_date
                    # Parse time
                    parsed_time = _parse_time(date_text)
                    if parsed_time:
                        time = parsed_time

        # Extract location
        location = "Location TBD"
        location_selectors = ['[class*="location"]', '[class*="venue"]']
        for selector in location_selectors:
            loc_el = element.query_selector(selector)
            if loc_el:
                loc_text = loc_el.inner_text().strip()
                if loc_text and len(loc_text) > 2:
                    location = loc_text
                    break

        # Extract description
        description = ""
        desc_selectors = ['p', '[class*="description"]']
        for selector in desc_selectors:
            desc_el = element.query_selector(selector)
            if desc_el:
                desc_text = desc_el.inner_text().strip()
                if desc_text and len(desc_text) > 10:
                    description = desc_text[:200]  # Limit length
                    break

        # Extract host
        host = "Unknown Host"
        host_selectors = ['[class*="host"]', '[class*="organizer"]']
        for selector in host_selectors:
            host_el = element.query_selector(selector)
            if host_el:
                host_text = host_el.inner_text().strip()
                if host_text and len(host_text) > 2:
                    host = host_text
                    break

        return {
            "title": title,
            "date": date,
            "time": time,
            "location": location,
            "url": url or "https://lu.ma/explore",
            "description": description,
            "host": host
        }

    except Exception as e:
        logger.debug(f"Error extracting event data: {str(e)}")
        return None


def _parse_date(text: str) -> Optional[str]:
    """
    Parse date from text string.

    Args:
        text: Text containing date

    Returns:
        ISO date string (YYYY-MM-DD) or None
    """
    # Try to extract date patterns
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # ISO format
        r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)

    return None


def _parse_time(text: str) -> Optional[str]:
    """
    Parse time from text string.

    Args:
        text: Text containing time

    Returns:
        Time string or None
    """
    # Try to extract time patterns
    patterns = [
        r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)',  # 7:00 PM
        r'\d{1,2}\s*(?:AM|PM|am|pm)',  # 7 PM
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return None


def _matches_keywords(event: Dict[str, str], keywords: List[str]) -> bool:
    """
    Check if event matches any of the keywords.

    Args:
        event: Event dictionary
        keywords: List of keywords to match

    Returns:
        True if event matches any keyword
    """
    if not keywords:
        return True

    # Combine searchable text
    searchable = f"{event['title']} {event['description']} {event['location']}".lower()

    # Check if any keyword is in the text
    for keyword in keywords:
        if keyword.lower() in searchable:
            return True

    return False


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the scraper
    events = fetch_luma_events(
        location="San Francisco",
        radius=5,
        keywords=["tech", "startup"],
        max_results=5
    )

    print(f"\nFound {len(events)} events:")
    for i, event in enumerate(events, 1):
        print(f"\n{i}. {event['title']}")
        print(f"   Date: {event['date']} at {event['time']}")
        print(f"   Location: {event['location']}")
        print(f"   URL: {event['url']}")
