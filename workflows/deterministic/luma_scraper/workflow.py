"""
Luma Event Scraper - Deterministic Workflow

This is a Layer 1 deterministic workflow that extracts events from Luma.
- NO LLM calls inside
- Takes structured input
- Returns structured output
- Uses browser automation (Playwright)
- Deterministic: same input -> same output

Uses the working approach: scrapes city pages (/sf, /nyc) and category pages (/tech, /ai)
rather than the /explore endpoint which has limited results.
"""

from playwright.sync_api import sync_playwright, Page, Browser
from typing import List, Dict, Optional
from pathlib import Path
import hashlib
import json
import time
import re
import logging

logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_TTL_SECONDS = 3600  # Cache for 1 hour
RATE_LIMIT_SECONDS = 1.0  # Wait between requests

# Known Luma categories
LUMA_CATEGORIES = ["tech", "ai", "crypto", "arts", "climate", "fitness", "wellness", "food"]

# City slug mapping
CITY_SLUGS = {
    "san francisco": "sf",
    "sf": "sf",
    "new york": "nyc",
    "nyc": "nyc",
    "los angeles": "la",
    "la": "la",
    "seattle": "seattle",
    "austin": "austin",
    "boston": "boston",
    "chicago": "chicago",
    "miami": "miami",
}

# US states for location parsing
US_STATES = [
    "California", "New York", "Texas", "Florida", "Washington", 
    "Oregon", "Colorado", "Massachusetts", "Illinois", "Georgia",
    "Arizona", "Nevada", "North Carolina", "Virginia", "Pennsylvania",
    "Ohio", "Michigan", "Tennessee", "Minnesota", "Utah", "DC"
]

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)


def _get_cache_path(key: str) -> Path:
    """Get cache file path for a given key."""
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{hash_key}.json"


def _get_cached(key: str) -> Optional[dict]:
    """Get cached data if it exists and is not expired."""
    cache_path = _get_cache_path(key)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if time.time() - data.get("timestamp", 0) < CACHE_TTL_SECONDS:
                return data.get("value")
        except:
            pass
    return None


def _set_cache(key: str, value) -> None:
    """Cache a value with timestamp."""
    cache_path = _get_cache_path(key)
    try:
        cache_path.write_text(json.dumps({
            "timestamp": time.time(),
            "value": value
        }, indent=2))
    except:
        pass


def _rate_limit():
    """Simple rate limiting."""
    time.sleep(RATE_LIMIT_SECONDS)


def _create_browser():
    """Create a Playwright browser instance."""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_default_timeout(30000)
    return browser, page, playwright


def _get_city_slug(location: str) -> str:
    """Convert location name to Luma city slug."""
    location_lower = location.lower().strip()
    return CITY_SLUGS.get(location_lower, "sf")


def fetch_events_from_page(page_slug: str) -> List[Dict]:
    """
    Fetch events from a Luma page (city or category).
    
    Args:
        page_slug: The page slug (e.g., "sf", "tech", "ai")
        
    Returns:
        List of event dictionaries
    """
    cache_key = f"page:{page_slug}"
    cached = _get_cached(cache_key)
    if cached:
        logger.info(f"[Cache hit] {page_slug}")
        return cached
    
    _rate_limit()
    
    browser, page, playwright = _create_browser()
    events = []
    
    try:
        url = f"https://lu.ma/{page_slug}"
        logger.info(f"Navigating to {url}...")
        page.goto(url, wait_until="networkidle")
        page.wait_for_load_state("domcontentloaded")
        
        # Scroll to trigger lazy loading
        for _ in range(3):
            page.evaluate("window.scrollBy(0, 500)")
            page.wait_for_timeout(500)
        page.wait_for_timeout(1000)
        
        # Find event links
        event_links = page.query_selector_all("a.event-link, a[class*='event-link']")
        if not event_links:
            event_links = page.query_selector_all("a[href^='/'][aria-label]")
        
        seen_slugs = set()
        for link in event_links:
            href = link.get_attribute("href")
            
            if not href or href.startswith("/#"):
                continue
            if href in ["/discover", "/pricing", "/signin", "/create", "/user"]:
                continue
            if href.startswith("/user/"):
                continue
            
            slug = href.lstrip("/")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            
            name = link.get_attribute("aria-label") or link.inner_text().strip()
            if name:
                events.append({
                    "title": name.strip(),
                    "url": f"https://lu.ma{href}",
                    "slug": slug,
                    "source": page_slug,
                })
        
        logger.info(f"Found {len(events)} events from /{page_slug}")
        _set_cache(cache_key, events)
        
    except Exception as e:
        logger.error(f"Error fetching from {page_slug}: {e}")
    finally:
        browser.close()
        playwright.stop()
    
    return events


def fetch_event_details(event_slug: str) -> Dict:
    """
    Fetch full details for a specific event.
    
    Args:
        event_slug: The event slug (e.g., "tiat14")
        
    Returns:
        Event dictionary with full details
    """
    cache_key = f"event:{event_slug}"
    cached = _get_cached(cache_key)
    if cached:
        logger.info(f"[Cache hit] {event_slug}")
        return cached
    
    _rate_limit()
    
    browser, page, playwright = _create_browser()
    event = {"slug": event_slug, "url": f"https://lu.ma/{event_slug}"}
    
    try:
        page.goto(event["url"], wait_until="networkidle")
        page.wait_for_load_state("domcontentloaded")
        
        # Get title
        title_el = page.query_selector("h1, [class*='event-title'], [class*='title']")
        if title_el:
            event["title"] = title_el.inner_text().strip()
        
        body_text = page.inner_text("body")
        
        # Get date
        date_pattern = r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(\w+\s+\d{1,2},\s+\d{4})'
        date_match = re.search(date_pattern, body_text)
        if date_match:
            event["date"] = date_match.group(0)
        
        # Get time
        time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM))\s*[-–]\s*(\d{1,2}:\d{2}\s*(?:AM|PM))'
        time_match = re.search(time_pattern, body_text, re.IGNORECASE)
        if time_match:
            event["time"] = time_match.group(0)
        
        # Get location
        state_pattern = "|".join(US_STATES)
        location_pattern = rf'\b([A-Z][a-zA-Z ]+,\s*(?:{state_pattern}))\b'
        loc_match = re.search(location_pattern, body_text)
        if loc_match:
            event["location"] = loc_match.group(1).strip()
        
        # Get description
        about_el = page.query_selector("[class*='description'], [class*='about']")
        if about_el:
            event["description"] = about_el.inner_text().strip()[:500]
        
        # Get host
        host_el = page.query_selector("[class*='host'] a, [class*='organizer'] a")
        if host_el:
            event["host"] = host_el.inner_text().strip()
        
        logger.info(f"Extracted details for: {event.get('title', event_slug)}")
        _set_cache(cache_key, event)
        
    except Exception as e:
        logger.error(f"Error fetching details for {event_slug}: {e}")
        event["error"] = str(e)
    finally:
        browser.close()
        playwright.stop()
    
    return event


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
    
    ALWAYS does comprehensive search - scrapes city page + all 8 category pages.

    Args:
        location: Location to search (e.g., "San Francisco", "New York")
        radius: Radius in miles (not used, kept for API compatibility)
        keywords: List of keywords to filter (e.g., ["hip-hop", "art"])
        max_results: Maximum number of results to return

    Returns:
        List of event dictionaries
    """
    logger.info(f"Starting Luma scraper for {location} with keywords: {keywords}")
    
    city_slug = _get_city_slug(location)
    all_events = []
    seen_slugs = set()
    
    # ALWAYS do comprehensive search - scrape city page + ALL category pages
    sources = [city_slug] + LUMA_CATEGORIES
    logger.info(f"Comprehensive search: scraping {len(sources)} sources: {sources}")
    
    # Fetch from each source
    for source in sources:
        events = fetch_events_from_page(source)
        for event in events:
            slug = event.get("slug", "")
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                all_events.append(event)
    
    logger.info(f"Found {len(all_events)} total events from {len(sources)} sources")
    
    # Filter by keywords if provided
    if keywords:
        filtered = []
        for event in all_events:
            # Get details to check keywords
            details = fetch_event_details(event["slug"])
            searchable = f"{details.get('title', '')} {details.get('description', '')}".lower()
            
            if any(kw.lower() in searchable for kw in keywords):
                event.update(details)
                filtered.append(event)
                
                if len(filtered) >= max_results:
                    break
        
        all_events = filtered
    
    # NOTE: We intentionally do NOT filter by location here.
    # The LLM in subsequent steps handles location relevance based on user intent.
    # This allows:
    # - "SF events" to include nearby Oakland/Berkeley if LLM deems relevant
    # - "Downtown SF only" to exclude Oakland if user is specific
    # - Global scalability without maintaining city lists
    
    # Fetch details for events that don't have them yet
    for event in all_events[:max_results]:
        if "location" not in event or "time" not in event:
            details = fetch_event_details(event["slug"])
            event.update(details)
    
    return all_events[:max_results]


# Alias for compatibility
def search_events(query: str, location: str = "San Francisco", max_results: int = 10) -> List[Dict]:
    """
    Search for events matching a query.
    
    Args:
        query: Search query (used as keywords)
        location: Location to search in
        max_results: Maximum results
        
    Returns:
        List of matching events
    """
    keywords = query.split() if query else None
    return fetch_luma_events(
        location=location,
        keywords=keywords,
        max_results=max_results
    )


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the scraper
    events = fetch_luma_events(
        location="San Francisco",
        keywords=["art"],
        max_results=5
    )

    print(f"\nFound {len(events)} events:")
    for i, event in enumerate(events, 1):
        print(f"\n{i}. {event.get('title', 'Unknown')}")
        print(f"   Date: {event.get('date', 'TBD')}")
        print(f"   Location: {event.get('location', 'TBD')}")
        print(f"   URL: {event.get('url', 'N/A')}")
