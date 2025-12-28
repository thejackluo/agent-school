"""
Luma Browser Navigation Module

This module provides functions to fetch Luma events using browser automation (Playwright).
It scrapes the website to extract event data without requiring an API key.

Usage:
    from luma_browser import get_events_from_calendar, get_events_from_city, get_event_details

    # Get events from a specific calendar/organizer
    events = get_events_from_calendar("tiat")

    # Get events from a city
    events = get_events_from_city("sf")

    # Get full details for a specific event
    event = get_event_details("tiat14")
"""

from playwright.sync_api import sync_playwright, Page, Browser
from typing import Optional
from pathlib import Path
import hashlib
import json
import time
import re
import os

# Configuration
CACHE_DIR = Path(__file__).parent / ".luma_cache"
CACHE_TTL_SECONDS = 3600  # Cache for 1 hour
RATE_LIMIT_SECONDS = 1.0  # Wait between requests

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
        pass  # Silently fail on cache errors


def _rate_limit():
    """Simple rate limiting - sleep between requests."""
    time.sleep(RATE_LIMIT_SECONDS)




def _parse_date_time(date_str: str, time_str: str) -> dict:
    """Parse date and time strings into structured data."""
    return {
        "date_text": date_str.strip() if date_str else None,
        "time_text": time_str.strip() if time_str else None,
    }


def _extract_event_from_card(card_element) -> dict:
    """Extract event data from an event card element on listing pages."""
    try:
        # Get the link element for URL
        link = card_element.query_selector("a")
        url = link.get_attribute("href") if link else None
        
        # Get event name from the card
        name_el = card_element.query_selector("h3, [class*='title'], [class*='name']")
        name = name_el.inner_text() if name_el else None
        
        # Get date/time text
        date_el = card_element.query_selector("[class*='date'], [class*='time'], time")
        date_text = date_el.inner_text() if date_el else None
        
        # Get image if available
        img = card_element.query_selector("img")
        cover_url = img.get_attribute("src") if img else None
        
        return {
            "name": name,
            "url": f"https://lu.ma{url}" if url and url.startswith("/") else url,
            "date_text": date_text,
            "cover_url": cover_url,
        }
    except Exception as e:
        print(f"Error extracting event from card: {e}")
        return {}


def _create_browser() -> tuple[Browser, Page]:
    """Create a Playwright browser instance."""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_default_timeout(30000)  # 30 second timeout
    return browser, page, playwright


def get_events_from_calendar(calendar_slug: str) -> list[dict]:
    """
    Fetch all events from a Luma calendar/organizer page.
    
    Args:
        calendar_slug: The calendar identifier (e.g., "tiat" for lu.ma/tiat)
        
    Returns:
        List of event dictionaries with basic info
    """
    browser, page, playwright = _create_browser()
    events = []
    
    try:
        url = f"https://lu.ma/{calendar_slug}"
        print(f"Navigating to {url}...")
        page.goto(url, wait_until="networkidle")
        
        # Wait for page to load
        page.wait_for_load_state("domcontentloaded")
        
        # Scroll down to trigger lazy loading of event cards
        for _ in range(3):
            page.evaluate("window.scrollBy(0, 500)")
            page.wait_for_timeout(500)
        
        # Wait a bit for events to render
        page.wait_for_timeout(1000)
        
        # Look for event links with the 'event-link' class
        # Luma uses: <a class="event-link content-link" aria-label="Event Title" href="/slug">
        event_links = page.query_selector_all("a.event-link, a[class*='event-link']")
        
        if not event_links:
            # Fallback: try to find any links that look like events
            event_links = page.query_selector_all("a[href^='/'][aria-label]")
        
        seen_urls = set()
        for link in event_links:
            href = link.get_attribute("href")
            
            # Filter out non-event links
            if not href or href.startswith("/#"):
                continue
            if href in ["/discover", "/pricing", "/signin", "/create", "/user"]:
                continue
            if href.startswith("/user/"):
                continue
            if href in seen_urls:
                continue
                
            seen_urls.add(href)
            
            # Get event name from aria-label (preferred) or inner text
            name = link.get_attribute("aria-label")
            if not name:
                name = link.inner_text().strip()
            
            # Clean up name (remove time prefix if present like "6:30 PM Event Name")
            if name:
                events.append({
                    "name": name.strip(),
                    "url": f"https://lu.ma{href}",
                    "slug": href.lstrip("/"),
                })
        
        print(f"Found {len(events)} events from calendar '{calendar_slug}'")
        
    except Exception as e:
        print(f"Error fetching events from calendar {calendar_slug}: {e}")
    finally:
        browser.close()
        playwright.stop()
    
    return events


def get_events_from_city(city_slug: str) -> list[dict]:
    """
    Fetch events from a city page.
    
    Args:
        city_slug: The city identifier (e.g., "sf", "nyc", "la")
        
    Returns:
        List of event dictionaries with basic info
    """
    # City pages have the same structure as calendar pages
    return get_events_from_calendar(city_slug)


def get_event_details(event_slug: str, use_cache: bool = True) -> dict:
    """
    Fetch full details for a specific event.
    
    Args:
        event_slug: The event identifier (e.g., "tiat14" for lu.ma/tiat14)
        use_cache: Whether to use cached results (default True)
        
    Returns:
        Dictionary with full event details
    """
    cache_key = f"event_details:{event_slug}"
    
    # Check cache first
    if use_cache:
        cached = _get_cached(cache_key)
        if cached:
            print(f"[Cache hit] {event_slug}")
            return cached
    
    # Rate limit before making request
    _rate_limit()
    
    browser, page, playwright = _create_browser()
    event = {}
    
    try:
        url = f"https://lu.ma/{event_slug}"
        print(f"Navigating to {url}...")
        page.goto(url, wait_until="networkidle")
        
        # Wait for page to load
        page.wait_for_load_state("domcontentloaded")
        
        # Extract event data
        event["url"] = url
        event["slug"] = event_slug
        
        # Get title - usually in h1 or prominent heading
        title_el = page.query_selector("h1, [class*='event-title'], [class*='title']")
        if title_el:
            event["name"] = title_el.inner_text().strip()
        
        # Get all text content for parsing
        body_text = page.inner_text("body")
        
        # Look for date patterns
        # Luma uses formats like "Sunday, January 4, 2026"
        date_pattern = r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(\w+\s+\d{1,2},\s+\d{4})'
        date_match = re.search(date_pattern, body_text)
        if date_match:
            event["date"] = date_match.group(0)
        
        # Look for time patterns like "2:00 PM - 3:00 PM"
        time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM))\s*[-–]\s*(\d{1,2}:\d{2}\s*(?:AM|PM))'
        time_match = re.search(time_pattern, body_text, re.IGNORECASE)
        if time_match:
            event["start_time"] = time_match.group(1)
            event["end_time"] = time_match.group(2)
            event["time"] = time_match.group(0)
        
        # Look for location using "City, State" pattern with known US states
        # CSS selectors don't work well on Luma - they pick up extra text
        us_states = [
            "California", "New York", "Texas", "Florida", "Washington", 
            "Oregon", "Colorado", "Massachusetts", "Illinois", "Georgia",
            "Arizona", "Nevada", "North Carolina", "Virginia", "Pennsylvania",
            "Ohio", "Michigan", "Tennessee", "Minnesota", "Utah", "DC"
        ]
        state_pattern = "|".join(us_states)
        # Match "City Name, State" pattern - use \w and spaces only (no newlines)
        # Pattern: Capital letter, then word chars/spaces, comma, state name
        location_pattern = rf'\b([A-Z][a-zA-Z ]+,\s*(?:{state_pattern}))\b'
        loc_match = re.search(location_pattern, body_text)
        if loc_match:
            event["location"] = loc_match.group(1).strip()
        
        # Get description/about section
        about_el = page.query_selector("[class*='description'], [class*='about']")
        if about_el:
            event["description"] = about_el.inner_text().strip()
        
        # Get cover image
        cover_img = page.query_selector("img[src*='cdn'], img[class*='cover'], img[class*='banner']")
        if cover_img:
            event["cover_url"] = cover_img.get_attribute("src")
        
        # Get host info
        host_el = page.query_selector("[class*='host'] a, [class*='organizer'] a")
        if host_el:
            event["host"] = host_el.inner_text().strip()
            event["host_url"] = host_el.get_attribute("href")
        
        # Try to get categories/tags
        tag_els = page.query_selector_all("a[href*='/tech'], a[href*='/ai'], a[href*='/arts'], a[href*='/crypto']")
        if tag_els:
            event["categories"] = [tag.inner_text().strip() for tag in tag_els if tag.inner_text().strip()]
        
        print(f"Extracted details for event: {event.get('name', event_slug)}")
        
        # Cache the result if successful
        if "error" not in event:
            _set_cache(cache_key, event)
        
    except Exception as e:
        print(f"Error fetching event details for {event_slug}: {e}")
        event["error"] = str(e)
    finally:
        browser.close()
        playwright.stop()
    
    return event


# Known Luma categories and cities
LUMA_CATEGORIES = ["tech", "ai", "crypto", "arts", "climate", "fitness", "wellness", "food"]
LUMA_CITIES = ["sf", "nyc", "la", "seattle", "austin", "boston", "chicago", "miami", "denver", "toronto"]


def get_events_from_multiple_sources(
    sources: list[str],
    include_details: bool = False,
    filter_location: Optional[str] = None
) -> list[dict]:
    """
    Fetch events from multiple Luma pages and deduplicate.
    
    Args:
        sources: List of page slugs (e.g., ["sf", "tech", "ai"])
        include_details: If True, fetch full details for each event (slower)
        filter_location: If set, only return events in this location (e.g., "San Francisco")
        
    Returns:
        Deduplicated list of events from all sources
    """
    all_events = []
    seen_slugs = set()
    
    for source in sources:
        print(f"\n--- Fetching from /{source} ---")
        events = get_events_from_calendar(source)
        
        for event in events:
            slug = event.get("slug", "")
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                event["source"] = source  # Track where we found it
                all_events.append(event)
    
    print(f"\nTotal unique events from {len(sources)} sources: {len(all_events)}")
    
    # If filtering by location, we need to get details to check location
    if filter_location:
        print(f"\nFiltering by location: '{filter_location}'...")
        filtered_events = []
        
        for event in all_events:
            slug = event.get("slug")
            if slug:
                details = get_event_details(slug)
                location = details.get("location", "")
                
                # Check if location matches (case-insensitive partial match)
                if filter_location.lower() in location.lower():
                    event.update(details)
                    filtered_events.append(event)
                    print(f"  ✓ {event.get('name', slug)[:50]} - {location}")
                else:
                    print(f"  ✗ {event.get('name', slug)[:50]} - {location}")
        
        print(f"\nEvents in '{filter_location}': {len(filtered_events)}")
        return filtered_events
    
    # Optionally fetch full details for each event
    if include_details:
        detailed_events = []
        for event in all_events:
            slug = event.get("slug")
            if slug:
                details = get_event_details(slug)
                detailed_events.append({**event, **details})
        return detailed_events
    
    return all_events


def get_all_events_in_location(
    location: str,
    include_city_page: bool = True,
    categories: Optional[list[str]] = None
) -> list[dict]:
    """
    Get ALL events in a specific location by scraping multiple sources.
    
    This addresses the issue that /sf only shows curated events, not all SF events.
    We scrape /sf + all category pages and filter by location.
    
    Args:
        location: Location to filter by (e.g., "San Francisco")
        include_city_page: Whether to include the city page (e.g., /sf)
        categories: List of categories to check. Defaults to all known categories.
        
    Returns:
        List of all events in the specified location
    """
    # Determine which city slug to use
    city_slug = None
    location_lower = location.lower()
    if "san francisco" in location_lower or location_lower == "sf":
        city_slug = "sf"
        location = "San Francisco"
    elif "new york" in location_lower or location_lower in ["nyc", "ny"]:
        city_slug = "nyc"
        location = "New York"
    elif "los angeles" in location_lower or location_lower == "la":
        city_slug = "la"
        location = "Los Angeles"
    elif "seattle" in location_lower:
        city_slug = "seattle"
        location = "Seattle"
    # Add more cities as needed
    
    # Build list of sources to scrape
    sources = []
    if include_city_page and city_slug:
        sources.append(city_slug)
    
    # Add category pages
    cats = categories or LUMA_CATEGORIES
    sources.extend(cats)
    
    print(f"=" * 60)
    print(f"Getting ALL events in: {location}")
    print(f"Scraping {len(sources)} sources: {sources}")
    print(f"=" * 60)
    
    return get_events_from_multiple_sources(
        sources=sources,
        filter_location=location
    )


def get_luma_events(
    calendar: Optional[str] = None,
    city: Optional[str] = None,
    event_slug: Optional[str] = None,
    include_details: bool = False,
    comprehensive: bool = False,
    categories: Optional[list[str]] = None
) -> list[dict]:
    """
    Main function to fetch Luma events.
    
    Args:
        calendar: Calendar/organizer slug (e.g., "tiat")
        city: City slug (e.g., "sf", "nyc")
        event_slug: Specific event slug for single event details
        include_details: If True, fetch full details for each event (slower)
        comprehensive: If True, fetch from city + all categories for complete results
        categories: Specific categories to search (used with comprehensive=True)
        
    Returns:
        List of event dictionaries
    """
    events = []
    
    if event_slug:
        # Single event lookup
        event = get_event_details(event_slug)
        return [event] if event else []
    
    if comprehensive and city:
        # Comprehensive mode: get ALL events in the city from multiple sources
        location_map = {
            "sf": "San Francisco",
            "nyc": "New York",
            "la": "Los Angeles",
            "seattle": "Seattle",
            "austin": "Austin",
            "boston": "Boston",
        }
        location = location_map.get(city, city)
        return get_all_events_in_location(location, categories=categories)
    
    if calendar:
        events = get_events_from_calendar(calendar)
    elif city:
        events = get_events_from_city(city)
    else:
        # Default to SF if nothing specified
        events = get_events_from_city("sf")
    
    # Optionally fetch full details for each event
    if include_details and events:
        detailed_events = []
        for event in events:
            slug = event.get("slug") or event.get("url", "").split("/")[-1]
            if slug:
                details = get_event_details(slug)
                detailed_events.append({**event, **details})
        return detailed_events
    
    return events


# CLI for testing
if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("Luma Browser Navigation Test")
    print("=" * 50)
    
    # Test 1: Get events from SF (curated only)
    print("\n[Test 1] Fetching CURATED events from San Francisco...")
    sf_events = get_events_from_city("sf")
    print(f"Found {len(sf_events)} curated SF events")
    if sf_events:
        print(f"First event: {sf_events[0]['name']}")
    
    # Test 2: Get events from a calendar
    print("\n[Test 2] Fetching events from 'tiat' calendar...")
    tiat_events = get_events_from_calendar("tiat")
    print(f"Found {len(tiat_events)} events")
    if tiat_events:
        print(f"First event: {tiat_events[0]['name']}")
    
    # Test 3: Get single event details
    print("\n[Test 3] Fetching details for 'tiat14'...")
    event_details = get_event_details("tiat14")
    print(f"Event: {event_details.get('name')}")
    print(f"Date: {event_details.get('date')}")
    print(f"Location: {event_details.get('location')}")
    
    # Test 4: COMPREHENSIVE mode - get ALL SF events from multiple sources
    # NOTE: This is slow as it scrapes many pages and checks each event's location
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        print("\n[Test 4] COMPREHENSIVE: Fetching ALL events in San Francisco...")
        print("(This scrapes multiple category pages - will take a few minutes)")
        
        # Just test with 2 categories for speed
        all_sf_events = get_luma_events(
            city="sf",
            comprehensive=True,
            categories=["tech", "ai"]  # Limit categories for faster test
        )
        print(f"\nFound {len(all_sf_events)} total SF events from comprehensive search")
        
        # Show events found on category pages but NOT on /sf
        sf_slugs = {e.get("slug") for e in sf_events}
        new_events = [e for e in all_sf_events if e.get("slug") not in sf_slugs]
        print(f"Events found on category pages but NOT on /sf: {len(new_events)}")
        for e in new_events[:5]:
            print(f"  - {e.get('name', 'Unknown')[:60]}")
    
    print("\n" + "=" * 50)
    print("Tests complete!")
    print("\nRun with --comprehensive flag to test multi-source fetching:")
    print("  uv run python luma_browser.py --comprehensive")
