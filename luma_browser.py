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
import json
import re


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


def get_event_details(event_slug: str) -> dict:
    """
    Fetch full details for a specific event.
    
    Args:
        event_slug: The event identifier (e.g., "tiat14" for lu.ma/tiat14)
        
    Returns:
        Dictionary with full event details
    """
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
        
        # Look for location
        location_el = page.query_selector("[class*='location'], [class*='address']")
        if location_el:
            event["location"] = location_el.inner_text().strip()
        else:
            # Try to find "San Francisco, California" type patterns
            location_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z][a-z]+)'
            loc_match = re.search(location_pattern, body_text)
            if loc_match:
                event["location"] = loc_match.group(1)
        
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
        
    except Exception as e:
        print(f"Error fetching event details for {event_slug}: {e}")
        event["error"] = str(e)
    finally:
        browser.close()
        playwright.stop()
    
    return event


def get_luma_events(
    calendar: Optional[str] = None,
    city: Optional[str] = None,
    event_slug: Optional[str] = None,
    include_details: bool = False
) -> list[dict]:
    """
    Main function to fetch Luma events.
    
    Args:
        calendar: Calendar/organizer slug (e.g., "tiat")
        city: City slug (e.g., "sf", "nyc")
        event_slug: Specific event slug for single event details
        include_details: If True, fetch full details for each event (slower)
        
    Returns:
        List of event dictionaries
    """
    events = []
    
    if event_slug:
        # Single event lookup
        event = get_event_details(event_slug)
        return [event] if event else []
    
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
    
    # Test 1: Get events from SF
    print("\n[Test 1] Fetching events from San Francisco...")
    sf_events = get_events_from_city("sf")
    print(f"Found {len(sf_events)} events")
    if sf_events:
        print(f"First event: {sf_events[0]}")
    
    # Test 2: Get events from a calendar
    print("\n[Test 2] Fetching events from 'tiat' calendar...")
    tiat_events = get_events_from_calendar("tiat")
    print(f"Found {len(tiat_events)} events")
    if tiat_events:
        print(f"First event: {tiat_events[0]}")
    
    # Test 3: Get single event details
    print("\n[Test 3] Fetching details for 'tiat14'...")
    event_details = get_event_details("tiat14")
    print(f"Event details: {json.dumps(event_details, indent=2)}")
    
    print("\n" + "=" * 50)
    print("Tests complete!")
