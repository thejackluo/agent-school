# Luma Event Scraper - Deterministic Workflow

This is a **Layer 1 Deterministic Workflow** for Agent School. It extracts events from Luma (lu.ma) using browser automation.

## What is a Deterministic Workflow?

A deterministic workflow is:
- **Pure Python code** with NO LLM calls inside
- Takes **structured input** (typed parameters)
- Returns **structured output** (list of dictionaries)
- **Deterministic**: Same input always produces the same output (given the same website state)
- Uses either **API calls** or **browser automation** (this one uses browser)

## Usage

### As Python Function

```python
from workflows.deterministic.luma_scraper.workflow import fetch_luma_events

# Find hip-hop parties in San Francisco
events = fetch_luma_events(
    location="San Francisco",
    radius=5,
    keywords=["hip-hop", "party"],
    max_results=10
)

for event in events:
    print(f"{event['title']} - {event['date']} at {event['location']}")
```

### Via Agent School CLI

```bash
# Interactive mode (natural language)
python main.py

> "Find tech startup events in San Francisco"

# The Router will:
# 1. Detect you want to find events
# 2. Use luma_scraper workflow
# 3. Execute and return results
```

### Direct Execution

```bash
cd workflows/deterministic/luma_scraper
python workflow.py
```

## Input Schema

```json
{
  "location": "string (required)",
  "radius": "integer (optional, default: 5)",
  "keywords": "array of strings (optional)",
  "max_results": "integer (optional, default: 20)"
}
```

## Output Schema

Returns an array of events:

```json
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
```

## How It Works

1. **Launches Playwright browser** in headless mode
2. **Navigates to Luma explore** with location parameter
3. **Waits for events to load** on the page
4. **Extracts event data** using multiple fallback selectors (robust to DOM changes)
5. **Filters by keywords** if provided
6. **Returns structured data** ready for LLM consumption

## Design Principles

### Why Browser Automation?

Luma's API is a paid service, so we use browser automation to access the public explore page. This is a common pattern when:
- API is paid/restricted
- No API exists
- Website has better data than API

### Robust Selector Strategy

The scraper uses **multiple fallback selectors** because website DOMs change frequently:

```python
event_selectors = [
    'article[data-testid="event-card"]',  # Preferred
    'div.event-card',                      # Fallback
    'article',                             # Generic
    '[class*="event"]'                     # Broadest
]
```

### Error Handling

- Graceful failures at element level (continues to next event)
- Logs warnings for debugging
- Returns partial results if some events fail to parse

## Integration with Layer 2 (Agent Plans)

This workflow can be used in agent plans:

```json
{
  "name": "personalized_event_finder",
  "steps": [
    {
      "id": 1,
      "type": "llm",
      "action": "parse_preferences",
      "prompt": "Extract location and keywords from: {user_input}",
      "output_var": "params"
    },
    {
      "id": 2,
      "type": "deterministic",
      "workflow": "luma_scraper",
      "input_from": "params",
      "output_var": "raw_events"
    },
    {
      "id": 3,
      "type": "llm",
      "action": "rank_and_format",
      "prompt": "Rank these events by relevance: {raw_events}",
      "output_var": "final_response"
    }
  ]
}
```

## Limitations

1. **Rate Limiting**: Luma may rate limit excessive requests
2. **DOM Changes**: Website structure may change, requiring selector updates
3. **Playwright Required**: Must have Playwright and browsers installed
4. **Client-Side Filtering**: Radius parameter does approximate filtering

## Testing

```bash
# Run the workflow directly
python workflow.py

# Or via pytest
pytest tests/ -k luma
```

## Maintenance

When Luma changes their website structure:

1. Inspect new DOM structure
2. Update selectors in `_extract_events_from_page()`
3. Test with various locations and keywords
4. Update version in metadata.json

## Example Use Cases

1. **Event Discovery**: "Find YC co-founder events in SF this weekend"
2. **Personal Recommendations**: "Hip-hop parties I might like"
3. **Data Aggregation**: Combine with other event sources
4. **Market Research**: Track types of events in different cities

## Files

- `workflow.py` - Main scraper implementation
- `input_schema.json` - Input validation schema
- `output_schema.json` - Output format specification
- `metadata.json` - Workflow metadata and configuration
- `README.md` - This file

## Related

- [Agent School Documentation](../../../README.md)
- [Creating Deterministic Workflows](../../../docs/creating-workflows.md)
- [Agent Plans](../../agent_plans/)
