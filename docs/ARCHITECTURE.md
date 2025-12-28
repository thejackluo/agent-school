# Agent School - System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                            │
│  "Find hip-hop parties in SF within 5 miles with hot girls" │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Plan Generator (File 2)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LLM (GPT-4 / Claude)                                 │  │
│  │  - Parse user intent                                  │  │
│  │  - Identify required steps                            │  │
│  │  - Generate structured plan                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  Output: Step-by-step execution plan                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Workflow Generator (File 1)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LLM (GPT-4 / Claude)                                 │  │
│  │  - Analyze available methods (API vs Browser)         │  │
│  │  - Generate executable workflow code                  │  │
│  │  - Ensure deterministic execution                     │  │
│  └──────────────────────────────────────────────────────┘  │
│  Output: Executable Python workflow                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Function Creator (File 3)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Combines Plan + Workflow                             │  │
│  │  - Wraps in MCP-compatible function                   │  │
│  │  - Adds error handling and validation                 │  │
│  │  - Generates function documentation                   │  │
│  └──────────────────────────────────────────────────────┘  │
│  Output: MCP Server Function                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server Interface                      │
│  - Exposes generated functions to AI agents                 │
│  - Handles authentication and authorization                 │
│  - Manages function registry                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      AI Agent Calls                          │
│  Agent can now call: fetch_sf_hiphop_parties()              │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Plan Generator (`backend/generators/plan_generator.py`)

**Purpose**: Convert natural language queries into structured, step-by-step execution plans.

**Responsibilities**:
- Parse user intent and extract key parameters (location, category, filters, radius, etc.)
- Identify required data sources (Luma API, web scraping, etc.)
- Generate logical sequence of operations
- Handle edge cases and validation requirements
- Output structured plan in JSON or Python dict format

**Example Input**:
```
"Find hip-hop parties with hot girls in SF within 5 miles"
```

**Example Output**:
```python
{
    "task": "fetch_sf_hiphop_parties",
    "steps": [
        {
            "step": 1,
            "action": "validate_location",
            "params": {"city": "SF", "radius_miles": 5}
        },
        {
            "step": 2,
            "action": "search_events",
            "params": {
                "category": ["music", "nightlife", "hip-hop"],
                "location": "San Francisco, CA",
                "radius_miles": 5,
                "keywords": ["hip-hop", "party"]
            }
        },
        {
            "step": 3,
            "action": "filter_results",
            "params": {"min_attendees": 50, "venue_type": "nightclub"}
        },
        {
            "step": 4,
            "action": "return_events",
            "params": {"format": "json", "fields": ["name", "date", "location", "url"]}
        }
    ],
    "fallback": "Use browser automation if API fails"
}
```

**LLM Prompt Strategy**:
```
You are a workflow planning expert. Given a user query for event search,
generate a detailed step-by-step plan to accomplish the task.

Consider:
1. Data source availability (API vs browser automation)
2. Required authentication/credentials
3. Parameter extraction from natural language
4. Error handling and fallback strategies
5. Output format requirements

Generate a structured JSON plan with clear steps.
```

### 2. Workflow Generator (`backend/workflows/workflow_generator.py`)

**Purpose**: Generate executable Python code that implements the plan.

**Responsibilities**:
- Convert plan steps into working Python code
- Choose optimal execution method (API calls vs Playwright browser automation)
- Handle authentication and API rate limiting
- Ensure deterministic, repeatable execution
- Add robust error handling
- Generate self-contained, executable workflow

**Execution Modes**:

#### Mode A: API-Based Workflow
Used when Luma API is available and has required endpoints.

```python
async def fetch_sf_hiphop_parties(radius_miles: int = 5):
    """
    Fetch hip-hop parties in SF within specified radius.
    Generated by Agent School on 2024-01-15.
    """
    import os
    import httpx
    from datetime import datetime, timedelta

    # Authentication
    api_key = os.getenv("LUMA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    # Build query parameters
    params = {
        "location": "San Francisco, CA",
        "radius": radius_miles,
        "category": "music,nightlife",
        "keywords": "hip-hop,party",
        "start_date": datetime.now().isoformat(),
        "end_date": (datetime.now() + timedelta(days=30)).isoformat()
    }

    # API call
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.lu.ma/events/search",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        events = response.json()

    # Filter and format results
    filtered_events = [
        {
            "name": event["name"],
            "date": event["start_date"],
            "location": event["venue"]["name"],
            "url": event["url"],
            "attendees": event["attendee_count"]
        }
        for event in events["data"]
        if event.get("attendee_count", 0) > 50
    ]

    return filtered_events
```

#### Mode B: Browser Automation Workflow
Used when API is unavailable or paid, requiring web scraping.

```python
async def fetch_sf_hiphop_parties_browser(radius_miles: int = 5):
    """
    Fetch hip-hop parties in SF using browser automation.
    Generated by Agent School on 2024-01-15.
    """
    from playwright.async_api import async_playwright
    import json

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to Luma
        await page.goto("https://lu.ma/explore")

        # Set location
        await page.click('input[placeholder="Location"]')
        await page.fill('input[placeholder="Location"]', "San Francisco, CA")
        await page.press('input[placeholder="Location"]', "Enter")

        # Set search keywords
        await page.fill('input[placeholder="Search events"]', "hip-hop party")
        await page.press('input[placeholder="Search events"]', "Enter")

        # Wait for results
        await page.wait_for_selector('.event-card')

        # Extract event data
        events = await page.evaluate('''() => {
            return Array.from(document.querySelectorAll('.event-card')).map(card => ({
                name: card.querySelector('.event-name')?.textContent,
                date: card.querySelector('.event-date')?.textContent,
                location: card.querySelector('.event-location')?.textContent,
                url: card.querySelector('a')?.href,
                attendees: parseInt(card.querySelector('.attendee-count')?.textContent || '0')
            }));
        }''')

        await browser.close()

        # Filter by radius and attendee count
        filtered_events = [e for e in events if e['attendees'] > 50]
        return filtered_events
```

**LLM Prompt Strategy**:
```
You are an expert Python developer. Given this execution plan, generate
production-ready Python code that implements it.

Requirements:
1. Use async/await for I/O operations
2. Include comprehensive error handling
3. Add type hints and docstrings
4. Make code deterministic and idempotent
5. Use environment variables for credentials
6. Choose optimal approach: API calls (preferred) or Playwright browser automation

Plan: {plan_json}

Available tools: httpx, playwright, beautifulsoup4, json, datetime
Generate complete, executable Python function.
```

### 3. Function Creator (`backend/functions/function_creator.py`)

**Purpose**: Combine plan and workflow into a production-ready MCP server function.

**Responsibilities**:
- Wrap generated workflow in MCP-compatible interface
- Add function metadata (name, description, parameters, return type)
- Implement input validation and sanitization
- Add comprehensive error handling and logging
- Generate OpenAPI-style documentation
- Register function in MCP server

**Output Structure**:
```python
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging

class EventSearchRequest(BaseModel):
    """Request schema for event search."""
    query: str = Field(..., description="Natural language event search query")
    location: Optional[str] = Field(None, description="City or address")
    radius_miles: int = Field(5, ge=1, le=50, description="Search radius in miles")
    max_results: int = Field(20, ge=1, le=100, description="Maximum results to return")

class Event(BaseModel):
    """Event data schema."""
    name: str
    date: str
    location: str
    url: str
    attendees: int

class EventSearchResponse(BaseModel):
    """Response schema for event search."""
    events: List[Event]
    total_count: int
    query: str
    execution_time_ms: float

async def search_luma_events(request: EventSearchRequest) -> EventSearchResponse:
    """
    Search Luma for events matching natural language query.

    This function was dynamically generated by Agent School to handle
    event searches on Luma.co. It combines LLM-generated planning and
    workflow execution to provide accurate, deterministic results.

    Args:
        request: Event search request with query and filters

    Returns:
        EventSearchResponse with matching events

    Raises:
        ValueError: If query is invalid or location cannot be geocoded
        HTTPException: If Luma API/website is unavailable

    Examples:
        >>> result = await search_luma_events(EventSearchRequest(
        ...     query="hip-hop parties in SF",
        ...     radius_miles=5
        ... ))
        >>> print(f"Found {result.total_count} events")
    """
    import time
    start_time = time.time()

    logger = logging.getLogger(__name__)
    logger.info(f"Searching events: {request.query}")

    try:
        # Step 1: Generate plan from query
        plan = await plan_generator.generate_plan(request.query)
        logger.debug(f"Generated plan: {plan}")

        # Step 2: Execute workflow
        events = await workflow_generator.execute_workflow(
            plan=plan,
            location=request.location,
            radius_miles=request.radius_miles
        )

        # Step 3: Format and validate results
        validated_events = [Event(**event) for event in events[:request.max_results]]

        execution_time = (time.time() - start_time) * 1000

        return EventSearchResponse(
            events=validated_events,
            total_count=len(validated_events),
            query=request.query,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Event search failed: {str(e)}", exc_info=True)
        raise

# MCP Server Registration
MCP_FUNCTION_METADATA = {
    "name": "search_luma_events",
    "description": "Search Luma for events using natural language queries",
    "parameters": EventSearchRequest.schema(),
    "returns": EventSearchResponse.schema(),
    "examples": [
        {
            "query": "hip-hop parties in SF within 5 miles",
            "expected": "Returns list of hip-hop/nightlife events in San Francisco"
        },
        {
            "query": "YC co-founder events",
            "expected": "Returns professional networking events for founders"
        }
    ]
}
```

### 4. Test Runner (`backend/tests/test_runner.py`)

**Purpose**: Validate that generated functions work correctly.

**Test Categories**:

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test full workflow end-to-end
3. **Validation Tests**: Ensure outputs match expected schemas
4. **Performance Tests**: Check execution time and resource usage
5. **Regression Tests**: Verify existing functions still work after updates

**Example Test**:
```python
import pytest
from backend.functions.function_creator import search_luma_events, EventSearchRequest

@pytest.mark.asyncio
async def test_sf_hiphop_events():
    """Test searching for hip-hop parties in SF."""
    request = EventSearchRequest(
        query="hip-hop parties with hot girls in SF within 5 miles",
        radius_miles=5,
        max_results=10
    )

    response = await search_luma_events(request)

    # Assertions
    assert response.total_count > 0, "Should find at least one event"
    assert all(event.location.lower().find('san francisco') >= 0
               for event in response.events), "All events should be in SF"
    assert response.execution_time_ms < 5000, "Should complete within 5 seconds"
    assert all(event.attendees > 0 for event in response.events), "Events should have attendees"

@pytest.mark.asyncio
async def test_yc_founder_events():
    """Test searching for YC founder events."""
    request = EventSearchRequest(
        query="YC co-founder events or private events",
        max_results=10
    )

    response = await search_luma_events(request)

    assert response.total_count > 0
    assert any('yc' in event.name.lower() or 'founder' in event.name.lower()
               for event in response.events)
```

## Data Flow Diagram

```
User Query
    │
    ├─→ Plan Generator (LLM)
    │       │
    │       ├─→ Extract Parameters (location, radius, category)
    │       ├─→ Identify Data Source (API vs Browser)
    │       └─→ Generate Step Plan (JSON)
    │
    ├─→ Workflow Generator (LLM)
    │       │
    │       ├─→ Choose Execution Mode (API/Browser)
    │       ├─→ Generate Python Code
    │       └─→ Add Error Handling
    │
    ├─→ Function Creator
    │       │
    │       ├─→ Wrap in MCP Function
    │       ├─→ Add Validation & Schemas
    │       ├─→ Generate Documentation
    │       └─→ Register in MCP Server
    │
    └─→ Test Runner
            │
            ├─→ Execute Test Cases
            ├─→ Validate Output
            └─→ Return Success/Failure
```

## Technology Integration

### LangChain & LangGraph

LangChain orchestrates the LLM calls, while LangGraph manages the state machine:

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph

# Define state
class WorkflowState:
    query: str
    plan: dict
    workflow_code: str
    function_code: str
    test_results: dict

# Create graph
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("generate_plan", plan_generator_node)
workflow.add_node("generate_workflow", workflow_generator_node)
workflow.add_node("create_function", function_creator_node)
workflow.add_node("run_tests", test_runner_node)

# Define edges
workflow.add_edge("generate_plan", "generate_workflow")
workflow.add_edge("generate_workflow", "create_function")
workflow.add_edge("create_function", "run_tests")

# Compile
app = workflow.compile()
```

### MCP Server Integration

The generated functions are exposed via Model Context Protocol:

```python
from mcp import MCPServer

server = MCPServer()

# Register generated function
server.register_function(
    name="search_luma_events",
    func=search_luma_events,
    metadata=MCP_FUNCTION_METADATA
)

# Start server
server.run(host="0.0.0.0", port=8000)
```

## Error Handling Strategy

### Layered Error Handling

1. **Input Validation**: Pydantic models validate request data
2. **LLM Failures**: Retry with exponential backoff
3. **API Failures**: Fallback to browser automation
4. **Browser Failures**: Retry with different selectors
5. **Output Validation**: Ensure response matches schema
6. **Logging**: Comprehensive logging at each layer

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def execute_with_retry(workflow_func, *args, **kwargs):
    """Execute workflow with automatic retry."""
    try:
        return await workflow_func(*args, **kwargs)
    except APIError:
        # Fallback to browser automation
        return await browser_fallback(*args, **kwargs)
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise
```

## Security Considerations

1. **API Key Management**: All keys in environment variables, never in code
2. **Input Sanitization**: Validate all user inputs to prevent injection
3. **Rate Limiting**: Respect API rate limits, add exponential backoff
4. **Browser Security**: Run Playwright in sandboxed mode
5. **Output Filtering**: Sanitize outputs to prevent XSS if serving to web
6. **Authentication**: Validate MCP server requests are authorized

## Performance Optimization

1. **Async Operations**: Use asyncio for concurrent I/O
2. **Caching**: Cache LLM responses for repeated queries
3. **Connection Pooling**: Reuse HTTP/browser connections
4. **Parallel Execution**: Run independent steps in parallel
5. **Lazy Loading**: Only load heavy dependencies when needed

## Scalability Considerations

- **Horizontal Scaling**: MCP server can run multiple instances behind load balancer
- **Function Registry**: Store generated functions in database for persistence
- **Rate Limiting**: Add rate limits per user/API key
- **Monitoring**: Track function execution time, success rate, error types
- **Version Control**: Version generated functions for rollback capability

## Extension Points

The architecture is designed to be extensible:

1. **New Data Sources**: Add new workflow generators for different platforms
2. **New Execution Modes**: Add support for different automation tools
3. **Custom LLMs**: Swap OpenAI for other providers
4. **Enhanced Validation**: Add custom validation rules
5. **Monitoring & Analytics**: Integrate observability tools

## Next Steps

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for the development roadmap.
