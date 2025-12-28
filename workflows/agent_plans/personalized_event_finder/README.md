# Personalized Event Finder - Agent Plan

This is a **Layer 2 Agent Plan** that demonstrates how to combine LLM intelligence with deterministic workflows.

## What is an Agent Plan?

An agent plan is:
- **Multi-step workflow** combining LLM steps and deterministic workflows
- Uses **LLMs for intelligence** (parsing, ranking, formatting)
- Uses **deterministic workflows for data** (scraping, APIs)
- **Orchestrates data flow** between steps
- Lives in **Layer 2** of Agent School architecture

## Architecture

```
User Input (Natural Language)
         ↓
   [Step 1: LLM Parse]
         ↓
   search_params: {location, keywords, size}
         ↓
   [Step 2: Deterministic Workflow]
         ↓
   raw_events: [{event1}, {event2}, ...]
         ↓
   [Step 3: LLM Filter]
         ↓
   filtered_events: [valid events only]
         ↓
   [Step 4: LLM Rank]
         ↓
   ranked_events: [sorted by relevance]
         ↓
   [Step 5: LLM Format]
         ↓
   Final Response (Conversational)
```

## How to Use

### Via Natural Language CLI

```bash
python main.py

> "Find hip-hop parties in San Francisco"
> "YC co-founder events in SF this weekend"
> "Small tech meetups in New York"
```

The Router (Layer 3) will:
1. Detect your intent to find events
2. Route to this agent plan
3. Execute all 5 steps
4. Return personalized results

### Programmatically

```python
from agent_school.core import Executor

executor = Executor()
result = executor.execute_plan(
    plan_name="personalized_event_finder",
    user_input="Find hip-hop parties in SF within 5 miles"
)

print(result["final_response"])
```

## Step-by-Step Breakdown

### Step 1: Parse User Preferences (LLM)
**Purpose**: Extract structured data from natural language

**Input**: `"Find hip-hop parties in SF within 5 miles"`

**Output**:
```json
{
  "location": "San Francisco",
  "keywords": ["hip-hop", "parties"],
  "size_preference": "medium",
  "max_results": 20
}
```

### Step 2: Fetch Events (Deterministic)
**Purpose**: Get raw event data using browser automation

**Uses**: `luma_scraper` deterministic workflow

**Input**: Structured params from Step 1

**Output**: Raw list of events from Luma

### Step 3: Validate and Filter (LLM)
**Purpose**: Clean and validate data

**Actions**:
- Remove events with missing critical data
- Ensure dates are in the future
- Remove duplicates
- Basic quality filtering

### Step 4: Rank by Relevance (LLM)
**Purpose**: Personalize results

**Ranking Factors**:
- Keyword match strength
- Event size matching preference
- Date proximity
- Venue quality indicators

**Output**: Top 10 most relevant events

### Step 5: Format Response (LLM)
**Purpose**: Create friendly, conversational output

**Features**:
- Acknowledges user's request
- Shows top 3-5 events
- Explains why each event matches
- Provides URLs
- Friendly closing

## Example Execution

**User Input**: `"Find YC co-founder events in San Francisco"`

**Step 1 Output**:
```json
{
  "location": "San Francisco",
  "keywords": ["YC", "co-founder", "startup", "founders"],
  "size_preference": "small",
  "max_results": 20
}
```

**Step 2 Output**: 47 raw events from Luma

**Step 3 Output**: 38 validated events (9 removed for missing data)

**Step 4 Output**: 10 ranked events (YC-related events ranked highest)

**Step 5 Output**:
```
I found some great co-founder events in San Francisco for you!

1. YC Co-Founder Matching - Jan 15, 2025 at 6:00 PM
   Location: YC Office, SF
   This is perfect for you - an official YC event specifically for finding co-founders!
   https://lu.ma/yc-cofounder-match

2. Startup Founder Meetup - Jan 17, 2025 at 7:00 PM
   Location: The Battery, SF
   Great networking opportunity with active founders in the YC community.
   https://lu.ma/founder-meetup-sf

...
```

## Design Principles

### 1. Clear Data Flow
Each step has explicit inputs and outputs defined in `plan.json`:
```json
{
  "input_from": ["raw_events", "search_params"],
  "output_var": "filtered_events"
}
```

### 2. Error Handling
Step 2 has `"error_handling": "continue"` so the plan doesn't fail if scraping partially fails.

### 3. LLM for Intelligence, Code for Data
- LLMs do: parsing, ranking, formatting (subjective tasks)
- Code does: web scraping, data fetching (objective tasks)

### 4. Cost Optimization
Only 4 LLM calls per execution. Step 2 (deterministic) is free.

## Customization

### Add More Data Sources

Modify Step 2 to call multiple scrapers:

```json
{
  "id": 2,
  "type": "parallel",
  "workflows": ["luma_scraper", "eventbrite_scraper", "meetup_scraper"],
  "output_var": "all_events"
}
```

### Change Ranking Criteria

Edit Step 4 prompt to emphasize different factors:
- Prioritize free events
- Prefer indoor/outdoor venues
- Filter by price range

### Adjust Output Format

Edit Step 5 prompt for different formats:
- JSON output for API consumers
- Markdown for documentation
- HTML for email newsletters

## Performance

- **Average Execution**: 10-15 seconds
- **Bottlenecks**:
  - Step 2: Browser automation (8-10 seconds)
  - Steps 3-5: LLM API latency (1-2 seconds each)

### Optimization Ideas

1. **Cache Step 2 results** for same location/keywords
2. **Run Steps 3-5 with smaller model** (faster, cheaper)
3. **Batch multiple user requests** for Step 2

## Cost Analysis

Estimated cost per execution: **$0.03 - $0.08**

Breakdown:
- Step 1: ~$0.01 (simple parsing)
- Step 2: $0.00 (deterministic)
- Step 3: ~$0.01-0.02 (validation)
- Step 4: ~$0.01-0.03 (ranking with context)
- Step 5: ~$0.01-0.02 (formatting)

Varies by:
- LLM provider (OpenAI vs Anthropic)
- Model choice (GPT-4 vs GPT-3.5)
- Number of events processed

## Testing

```bash
# Test the full plan
python -c "
from agent_school.core import Executor
executor = Executor()
result = executor.execute_plan(
    'personalized_event_finder',
    'Find tech events in SF'
)
print(result['final_response'])
"
```

## Integration with Layer 3 (Router)

The Router automatically routes natural language queries to this plan:

**User**: "Find hip-hop parties in SF"
**Router Intent**: `find_events`
**Router Action**: Execute `personalized_event_finder` plan

## Files

- `plan.json` - Step definitions and data flow
- `metadata.json` - Plan metadata
- `README.md` - This file

## Related

- [Luma Scraper (Layer 1)](../../deterministic/luma_scraper/)
- [Creating Agent Plans](../../../docs/creating-agent-plans.md)
- [Router System (Layer 3)](../../../agent_school/core/router.py)
