# Agent School

**AI-Powered Workflow Generation System**

Agent School dynamically creates workflows using LLMs. Instead of hardcoding scrapers and automation, describe what you want and get working code instantly.

## The Big Idea

Traditional approach: Write code for every data source
```python
def fetch_luma_events(): ...
def fetch_eventbrite_events(): ...
def fetch_twitter_data(): ...
# Manually maintain hundreds of scrapers
```

**Agent School approach:** Describe once, generate forever
```bash
$ python main.py

You: "Find hip-hop parties in SF within 5 miles"
Agent School: [Generates code, executes, returns results]
```

## Three-Layer Architecture

Agent School uses a clean three-layer design:

### Layer 1: Deterministic Workflows
**Pure Python code with NO LLM calls**

- Input: Structured parameters `{"location": "SF", "radius": 5}`
- Output: Structured data `[{"title": "...", "date": "..."}]`
- Method: API calls OR browser automation
- Lives in: `workflows/deterministic/`

**Example:**
```python
# workflows/deterministic/luma_scraper/workflow.py
def fetch_luma_events(location: str, radius: int, keywords: list) -> list[dict]:
    # Pure Playwright automation code
    # Runs the same every time
    return events
```

### Layer 2: Agent Plans (LLM Orchestration)
**Multi-step workflows that combine LLM + deterministic code**

- Step 1: Parse natural language → structured params (LLM)
- Step 2: Call deterministic workflow (Pure code)
- Step 3: Rank/filter results (LLM)
- Step 4: Format response (LLM)
- Lives in: `workflows/agent_plans/`

**Example:**
```json
{
  "name": "personalized_event_finder",
  "steps": [
    {
      "id": 1,
      "type": "llm",
      "action": "parse_preferences",
      "prompt": "Extract location, keywords, size preference from: {user_input}",
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
      "action": "rank_events",
      "prompt": "Rank these events by user preferences: {params}",
      "input_from": ["raw_events", "params"],
      "output_var": "ranked_events"
    }
  ]
}
```

### Layer 3: Router
**Natural language interface**

- Analyzes user intent
- Routes to appropriate agent plan
- Guides users through creation
- Acts as conversational mentor

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd agent-school

# Install dependencies (requires UV)
uv sync

# Copy environment template
cp .env.example .env

# Add your API keys to .env
# Need at least one: OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### Usage for Non-Technical Users

Just run and chat!

```bash
# Interactive mode
python main.py

# Or single query
python main.py chat "find tech events in San Francisco"
```

**Example conversation:**
```
You: Find hip-hop parties in SF within 5 miles

Agent School: I don't have a workflow for Luma yet. Let's create one!
What platform: lu.ma
What should it do: Extract events from Luma
Create this workflow? Yes

[Generates deterministic workflow...]

Success! Workflow created: lu_ma_scraper
Method: browser (Luma API is paid)

You: Now find hip-hop parties in SF

Agent School: [Executes workflow and returns results]
Here are 3 hip-hop parties in SF:
1. Bass Night at The Midway - Jan 15
2. Hip-Hop Open Mic at Cafe Du Nord - Jan 17
...
```

### Usage for Technical Users

```bash
# Create deterministic workflow
python main.py create-workflow lu.ma --desc "Scrape Luma events"

# Create agent plan
python main.py create-plan personalized_finder \
  --uses luma_scraper \
  --goal "Find personalized events"

# List all workflows
python main.py list-all

# View system info
python main.py info
```

## Project Structure

```
agent-school/
├── agent_school/
│   ├── core/                           # Three-layer system
│   │   ├── deterministic_generator.py  # Layer 1: Pure code generator
│   │   ├── agent_planner.py            # Layer 2: LLM orchestration
│   │   ├── router.py                   # Layer 3: Intent routing
│   │   ├── executor.py                 # Executes agent plans
│   │   ├── registry.py                 # Workflow discovery
│   │   └── validator.py                # Schema validation
│   │
│   ├── utils/                          # Utilities
│   │   ├── logger.py
│   │   ├── code_validator.py
│   │   └── browser_helper.py
│   │
│   ├── config.py                       # Configuration
│   └── optimized_prompts.py            # LLM prompts
│
├── workflows/                          # Generated workflows
│   ├── deterministic/                  # Layer 1 storage
│   │   └── luma_scraper/
│   │       ├── workflow.py
│   │       ├── input_schema.json
│   │       ├── output_schema.json
│   │       └── metadata.json
│   │
│   ├── agent_plans/                    # Layer 2 storage
│   │   └── personalized_event_finder/
│   │       ├── plan.json
│   │       └── metadata.json
│   │
│   └── registry.json                   # Auto-generated index
│
├── main.py                             # Natural language CLI
├── tests/                              # Test suite
└── README.md
```

## How It Works

### Creating a Deterministic Workflow

```python
from agent_school.core import DeterministicGenerator

generator = DeterministicGenerator()

# LLM analyzes lu.ma and decides: API or browser?
# Result: API is paid → Use Playwright

workflow = generator.generate_workflow(
    name="luma_scraper",
    description="Extract events from Luma",
    target_platform="lu.ma"
)

# Generated code is pure Python, no LLM calls
# Saved to: workflows/deterministic/luma_scraper/workflow.py
```

### Creating an Agent Plan

```python
from agent_school.core import AgentPlanner

planner = AgentPlanner()

plan = planner.create_plan(
    name="personalized_event_finder",
    description="Find events matching user preferences",
    goal="Return top 5 personalized events",
    uses_workflows=["luma_scraper"],
    example_inputs=[
        "Find small startup events in SF",
        "Hip-hop parties within 10 miles"
    ]
)

# Plan defines LLM + deterministic workflow orchestration
# Saved to: workflows/agent_plans/personalized_event_finder/plan.json
```

### Executing with Router

```python
from agent_school.core import Router

router = Router()

# Natural language input
response = router.route("Find tech events in SF within 5 miles")

# Router:
# 1. Detects intent: find_events
# 2. Finds matching plan: personalized_event_finder
# 3. Executes plan with Executor
# 4. Returns formatted results
```

## Configuration

All settings in `.env`:

```bash
# Required: At least one LLM API key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
LUMA_API_KEY=...              # If you have paid Luma API access

# Settings
DEFAULT_LLM_PROVIDER=anthropic
CACHE_DIR=./workflows
DEBUG=false
```

## Testing

```bash
# Run fast tests (no LLM calls)
uv run pytest tests/

# Run all tests including LLM generation tests
uv run pytest tests/ --run-llm-tests

# Run specific test file
uv run pytest tests/test_config.py -v
```

**Test Results:**
- 21/22 tests passing
- Config validation: PASS
- Registry operations: PASS
- Code validation: PASS
- Schema extraction: PASS

## Use Cases

### 1. Event Discovery
"Find YC co-founder events in SF this weekend"

### 2. Data Aggregation
"Get tech news from multiple sources and summarize"

### 3. Market Research
"Track competitor pricing on 5 e-commerce sites"

### 4. Social Monitoring
"Find tweets about our product from last week"

## Key Features

### For Non-Technical Users
- Natural language interface
- Conversational mentor guides you
- No commands to memorize
- Just describe what you want

### For Developers
- Three-layer clean architecture
- Schema validation at every boundary
- Comprehensive test coverage
- Technical CLI commands available
- MCP integration ready

### For LLM Agents
- Dynamic tool generation
- MCP-compatible schemas
- Deterministic execution
- Composable workflows

## Security

Generated code goes through validation:
- Syntax checking
- Security analysis (no eval/exec without warning)
- Import whitelist
- Safe execution environment

For production:
- Run workflows in containers
- Implement rate limiting
- Add authentication
- Monitor executions

## Advanced Usage

### Creating Custom Prompts

Modify `agent_school/optimized_prompts.py` to customize how code is generated:

```python
WORKFLOW_SYSTEM_PROMPT = """
Your custom instructions here...
"""
```

### Adding New Platforms

```python
# The system learns new platforms automatically
generator = DeterministicGenerator()
generator.generate_workflow(
    name="custom_scraper",
    target_platform="example.com",  # Any platform!
    description="Extract data from example.com"
)
# LLM figures out how to scrape it
```

### Composing Agent Plans

```python
planner = AgentPlanner()

# Multi-source aggregation
plan = planner.create_plan(
    name="multi_source_events",
    uses_workflows=["luma_scraper", "eventbrite_scraper", "meetup_scraper"],
    goal="Aggregate events from multiple sources"
)
```

## Future Enhancements

- [ ] Dynamic data source discovery (LLM searches for APIs/platforms)
- [ ] Visual workflow editor
- [ ] Hosted MCP server
- [ ] Workflow marketplace
- [ ] A/B testing for generated code
- [ ] Performance monitoring
- [ ] Cost tracking per workflow
- [ ] Multi-modal support (images, videos)

## Examples

See `examples/` directory for:
- Creating Luma event scraper
- Building personalized event finder
- Multi-source data aggregation
- Custom platform integration

## CLI Commands

### Natural Language (Recommended)
```bash
python main.py                       # Interactive chat mode
python main.py chat "your query"     # Single query
```

### Technical Commands
```bash
python main.py create-workflow <platform>
python main.py create-plan <name> --uses <workflows>
python main.py list-all
python main.py info
```

## Documentation for Developers

### Creating Deterministic Workflows

Workflows must:
1. Accept structured input (typed parameters)
2. Return structured output (list of dicts)
3. Have no LLM calls inside
4. Be deterministic (same input → same output)
5. Include error handling

### Creating Agent Plans

Plans must:
1. Define clear steps (llm or deterministic)
2. Specify data flow (input_from, output_var)
3. Reference existing workflows
4. Include example inputs

### Schema Format

Input schemas:
```json
{
  "location": "str",
  "radius": "int",
  "keywords": "list",
  "optional_param": "str"
}
```

Output schemas:
```json
{
  "type": "list",
  "items": {
    "title": "str",
    "date": "str",
    "location": "str",
    "url": "str"
  }
}
```

## Contributing

Contributions welcome! Areas to improve:

1. Add more platform integrations
2. Improve code generation quality
3. Add workflow templates
4. Enhance natural language understanding
5. Build web interface

## License

[Add license]

## Contact

[Add contact]

---

**Built with:** Anthropic Claude, OpenAI GPT, LangChain, LangGraph, Playwright, UV

**Agent School v0.1.0** - Empowering LLMs to create their own tools
