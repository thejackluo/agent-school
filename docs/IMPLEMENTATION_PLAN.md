# Agent School - Implementation Plan

## Project Goals

1. Create a system that dynamically generates executable functions for AI agents
2. Implement Luma event extraction as the primary case study
3. Build an extensible architecture that can support multiple data sources
4. Ensure deterministic, reliable workflow execution
5. Expose functions via MCP server for AI agent consumption

## Development Phases

### Phase 1: Foundation & Core Infrastructure (Week 1)

#### 1.1 Project Setup ✅
- [x] Initialize UV package manager
- [x] Install dependencies (OpenAI, Anthropic, LangChain, LangGraph, Playwright)
- [x] Set up environment variables (.env)
- [x] Create folder structure (backend/, docs/)
- [x] Write project documentation

#### 1.2 Basic LLM Integration
**Files**: `backend/core/llm_client.py`

**Tasks**:
- [ ] Create LLM client wrapper supporting both OpenAI and Anthropic
- [ ] Implement retry logic with exponential backoff
- [ ] Add response caching to reduce API costs
- [ ] Create prompt templates for different tasks
- [ ] Add cost tracking and logging

**Implementation**:
```python
# backend/core/llm_client.py
from enum import Enum
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import os

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LLMClient:
    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI):
        self.provider = provider
        if provider == LLMProvider.OPENAI:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4-turbo-preview"
        else:
            self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-opus-20240229"

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        # Implementation here
        pass
```

**Acceptance Criteria**:
- Can successfully call both OpenAI and Anthropic APIs
- Implements retry logic (3 attempts with exponential backoff)
- Caches identical requests for 1 hour
- Logs token usage and costs

#### 1.3 Playwright Browser Setup
**Files**: `backend/core/browser_client.py`

**Tasks**:
- [ ] Create browser automation client using Playwright
- [ ] Implement stealth mode to avoid detection
- [ ] Add screenshot capture for debugging
- [ ] Create reusable navigation patterns
- [ ] Implement error recovery

**Acceptance Criteria**:
- Can launch browser in headless mode
- Can navigate to Luma and interact with elements
- Takes screenshots on errors for debugging
- Handles network timeouts gracefully

---

### Phase 2: Plan Generator (Week 1-2)

#### 2.1 Plan Generator Core
**Files**: `backend/generators/plan_generator.py`

**Tasks**:
- [ ] Create plan generator using LangChain
- [ ] Design prompt template for plan generation
- [ ] Implement query parsing and intent extraction
- [ ] Add structured output validation (Pydantic models)
- [ ] Create plan optimization logic

**Prompt Template**:
```
You are a workflow planning expert for web scraping and API integration.

Given a user query about finding events, create a detailed execution plan.

User Query: {query}

Available Data Sources:
- Luma API (requires API key, may be unavailable)
- Luma Website (always available, requires browser automation)

Your plan should include:
1. Intent Analysis: What is the user trying to find?
2. Parameter Extraction: Location, radius, category, date range, keywords
3. Data Source Selection: API or browser automation?
4. Step-by-Step Workflow: Numbered steps to accomplish the task
5. Fallback Strategy: What to do if primary method fails
6. Output Format: How to structure the results

Generate a JSON plan following this schema:
{plan_schema}
```

**Plan Schema**:
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PlanStep(BaseModel):
    step_number: int
    action: str = Field(..., description="Action to perform")
    method: str = Field(..., description="'api' or 'browser'")
    params: Dict[str, Any] = Field(default_factory=dict)
    fallback: Optional[str] = None

class ExecutionPlan(BaseModel):
    task_name: str
    intent: str
    extracted_params: Dict[str, Any]
    steps: List[PlanStep]
    estimated_duration_seconds: int
    fallback_strategy: str
```

**Acceptance Criteria**:
- Correctly parses 90%+ of test queries
- Generates valid JSON plans that match schema
- Extracts location, category, and other parameters accurately
- Chooses appropriate data source (API vs browser)
- Plans execute in reasonable time (<30 seconds)

#### 2.2 Plan Validation & Testing
**Files**: `backend/generators/tests/test_plan_generator.py`

**Tasks**:
- [ ] Create test suite with 20+ diverse queries
- [ ] Validate plan structure and completeness
- [ ] Test edge cases (ambiguous queries, missing params)
- [ ] Benchmark plan generation speed
- [ ] Test with both OpenAI and Anthropic

**Test Queries**:
```python
TEST_QUERIES = [
    "Find hip-hop parties in SF within 5 miles",
    "YC co-founder events",
    "Private networking events for founders",
    "Tech conferences in SF next month under $100",
    "Family-friendly events in Berkeley this weekend",
    "Find all events by organization X",
    # ... more test cases
]
```

---

### Phase 3: Workflow Generator (Week 2-3)

#### 3.1 API-Based Workflow Generator
**Files**: `backend/workflows/api_workflow_generator.py`

**Tasks**:
- [ ] Create workflow generator for API-based execution
- [ ] Design prompt for generating Python code
- [ ] Implement code validation and syntax checking
- [ ] Add import management (ensure required libraries are used)
- [ ] Create workflow execution sandbox

**Prompt Template**:
```
You are an expert Python developer specializing in API integration.

Generate production-ready Python code to execute this plan:
{plan_json}

Requirements:
1. Use async/await with httpx for API calls
2. Add comprehensive error handling
3. Include type hints and docstrings
4. Make code deterministic and idempotent
5. Use environment variables for API keys
6. Return structured data matching this schema: {output_schema}

Available libraries: httpx, json, datetime, typing, pydantic

Generate a complete async function.
```

**Acceptance Criteria**:
- Generates syntactically valid Python code
- Code passes pylint/mypy checks
- Includes proper error handling
- Returns data in specified format
- Can be executed in sandbox environment

#### 3.2 Browser-Based Workflow Generator
**Files**: `backend/workflows/browser_workflow_generator.py`

**Tasks**:
- [ ] Create workflow generator for Playwright automation
- [ ] Design prompts for generating browser interaction code
- [ ] Implement selector generation and validation
- [ ] Add screenshot capture for debugging
- [ ] Create retry logic for flaky selectors

**Prompt Template**:
```
You are an expert in web scraping using Playwright.

Generate Python code using Playwright to execute this plan:
{plan_json}

Target Website: https://lu.ma/explore

Requirements:
1. Use async Playwright API
2. Launch browser in headless mode
3. Navigate to Luma and search for events
4. Extract event data from the page
5. Handle pagination if needed
6. Return structured data matching: {output_schema}

Available tools: playwright.async_api

Generate complete async function with proper error handling.
```

**Acceptance Criteria**:
- Generated code successfully navigates Luma
- Extracts event data accurately (>95% accuracy)
- Handles pagination correctly
- Recovers from network errors
- Takes debug screenshots on failure

#### 3.3 Workflow Execution Engine
**Files**: `backend/workflows/workflow_executor.py`

**Tasks**:
- [ ] Create execution engine that runs generated workflows
- [ ] Implement sandboxing for security
- [ ] Add timeout management
- [ ] Create execution result validation
- [ ] Implement workflow caching

**Acceptance Criteria**:
- Can execute both API and browser workflows
- Enforces execution timeouts (max 60 seconds)
- Validates outputs match expected schema
- Caches successful workflows for reuse
- Logs execution metrics (time, memory, success rate)

---

### Phase 4: Function Creator & MCP Server (Week 3-4)

#### 4.1 Function Creator
**Files**: `backend/functions/function_creator.py`

**Tasks**:
- [ ] Create function creator that wraps workflows in MCP functions
- [ ] Generate Pydantic models for input/output schemas
- [ ] Add function metadata (name, description, examples)
- [ ] Implement input validation and sanitization
- [ ] Create function registry for persistence

**Function Wrapper Template**:
```python
async def {function_name}(request: {RequestModel}) -> {ResponseModel}:
    """
    {description}

    Generated by Agent School on {timestamp}.

    Args:
        request: {request_description}

    Returns:
        {ResponseModel} with results

    Raises:
        ValueError: If inputs are invalid
        HTTPException: If execution fails

    Examples:
        {examples}
    """
    # Validation
    # Plan generation
    # Workflow execution
    # Result formatting
    pass
```

**Acceptance Criteria**:
- Generated functions have proper type hints
- Pydantic models validate all inputs/outputs
- Functions include comprehensive docstrings
- Can be registered in function registry
- Metadata is MCP-compatible

#### 4.2 MCP Server Implementation
**Files**: `backend/server/mcp_server.py`

**Tasks**:
- [ ] Create MCP server to expose generated functions
- [ ] Implement function registration and discovery
- [ ] Add authentication and authorization
- [ ] Create API documentation endpoint
- [ ] Implement rate limiting

**Acceptance Criteria**:
- MCP server runs and accepts connections
- AI agents can discover available functions
- Functions can be called via MCP protocol
- Requests are authenticated and rate-limited
- Server handles concurrent requests

#### 4.3 Function Registry & Persistence
**Files**: `backend/functions/function_registry.py`

**Tasks**:
- [ ] Create registry to store generated functions
- [ ] Implement versioning for functions
- [ ] Add function search and filtering
- [ ] Create function update/deletion logic
- [ ] Store function metadata in SQLite database

---

### Phase 5: Testing & Validation (Week 4)

#### 5.1 Test Runner Implementation
**Files**: `backend/tests/test_runner.py`

**Tasks**:
- [ ] Create comprehensive test runner
- [ ] Implement unit tests for each component
- [ ] Add integration tests for full workflow
- [ ] Create validation tests for output schemas
- [ ] Implement performance benchmarks

**Test Categories**:
```python
# Unit Tests
test_llm_client()
test_plan_generator()
test_workflow_generator()
test_function_creator()

# Integration Tests
test_end_to_end_api_workflow()
test_end_to_end_browser_workflow()
test_mcp_server_integration()

# Validation Tests
test_output_schema_compliance()
test_error_handling()
test_edge_cases()

# Performance Tests
test_execution_speed()
test_concurrent_requests()
test_memory_usage()
```

**Acceptance Criteria**:
- All unit tests pass (100% coverage for core logic)
- Integration tests pass with real Luma data
- Performance tests show <5s average execution time
- Can run test suite with single command

#### 5.2 Main Test File
**Files**: `main.py`

**Tasks**:
- [ ] Create main entry point for running test workflows
- [ ] Implement CLI interface for testing queries
- [ ] Add example queries with expected outputs
- [ ] Create demo mode for showcasing system

**CLI Interface**:
```bash
# Test specific query
python main.py --query "Find hip-hop parties in SF within 5 miles"

# Run all examples
python main.py --run-examples

# Start MCP server
python main.py --start-server

# Run test suite
python main.py --test
```

---

### Phase 6: Documentation & Polish (Week 4-5)

#### 6.1 API Documentation
**Files**: `docs/API_REFERENCE.md`

**Tasks**:
- [ ] Document all public APIs and functions
- [ ] Create usage examples for each function
- [ ] Add code snippets and sample outputs
- [ ] Document error codes and handling
- [ ] Create OpenAPI/Swagger spec

#### 6.2 Usage Guide
**Files**: `docs/USAGE_GUIDE.md`

**Tasks**:
- [ ] Write step-by-step usage guide
- [ ] Create tutorials for common scenarios
- [ ] Add troubleshooting section
- [ ] Document best practices
- [ ] Include performance optimization tips

#### 6.3 Deployment Guide
**Files**: `docs/DEPLOYMENT.md`

**Tasks**:
- [ ] Document deployment process
- [ ] Create Docker container configuration
- [ ] Write production environment setup guide
- [ ] Add monitoring and logging setup
- [ ] Document scaling strategies

---

## Implementation Priorities

### P0 (Must Have for MVP)
1. Basic LLM integration with OpenAI
2. Plan generator with query parsing
3. Browser-based workflow generator (since Luma API may be unavailable)
4. Function creator with basic validation
5. Main test file with example queries

### P1 (Should Have)
1. API-based workflow generator (if Luma API becomes available)
2. MCP server implementation
3. Function registry and persistence
4. Comprehensive test suite
5. Complete documentation

### P2 (Nice to Have)
1. Anthropic Claude integration as fallback
2. Advanced caching and optimization
3. Web UI for testing workflows
4. Analytics and monitoring dashboard
5. Multi-platform support (beyond Luma)

---

## Development Milestones

### Milestone 1: Foundation Complete
- LLM client working
- Browser client working
- Basic plan generator creates valid plans

### Milestone 2: Workflow Generation Working
- Can generate API workflows
- Can generate browser workflows
- Workflows execute successfully against Luma

### Milestone 3: Function Creation Complete
- Functions are properly wrapped
- MCP server exposes functions
- AI agents can call functions

### Milestone 4: Production Ready
- All tests passing
- Documentation complete
- Performance meets requirements
- Ready for internal deployment

---

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock external dependencies (LLMs, APIs, browser)
- Aim for >80% code coverage

### Integration Tests
- Test full workflow end-to-end
- Use real Luma website (in test environment)
- Validate output accuracy

### Performance Tests
- Benchmark execution time for each component
- Test concurrent request handling
- Monitor memory usage and resource consumption

### User Acceptance Tests
- Test with real user queries
- Validate results meet user expectations
- Gather feedback for improvements

---

## Risk Mitigation

### Risk 1: Luma Website Changes
**Impact**: Browser workflows break if Luma updates their HTML/CSS
**Mitigation**:
- Create flexible selectors that adapt to minor changes
- Implement visual regression testing
- Add fallback selectors
- Monitor Luma for changes and update workflows quickly

### Risk 2: LLM Rate Limits or Costs
**Impact**: High API costs or rate limit errors
**Mitigation**:
- Implement aggressive caching
- Use cheaper models (GPT-3.5) for simple tasks
- Add request throttling
- Monitor and alert on unusual usage

### Risk 3: Generated Code Security
**Impact**: Generated code could be malicious or buggy
**Mitigation**:
- Execute in sandboxed environment
- Validate generated code with AST parsing
- Whitelist allowed imports and functions
- Run static analysis (pylint, mypy) on generated code

### Risk 4: Workflow Accuracy
**Impact**: Generated workflows don't extract data correctly
**Mitigation**:
- Implement comprehensive validation tests
- Add manual review step for new workflows
- Track success/failure rates
- Continuously improve prompts based on failures

---

## Success Metrics

### Accuracy
- Plan generation: >90% of queries produce valid plans
- Workflow execution: >95% success rate on Luma data extraction
- Output validation: 100% of outputs match expected schema

### Performance
- Plan generation: <2 seconds
- Workflow execution: <5 seconds (API), <15 seconds (browser)
- End-to-end: <20 seconds from query to result

### Reliability
- System uptime: >99%
- Error recovery: 100% of errors logged and handled gracefully
- MCP server: Can handle 100+ concurrent requests

### Developer Experience
- Documentation: Complete and clear
- Setup time: <10 minutes to get running
- Test coverage: >80% code coverage

---

## Next Steps

1. **Immediate**: Complete Phase 1 (Foundation & Core Infrastructure)
2. **This Week**: Implement Plan Generator (Phase 2)
3. **Next Week**: Build Workflow Generator (Phase 3)
4. **Following Week**: Create Function Creator & MCP Server (Phase 4)
5. **Final Week**: Testing, documentation, and polish (Phases 5-6)

---

## Questions & Decisions Needed

1. **Luma API Access**: Do we have access to Luma API? If not, focus on browser automation.
2. **LLM Provider**: Primary = OpenAI, Fallback = Anthropic?
3. **Deployment Environment**: Where will this run? (Local, Cloud, Docker)
4. **Function Persistence**: SQLite or more robust database?
5. **MCP Server**: Run as separate service or embedded?

---

## Appendix: File Structure

```
agent-school/
├── backend/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── llm_client.py           # LLM wrapper
│   │   └── browser_client.py       # Playwright wrapper
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── plan_generator.py       # File 2: Plan generation
│   │   └── tests/
│   │       └── test_plan_generator.py
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── workflow_generator.py   # File 1: Workflow generation
│   │   ├── api_workflow_generator.py
│   │   ├── browser_workflow_generator.py
│   │   ├── workflow_executor.py
│   │   └── tests/
│   │       └── test_workflow_generator.py
│   ├── functions/
│   │   ├── __init__.py
│   │   ├── function_creator.py     # File 3: Function creation
│   │   ├── function_registry.py
│   │   └── tests/
│   │       └── test_function_creator.py
│   ├── server/
│   │   ├── __init__.py
│   │   └── mcp_server.py          # MCP server
│   └── tests/
│       ├── __init__.py
│       └── test_runner.py         # File 4: Main test runner
├── docs/
│   ├── README.md                  # Overview
│   ├── ARCHITECTURE.md            # System design
│   ├── IMPLEMENTATION_PLAN.md     # This file
│   ├── API_REFERENCE.md           # API docs
│   ├── USAGE_GUIDE.md            # How to use
│   └── DEPLOYMENT.md             # Deployment guide
├── main.py                        # Entry point
├── pyproject.toml                # Dependencies
├── .env                          # API keys
└── README.md                     # Project overview
```
