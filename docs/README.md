# Agent School - Documentation

## Overview

Agent School is an AI-powered system that dynamically generates executable functions for AI agents to call. The system uses large language models (LLMs) to create deterministic workflows that can interact with external APIs or browser automation tools to accomplish complex tasks.

## Project Purpose

This project solves the challenge of creating reliable, repeatable workflows for AI agents to extract and process data from various sources. Instead of manually coding specific integrations, Agent School uses LLMs to generate the workflows themselves, making it adaptable to new data sources and requirements.

## Case Study: Luma Event Extraction

The initial implementation focuses on extracting events from Luma.co, a platform for discovering events. The system can:

- Generate workflows that call the Luma API (when available)
- Create browser automation scripts when API access is limited
- Parse natural language queries (e.g., "Find hip-hop parties with hot girls in SF within 5 miles")
- Generate deterministic, reusable functions that can be called by AI agents

## Architecture Overview

### Core Components

```
agent-school/
├── backend/
│   ├── workflows/          # Workflow generation (File 1)
│   │   └── workflow_generator.py
│   ├── generators/         # Plan generation (File 2)
│   │   └── plan_generator.py
│   ├── functions/          # Function creation (File 3)
│   │   └── function_creator.py
│   └── tests/             # Testing infrastructure (File 4)
│       └── test_runner.py
├── docs/                   # Documentation
└── main.py                # Entry point
```

### System Flow

1. **User Input** → Natural language query (e.g., "Find YC co-founder events")
2. **Plan Generator** → LLM creates step-by-step plan for the task
3. **Workflow Generator** → LLM generates executable workflow (API calls or browser automation)
4. **Function Creator** → Combines plan + workflow into callable function
5. **MCP Server** → Exposes function for AI agents to call
6. **Test Runner** → Validates function works correctly

## Key Features

- **Dynamic Workflow Generation**: Uses LLMs to create workflows instead of hardcoding them
- **Multi-Modal Execution**: Supports both API calls and browser automation (via Playwright)
- **Natural Language Interface**: Accepts human-readable queries and converts to executable code
- **Deterministic Output**: Generates reliable, repeatable functions
- **MCP Server Integration**: Functions can be called by AI agents through Model Context Protocol
- **Extensible Architecture**: Easy to add new data sources beyond Luma

## Technology Stack

- **Python 3.13+**: Core language
- **UV Package Manager**: Dependency management
- **OpenAI API**: Primary LLM for generation
- **Anthropic Claude**: Alternative/fallback LLM
- **LangChain**: LLM orchestration framework
- **LangGraph**: State machine for agentic workflows
- **Playwright**: Browser automation
- **Model Context Protocol (MCP)**: AI agent function calling

## Environment Variables

Required environment variables in `.env`:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LUMA_API_KEY=your_luma_key  # If available
```

## Use Cases

### Example 1: Event Search by Vibes
```
Query: "Find hip-hop parties with hot girls in SF within 5 miles"
→ System generates workflow to search Luma with location + category filters
→ Returns list of matching events with details
```

### Example 2: Professional Networking
```
Query: "Find YC co-founder events or private events"
→ System generates workflow to filter by professional categories
→ Handles authentication if required for private events
→ Returns curated event list
```

### Example 3: Custom Filters
```
Query: "Find tech conferences in SF next month under $100"
→ System generates workflow with date, location, category, and price filters
→ Returns filtered results
```

## Documentation Structure

- **README.md** (this file): Project overview and getting started
- **ARCHITECTURE.md**: Detailed system design and component interactions
- **IMPLEMENTATION_PLAN.md**: Step-by-step development roadmap
- **API_REFERENCE.md**: Function signatures and usage examples
- **USAGE_GUIDE.md**: How to use the system and create new workflows

## Getting Started

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Set Up Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run Basic Example**:
   ```bash
   python main.py
   ```

4. **Run Tests**:
   ```bash
   python backend/tests/test_runner.py
   ```

## Next Steps

- Read the [Architecture Document](./ARCHITECTURE.md) to understand system design
- Review the [Implementation Plan](./IMPLEMENTATION_PLAN.md) for development roadmap
- Check the [Usage Guide](./USAGE_GUIDE.md) for practical examples
- Explore the [API Reference](./API_REFERENCE.md) for function documentation

## Contributing

This is an internal project. For questions or issues, contact the development team.

## License

Proprietary - Internal use only
