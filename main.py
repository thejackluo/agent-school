"""
Agent School - Natural Language CLI

A conversational interface that guides users through workflow creation and execution.
No technical knowledge required - just tell it what you want in plain English!

Usage:
    python main.py                          # Interactive mode
    python main.py "find events in SF"      # Single query
    python main.py --help                   # Show help
"""

import sys
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import print as rprint
import json

from agent_school.core.router import Router
from agent_school.core.deterministic_generator import DeterministicGenerator
from agent_school.core.agent_planner import AgentPlanner
from agent_school.core.registry import Registry
from agent_school.config import Config

app = typer.Typer(add_completion=False)
console = Console()


def print_welcome():
    """Print welcome message."""
    welcome = """
# Welcome to Agent School!

I'm your AI mentor for creating automated workflows.

Just tell me what you want to do in plain English:
- "Find hip-hop parties in SF"
- "Create a workflow to scrape Luma events"
- "Show me what workflows I have"

I'll guide you through everything - no technical knowledge needed!
    """
    console.print(Panel(Markdown(welcome), border_style="cyan", title="Agent School"))


def print_response(response_data: Dict):
    """Pretty print router response."""
    response_text = response_data.get("response", "")

    if response_data.get("action") == "execute":
        # Show execution results
        console.print("\n[bold green]Results:[/bold green]")
        console.print(Panel(response_text, border_style="green"))

        # Show plan used
        if "plan_used" in response_data:
            console.print(f"\n[dim]Plan used: {response_data['plan_used']}[/dim]")

    elif response_data.get("needs_confirmation"):
        # Interactive guidance
        console.print(Panel(Markdown(response_text), border_style="yellow"))

    else:
        # General response
        console.print(Panel(Markdown(response_text), border_style="blue"))


def handle_create_workflow_interactive(router: Router, extracted_info: Dict):
    """Interactive workflow creation."""
    console.print("\n[bold cyan]Let's create a new workflow![/bold cyan]\n")

    # Ask for platform if not provided
    platform = extracted_info.get("platform")
    if not platform:
        platform = Prompt.ask("[bold]What platform/website?[/bold]", default="lu.ma")

    # Ask for description
    description = Prompt.ask("[bold]What should this workflow do?[/bold]",
                            default="Extract events from the platform")

    # Generate workflow name
    name = platform.replace(".", "_").replace("-", "_") + "_scraper"

    console.print(f"\n[dim]Creating workflow: {name}[/dim]")
    console.print(f"[dim]Platform: {platform}[/dim]")
    console.print(f"[dim]Description: {description}[/dim]\n")

    # Confirm
    if not typer.confirm("Create this workflow?"):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Create workflow
    try:
        generator = DeterministicGenerator(registry=router.registry)
        result = generator.generate_workflow(
            name=name,
            description=description,
            target_platform=platform,
            constraints={}
        )

        console.print(f"\n[bold green]Success![/bold green] Workflow created: {name}")
        console.print(f"Method: {result['method']}")
        console.print(f"Saved to: {result['path']}")

        # Show code preview
        code_preview = "\n".join(result['code'].splitlines()[:20])
        console.print("\n[bold]Generated Code (preview):[/bold]")
        console.print(Panel(
            Syntax(code_preview, "python", theme="monokai", line_numbers=True),
            border_style="blue"
        ))

        # Suggest next step
        console.print("\n[bold cyan]What's next?[/bold cyan]")
        console.print("Now you can:")
        console.print(f"1. Test it: python workflows/deterministic/{name}/workflow.py")
        console.print(f"2. Create an agent plan that uses this workflow")
        console.print(f"3. Or just ask me: 'Find events using {name}'")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@app.command()
def chat(query: Optional[str] = typer.Argument(None)):
    """
    Natural language interface - just tell me what you want!

    Examples:
        python main.py chat "find events in SF"
        python main.py chat  # Interactive mode
    """
    router = Router()

    if query:
        # Single query mode
        console.print(f"\n[bold]You:[/bold] {query}\n")
        response = router.route(query)
        print_response(response)

        # Handle follow-up if needed
        if response.get("needs_confirmation"):
            if response.get("action") == "create_workflow":
                handle_create_workflow_interactive(router, response.get("extracted_info", {}))

    else:
        # Interactive mode
        print_welcome()

        console.print("\n[dim]Type 'quit' or 'exit' to leave[/dim]\n")

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("\n[bold]Goodbye! Come back anytime![/bold]\n")
                    break

                if not user_input.strip():
                    continue

                # Route the input
                response = router.route(user_input)
                console.print()
                print_response(response)

                # Handle interactive flows
                if response.get("needs_confirmation"):
                    if response.get("action") == "create_workflow":
                        handle_create_workflow_interactive(router, response.get("extracted_info", {}))

            except KeyboardInterrupt:
                console.print("\n\n[bold]Goodbye![/bold]\n")
                break
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}\n")


@app.command()
def create_workflow(
    platform: str = typer.Argument(..., help="Platform to scrape (e.g., lu.ma)"),
    description: str = typer.Option("Extract data", "--desc", "-d", help="What this workflow does"),
):
    """
    [Technical] Create a deterministic workflow.

    For non-technical users: Use 'chat' command instead!
    """
    name = platform.replace(".", "_").replace("-", "_") + "_scraper"

    console.print(f"[INFO] Creating workflow: {name}")

    generator = DeterministicGenerator()
    result = generator.generate_workflow(
        name=name,
        description=description,
        target_platform=platform
    )

    console.print(f"[OK] Created: {name}")
    console.print(f"Method: {result['method']}")
    console.print(f"Path: {result['path']}")


@app.command()
def create_plan(
    name: str = typer.Argument(..., help="Plan name"),
    workflows: str = typer.Option(..., "--uses", help="Comma-separated workflow names"),
    goal: str = typer.Option(..., "--goal", help="What this plan achieves"),
):
    """
    [Technical] Create an agent plan.

    For non-technical users: Use 'chat' command instead!
    """
    workflows_list = [w.strip() for w in workflows.split(",")]

    console.print(f"[INFO] Creating plan: {name}")

    planner = AgentPlanner()
    plan = planner.create_plan(
        name=name,
        description=goal,
        goal=goal,
        uses_workflows=workflows_list
    )

    console.print(f"[OK] Created: {name}")
    console.print(f"Steps: {len(plan['steps'])}")


@app.command()
def list_all():
    """List all workflows and plans."""
    registry = Registry()

    workflows = registry.list_workflows()
    plans = registry.list_plans()

    console.print("\n[bold cyan]Workflows:[/bold cyan]")
    if workflows:
        for w in workflows:
            console.print(f"  - {w['name']} ({w['method']}): {w['description']}")
    else:
        console.print("  [dim]No workflows yet[/dim]")

    console.print("\n[bold cyan]Agent Plans:[/bold cyan]")
    if plans:
        for p in plans:
            console.print(f"  - {p['name']}: {p['description']}")
    else:
        console.print("  [dim]No plans yet[/dim]")

    console.print()


@app.command()
def info():
    """Show system information."""
    console.print("\n[bold cyan]Agent School v0.1.0[/bold cyan]\n")
    console.print(f"LLM Provider: {Config.DEFAULT_LLM_PROVIDER}")
    console.print(f"Workflows Dir: {Config.CACHE_DIR}")
    console.print(f"OpenAI Key: {'Set' if Config.OPENAI_API_KEY else 'Not set'}")
    console.print(f"Anthropic Key: {'Set' if Config.ANTHROPIC_API_KEY else 'Not set'}")

    registry = Registry()
    stats = registry.stats()
    console.print(f"\nWorkflows: {stats['deterministic_workflows']}")
    console.print(f"Plans: {stats['agent_plans']}")
    console.print()


if __name__ == "__main__":
    # Default to chat mode if no command specified
    if len(sys.argv) == 1:
        # No arguments, go to interactive chat
        chat()
    else:
        app()
