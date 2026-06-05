#!/usr/bin/env python3
"""CLI interface for the liver disease AI agent."""

from __future__ import annotations
import argparse
import json
import sys
import textwrap
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich import box
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent import LiverAgent

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🔬  HepatoAI — Liver Disease Clinical AI           ║
║      Powered by Claude claude-sonnet-4-6 + AASLD/EASL Guidelines         ║
╚══════════════════════════════════════════════════════════════╝
"""

TOOL_ICONS = {
    "parse_lab_values": "🧪",
    "calculate_severity_scores": "📊",
    "differential_diagnosis": "🔍",
    "assess_fibrosis_stage": "📈",
    "get_treatment_guidelines": "💊",
    "generate_clinical_summary": "📋",
    "flag_urgent_findings": "🚨",
}

TOOL_LABELS = {
    "parse_lab_values": "Parsing Laboratory Values",
    "calculate_severity_scores": "Calculating Severity Scores (Child-Pugh/MELD/ALBI)",
    "differential_diagnosis": "Generating Differential Diagnosis",
    "assess_fibrosis_stage": "Assessing Fibrosis Stage",
    "get_treatment_guidelines": "Retrieving Treatment Guidelines",
    "generate_clinical_summary": "Generating Clinical Summary",
    "flag_urgent_findings": "Screening for Urgent Findings",
}


def display_banner(mode: str) -> None:
    mode_label = "👨‍⚕️ Physician Mode" if mode == "physician" else "🏥 Patient Mode"
    console.print(BANNER, style="bold cyan")
    console.print(f"  Mode: [bold green]{mode_label}[/bold green]", justify="center")
    console.print(
        "  Commands: [dim]'quit'[/dim] to exit | [dim]'reset'[/dim] to clear history | "
        "[dim]'usage'[/dim] to show token usage | [dim]'mode'[/dim] to switch mode\n"
    )


def display_tool_start(tool_name: str, tool_input: dict) -> None:
    icon = TOOL_ICONS.get(tool_name, "⚙️")
    label = TOOL_LABELS.get(tool_name, tool_name.replace("_", " ").title())
    console.print(f"\n  {icon} [italic dim]{label}...[/italic dim]")


def display_tool_result(tool_name: str, result: dict) -> None:
    if "error" in result:
        console.print(Panel(
            f"[red]Error: {result['error']}[/red]",
            title=f"⚠️ Tool Error: {tool_name}",
            border_style="red",
        ))
        return

    icon = TOOL_ICONS.get(tool_name, "⚙️")

    if tool_name == "parse_lab_values":
        _display_lab_results(result, icon)
    elif tool_name == "calculate_severity_scores":
        _display_severity_scores(result, icon)
    elif tool_name == "flag_urgent_findings":
        _display_urgent_findings(result, icon)
    elif tool_name == "differential_diagnosis":
        _display_differential(result, icon)
    elif tool_name == "assess_fibrosis_stage":
        _display_fibrosis(result, icon)
    elif tool_name == "get_treatment_guidelines":
        _display_treatment(result, icon)
    else:
        # Generic display
        console.print(Panel(
            json.dumps(result, indent=2)[:1000],
            title=f"{icon} {TOOL_LABELS.get(tool_name, tool_name)}",
            border_style="blue",
        ))


def _display_lab_results(result: dict, icon: str) -> None:
    if not result.get("values"):
        return

    table = Table(
        title=f"{icon} Laboratory Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("Test", style="white")
    table.add_column("Value", justify="right")
    table.add_column("Reference Range", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Interpretation", style="italic")

    status_styles = {
        "normal": "green",
        "high": "yellow",
        "low": "yellow",
        "critical_high": "bold red",
        "critical_low": "bold red",
        "abnormal": "red",
    }

    for lab in result["values"]:
        status = lab.get("status", "normal")
        style = status_styles.get(status, "white")
        status_display = {
            "normal": "✓ Normal",
            "high": "↑ High",
            "low": "↓ Low",
            "critical_high": "⚠ CRIT HIGH",
            "critical_low": "⚠ CRIT LOW",
            "abnormal": "⚠ Abnormal",
        }.get(status, status)

        table.add_row(
            lab.get("name", ""),
            f"{lab.get('value', '')} {lab.get('unit', '')}",
            lab.get("reference_range", ""),
            f"[{style}]{status_display}[/{style}]",
            lab.get("interpretation", ""),
        )

    console.print(table)
    summary = result.get("summary", "")
    if summary:
        console.print(f"  [dim]{summary}[/dim]\n")


def _display_severity_scores(result: dict, icon: str) -> None:
    panels = []

    cp = result.get("child_pugh")
    if cp:
        grade = cp.get("grade", "?")
        grade_color = {"A": "green", "B": "yellow", "C": "red"}.get(grade, "white")
        content = (
            f"[bold {grade_color}]Grade {grade}[/bold {grade_color}] "
            f"(Score: {cp.get('total_score', '?')}/15)\n\n"
            f"[dim]Bilirubin: {cp.get('bilirubin_points', '?')} pts | "
            f"Albumin: {cp.get('albumin_points', '?')} pts | "
            f"INR: {cp.get('inr_points', '?')} pts | "
            f"Ascites: {cp.get('ascites_points', '?')} pts | "
            f"Enceph: {cp.get('encephalopathy_points', '?')} pts[/dim]\n\n"
            f"1-yr survival: {cp.get('one_year_survival', '?')} | "
            f"2-yr survival: {cp.get('two_year_survival', '?')}"
        )
        panels.append(Panel(content, title="Child-Pugh Score", border_style=grade_color))

    meld = result.get("meld")
    if meld:
        score = meld.get("score", 0)
        color = "green" if score < 10 else "yellow" if score < 20 else "red"
        content = (
            f"[bold {color}]Score: {score}[/bold {color}]\n"
            f"Category: {meld.get('category', '?')}\n"
            f"3-month mortality: {meld.get('three_month_mortality', '?')}"
        )
        panels.append(Panel(content, title="MELD Score", border_style=color))

    albi = result.get("albi")
    if albi:
        grade = albi.get("grade", "?")
        grade_color = {1: "green", 2: "yellow", 3: "red"}.get(grade, "white")
        content = (
            f"[bold {grade_color}]Grade {grade}[/bold {grade_color}]\n"
            f"Score: {albi.get('score', '?')}"
        )
        panels.append(Panel(content, title="ALBI Score", border_style=grade_color))

    if panels:
        console.print(Columns(panels, equal=False, expand=False))

    missing = result.get("missing_values", [])
    if missing:
        console.print(f"  [dim]Missing: {', '.join(missing)}[/dim]")
    console.print()


def _display_urgent_findings(result: dict, icon: str) -> None:
    if not result.get("findings"):
        console.print(f"  [green]✓ No urgent or critical findings identified[/green]\n")
        return

    border_color = "red" if result.get("has_critical") else "yellow"
    title_text = (
        f"🚨 CRITICAL ALERT" if result.get("has_critical")
        else "⚠️  Urgent Findings"
    )

    findings_text = ""
    for f in result["findings"]:
        level = f.get("urgency_level", "URGENT")
        color = "bold red" if level == "CRITICAL" else "yellow"
        findings_text += (
            f"[{color}][{level}][/{color}] {f.get('finding', '')}\n"
            f"  → Action: {f.get('recommended_action', '')}\n"
            f"  → Timeframe: {f.get('timeframe', '')}\n"
            f"  → Rationale: [dim]{f.get('rationale', '')}[/dim]\n\n"
        )

    console.print(Panel(
        findings_text.strip(),
        title=title_text,
        border_style=border_color,
    ))

    if result.get("emergency_actions"):
        for action in result["emergency_actions"]:
            console.print(f"  [bold red]{action}[/bold red]")
    console.print()


def _display_differential(result: dict, icon: str) -> None:
    if not result.get("differentials"):
        return

    console.print(Rule(f"{icon} Differential Diagnosis", style="blue"))
    console.print(f"  Primary: [bold green]{result.get('primary_diagnosis', 'Unknown')}[/bold green]\n")

    for dx in result["differentials"]:
        prob = dx.get("probability", "low")
        prob_color = {"high": "red", "moderate": "yellow", "low": "dim"}.get(prob, "dim")
        prob_bar = {"high": "███", "moderate": "██░", "low": "█░░"}.get(prob, "░░░")

        console.print(f"  [{prob_color}]{dx.get('rank', '?')}. {dx.get('condition', '')}[/{prob_color}] "
                      f"[{prob_color}]{prob_bar} {prob.upper()}[/{prob_color}]")

        if dx.get("next_steps"):
            steps = " | ".join(dx["next_steps"][:3])
            console.print(f"     [dim]Next steps: {steps}[/dim]")

    if result.get("urgent_considerations"):
        console.print()
        for uc in result["urgent_considerations"]:
            console.print(f"  [bold red]{uc}[/bold red]")
    console.print()


def _display_fibrosis(result: dict, icon: str) -> None:
    stage = result.get("stage", "?")
    confidence = result.get("confidence", "low")

    stage_colors = {
        "F0": "green", "F1": "green", "F0-F1": "green",
        "F2": "yellow", "F1-F2": "yellow",
        "F3": "orange_red1", "F3-F4": "red", "F0-F2": "yellow",
        "F4": "red",
    }
    color = stage_colors.get(stage, "white")

    content = (
        f"[bold {color}]Stage: {stage}[/bold {color}]\n"
        f"{result.get('description', '')}\n\n"
        f"Confidence: {confidence.upper()}\n\n"
    )
    if result.get("supporting_evidence"):
        content += "Evidence:\n" + "\n".join(f"• {e}" for e in result["supporting_evidence"])

    console.print(Panel(content, title=f"{icon} Fibrosis Assessment", border_style=color))
    console.print(f"  [dim]Clinical significance: {result.get('clinical_significance', '')}[/dim]\n")


def _display_treatment(result: dict, icon: str) -> None:
    console.print(Rule(f"{icon} Treatment Guidelines: {result.get('diagnosis', '')}", style="green"))

    if result.get("first_line_treatment"):
        console.print("  [bold green]First-Line Treatment:[/bold green]")
        for item in result["first_line_treatment"]:
            console.print(f"  • {item}")

    if result.get("treatment_goals"):
        console.print("\n  [bold blue]Treatment Goals:[/bold blue]")
        for goal in result["treatment_goals"][:3]:
            console.print(f"  ✓ {goal}")

    if result.get("monitoring_parameters"):
        console.print("\n  [bold yellow]Monitoring:[/bold yellow]")
        for param in result["monitoring_parameters"][:3]:
            console.print(f"  → {param}")

    if result.get("guideline_source"):
        console.print(f"\n  [dim italic]Source: {result['guideline_source']}[/dim italic]")
    console.print()


def display_response(text: str) -> None:
    """Render the final assistant response as markdown."""
    if text.strip():
        console.print(Panel(
            Markdown(text),
            title="🤖 HepatoAI",
            border_style="bright_blue",
            padding=(1, 2),
        ))


def display_usage(agent: LiverAgent) -> None:
    usage = agent.usage_summary
    table = Table(title="Token Usage", box=box.SIMPLE, show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Tokens", justify="right", style="white")
    table.add_row("Input tokens", f"{usage['total_input_tokens']:,}")
    table.add_row("Output tokens", f"{usage['total_output_tokens']:,}")
    table.add_row("Cache reads", f"[green]{usage['cache_read_tokens']:,}[/green]")
    table.add_row("Cache writes", f"[dim]{usage['cache_creation_tokens']:,}[/dim]")

    # Estimate cost (claude-sonnet-4-6: $3/1M input, $15/1M output, $0.30/1M cache read, $3.75/1M cache write)
    cost = (
        usage['total_input_tokens'] * 3 / 1_000_000
        + usage['total_output_tokens'] * 15 / 1_000_000
        + usage['cache_read_tokens'] * 0.30 / 1_000_000
        + usage['cache_creation_tokens'] * 3.75 / 1_000_000
    )
    table.add_row("[bold]Estimated cost[/bold]", f"[bold]${cost:.4f}[/bold]")
    console.print(table)


def run_interactive(mode: str, verbose: bool) -> None:
    """Run the agent in interactive chat mode."""
    display_banner(mode)
    agent = LiverAgent(mode=mode, verbose=verbose)

    console.print("[dim]Ready for clinical queries. Type your question below.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit", "q"):
            display_usage(agent)
            console.print("[dim]Session ended.[/dim]")
            break
        if cmd == "reset":
            agent.reset()
            console.print("[green]✓ Conversation history cleared.[/green]\n")
            continue
        if cmd == "usage":
            display_usage(agent)
            continue
        if cmd == "mode":
            agent.mode = "patient" if agent.mode == "physician" else "physician"
            console.print(f"[green]✓ Switched to {'Patient' if agent.mode == 'patient' else 'Physician'} mode.[/green]\n")
            continue

        # Run the agent turn
        console.print()
        full_response = ""

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("HepatoAI is analyzing...", total=None)

            def on_tool_start(tool_name: str, tool_input: dict) -> None:
                progress.update(task, description=f"{TOOL_ICONS.get(tool_name, '⚙️')} {TOOL_LABELS.get(tool_name, tool_name)}...")

            def on_tool_end(tool_name: str, result: dict) -> None:
                progress.stop()
                display_tool_result(tool_name, result)
                progress.start()
                progress.update(task, description="HepatoAI is analyzing...")

            full_response = agent.run_turn(
                user_input,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            )

        if full_response.strip():
            display_response(full_response)
        console.print()


def run_single_query(query: str, mode: str, verbose: bool, output_json: bool) -> None:
    """Run a single query and output the result."""
    agent = LiverAgent(mode=mode, verbose=verbose)

    if not output_json:
        console.print(f"[dim]Processing: {query[:80]}{'...' if len(query) > 80 else ''}[/dim]\n")

    tool_results_collected = []

    def on_tool_end(tool_name: str, result: dict) -> None:
        tool_results_collected.append({"tool": tool_name, "result": result})
        if not output_json:
            display_tool_result(tool_name, result)

    response = agent.run_turn(
        query,
        on_tool_end=on_tool_end,
    )

    if output_json:
        output = {
            "query": query,
            "mode": mode,
            "tool_results": tool_results_collected,
            "response": response,
            "usage": agent.usage_summary,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        if response.strip():
            display_response(response)
        display_usage(agent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HepatoAI — Liver Disease Diagnosis & Treatment Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python cli.py                                          # Interactive mode (physician)
          python cli.py --mode patient                           # Interactive mode (patient)
          python cli.py -q "ALT 250, AST 180, HBsAg positive"   # Single query
          python cli.py -q "..." --json                          # JSON output
          python cli.py --verbose                                # Debug token usage
        """),
    )
    parser.add_argument(
        "--mode",
        choices=["physician", "patient"],
        default="physician",
        help="Interaction mode: physician (clinical) or patient (simplified)",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Single query mode — run this query and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON (for single query mode)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug info including token usage per call",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a demo case (45-year-old with HBV cirrhosis)",
    )

    args = parser.parse_args()

    if args.demo:
        demo_query = (
            "Patient: 45-year-old male. "
            "History: known HBV infection for 10 years, alcohol use (3-4 drinks/day). "
            "Symptoms: increasing abdominal distension, jaundice, confusion. "
            "Labs: ALT 85 U/L, AST 120 U/L, ALP 180 U/L, total bilirubin 4.2 mg/dL, "
            "albumin 2.8 g/dL, INR 1.9, creatinine 1.4 mg/dL, platelets 85 ×10³/μL, "
            "AFP 45 ng/mL, HBsAg positive, HBV DNA 50,000 IU/mL. "
            "Imaging: ultrasound shows nodular liver, splenomegaly, moderate ascites."
        )
        console.print("[bold cyan]🔬 Running Demo Case:[/bold cyan]")
        console.print(Panel(demo_query, title="Case Presentation", border_style="dim"))
        console.print()
        run_single_query(demo_query, args.mode, args.verbose, args.json)
    elif args.query:
        run_single_query(args.query, args.mode, args.verbose, args.json)
    else:
        run_interactive(args.mode, args.verbose)


if __name__ == "__main__":
    main()
