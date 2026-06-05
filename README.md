# HepatoAI — Liver Disease Diagnosis & Treatment Agent

A clinical decision support AI agent for hepatology, powered by Claude claude-sonnet-4-6 with evidence-based tools grounded in AASLD and EASL guidelines.

## Features

- **7 specialized tools** for hepatology decision support
- **Prompt caching** for efficient API usage
- **Physician & Patient modes** for different audiences  
- **Rich CLI** with color-coded lab results, severity scores, and clinical panels
- **Agentic loop** — Claude autonomously decides which tools to use

## Quick Start

```bash
cd "liver agent"
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here

# Interactive mode (physician)
python cli.py

# Interactive mode (patient-friendly)
python cli.py --mode patient

# Single query
python cli.py -q "ALT 250, AST 180, HBsAg positive, cirrhosis on ultrasound"

# Run a built-in demo case
python cli.py --demo

# JSON output for integration
python cli.py -q "..." --json
```

## Tools

| Tool | Description |
|------|-------------|
| `parse_lab_values` | Extract & flag lab results from free text |
| `calculate_severity_scores` | Child-Pugh, MELD, ALBI scoring |
| `differential_diagnosis` | Ranked differential for liver diseases |
| `assess_fibrosis_stage` | Metavir F0–F4 staging |
| `get_treatment_guidelines` | AASLD/EASL treatment protocols |
| `generate_clinical_summary` | Structured clinical summary |
| `flag_urgent_findings` | Critical alerts (ALF, varices, HCC) |

## Diseases Covered

- HBV / HCV / HDV hepatitis
- NAFLD / MASH
- Alcoholic liver disease
- Autoimmune hepatitis, PBC, PSC
- Liver cirrhosis and complications (ascites, HE, varices, SBP, HRS)
- Hepatocellular carcinoma (BCLC staging)
- Acute liver failure
- Drug-induced liver injury (DILI)
- Wilson's disease, hemochromatosis

## Commands (interactive mode)

| Command | Action |
|---------|--------|
| `quit` | Exit the session |
| `reset` | Clear conversation history |
| `usage` | Show token usage and estimated cost |
| `mode` | Toggle physician ↔ patient mode |

## Architecture

```
cli.py          CLI interface (Rich formatting, interactive loop)
agent.py        Agentic loop (multi-turn tool use with Claude claude-sonnet-4-6)
tools.py        7 tool implementations (medical logic, scoring formulas)
prompts.py      System prompts with prompt caching (cache_control)
models.py       Pydantic models for structured output
```

## Disclaimer

This tool is for **clinical decision support only**. It does not replace clinical judgment, direct patient evaluation, or physician consultation. Always verify recommendations against current guidelines and individual patient circumstances.
