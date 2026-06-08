"""FastAPI web backend for HepatoAI.
Text + tool calling: qwen2.5:7b via Ollama (free, local, private)
Image analysis:      Claude Sonnet 4.6 via Anthropic API (paid, accurate)
"""

from __future__ import annotations
import asyncio
import base64
import json
import os
import uuid
from typing import AsyncGenerator

import anthropic
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tools import TOOLS, execute_tool
from prompts import PHYSICIAN_SYSTEM_PROMPT, PATIENT_SYSTEM_PROMPT

app = FastAPI(title="HepatoAI")

# ── Model config ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL        = "qwen2.5:7b"   # Ollama: text queries + tool calling (free)
MODEL_VISION = "gpt-4o"       # OpenAI: CT/pathology image analysis (paid)

sessions: dict[str, list[dict]] = {}


def _make_client() -> OpenAI:
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",          # Ollama doesn't need a real key
    )


def _to_tools(tools: list[dict]) -> list[dict]:
    """Convert from Anthropic-style to OpenAI/Ollama tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]

# Tools the model can call — generate_clinical_summary excluded (auto-generated at end)
AGENT_TOOLS = [t for t in TOOLS if t["name"] != "generate_clinical_summary"]
OL_TOOLS = _to_tools(AGENT_TOOLS)


class ChatRequest(BaseModel):
    message: str
    session_id: str
    mode: str = "physician"
    lang: str = "en"
    image_b64: str = ""       # base64-encoded image (optional)
    image_type: str = "jpeg"  # jpeg, png, etc.


class NewSessionRequest(BaseModel):
    mode: str = "physician"


@app.post("/api/session")
def new_session(req: NewSessionRequest):
    sid = str(uuid.uuid4())
    sessions[sid] = []
    return {"session_id": sid, "mode": req.mode}


@app.delete("/api/session/{session_id}")
def reset_session(session_id: str):
    sessions[session_id] = []
    return {"ok": True}


def _system_prompt(mode: str = "physician", lang: str = "en") -> str:
    # Always physician mode, always English
    return PHYSICIAN_SYSTEM_PROMPT + "\n\nIMPORTANT: Always respond in English. You are in physician mode — use precise clinical terminology."


async def stream_agent(
    message: str, session_id: str, mode: str, lang: str,
    image_b64: str = "", image_type: str = "jpeg",
) -> AsyncGenerator[str, None]:
    client = _make_client()
    history = sessions.setdefault(session_id, [])
    loop = asyncio.get_event_loop()

    def send(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    # If image attached: analyze with Claude Sonnet 4.6, then pass findings to tool model
    if image_b64:
        yield send("tool_start", {"name": "imaging_analysis", "input": {"model": MODEL_VISION}})

        vision_prompt = (
            "You are an expert radiologist and hepatologist. "
            "Look at this medical image carefully and tell me what you see. "
            "Focus on what is actually visible — do not guess or fill in sections you cannot see.\n\n"
            "Describe:\n"
            "- What type of image is this and what region?\n"
            "- What are the most important findings?\n"
            "- Any abnormalities in the liver, spleen, bile ducts, or vessels?\n"
            "- Any masses, lesions, fluid, or lymphadenopathy?\n"
            "- Your overall impression and most likely diagnosis\n\n"
            "Be specific about what you can clearly see, and say 'not clearly visible' for things you cannot assess from this image."
        )
        if message:
            vision_prompt += f"\n\nPhysician's clinical context: {message}"

        def _gpt_vision():
            gpt = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            return gpt.chat.completions.create(
                model=MODEL_VISION,
                max_tokens=2048,
                temperature=0.3,
                messages=[
                    # Clean system prompt — no hepatology bloat, just radiology focus
                    {
                        "role": "system",
                        "content": "You are an expert radiologist and hepatologist. Analyze medical images accurately. Only describe what you can clearly see. Do not fabricate findings."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_type};base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": vision_prompt},
                        ],
                    }
                ],
            )

        vision_response = await loop.run_in_executor(None, _gpt_vision)
        imaging_findings = vision_response.choices[0].message.content or "Unable to analyze image."

        usage = vision_response.usage
        cost = (usage.prompt_tokens * 2.5 + usage.completion_tokens * 10) / 1_000_000
        yield send("tool_result", {
            "name": "imaging_analysis",
            "result": {
                "findings": imaging_findings,
                "model": MODEL_VISION,
                "tokens_used": f"in:{usage.prompt_tokens} out:{usage.completion_tokens}",
                "estimated_cost": f"${cost:.4f}",
            }
        })

        # Now pass imaging findings to the tool-calling model for clinical analysis
        combined_message = (
            f"IMAGING ANALYSIS REPORT (medgemma:4b):\n{imaging_findings}\n\n"
            f"{'ADDITIONAL CLINICAL CONTEXT: ' + message if message else ''}\n\n"
            "Based on the above imaging findings, please provide a comprehensive hepatology assessment "
            "using the available clinical tools (lab analysis, differential diagnosis, severity scoring, "
            "treatment guidelines, urgent findings check)."
        )
        user_content = combined_message
    else:
        user_content = message

    history.append({"role": "user", "content": user_content})

    system_msg = {"role": "system", "content": _system_prompt(mode, lang)}
    iterations = 0
    final_text = ""
    called_tools: set[str] = set()   # prevent duplicate tool calls
    tool_results_collected: dict = {}  # store key tool results for summary

    while iterations < 6:
        iterations += 1
        messages = [system_msg] + history

        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=MODEL,
                    max_tokens=4096,
                    temperature=0,        # 固定输出，消除随机性
                    messages=messages,
                    tools=OL_TOOLS,
                    tool_choice="auto",
                ),
            )
        except Exception as e:
            err_msg = str(e)
            if "connection" in err_msg.lower() or "refused" in err_msg.lower():
                yield send("text", {"content": "❌ 无法连接到 Ollama。请在终端运行：ollama serve"})
            else:
                yield send("text", {"content": f"❌ 错误：{err_msg[:200]}"})
            yield send("done", {"final_text": ""})
            return

        msg = response.choices[0].message
        usage = response.usage

        yield send("usage", {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "cache_read": 0,
            "cache_write": 0,
        })

        if msg.content:
            final_text += msg.content
            yield send("text", {"content": msg.content})

        # Append assistant message
        history.append({
            "role": "assistant",
            "content": msg.content or "",
            **({"tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]} if msg.tool_calls else {}),
        })

        if not msg.tool_calls or response.choices[0].finish_reason == "stop":
            break

        # Execute tools — skip duplicates and generate_clinical_summary
        SKIP_TOOLS: set[str] = set()  # generate_clinical_summary excluded from OL_TOOLS
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            is_duplicate = tool_name in called_tools
            is_skipped = tool_name in SKIP_TOOLS

            if is_skipped or is_duplicate:
                # Still need to add a tool result to keep conversation history valid
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Skipped — not needed for this response.",
                })
                continue

            called_tools.add(tool_name)

            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            yield send("tool_start", {"name": tool_name, "input": tool_input})

            result = await loop.run_in_executor(
                None, lambda tn=tool_name, ti=tool_input: execute_tool(tn, ti)
            )

            yield send("tool_result", {"name": tool_name, "result": result})

            # Collect key results for summary
            tool_results_collected[tool_name] = result

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    # ── Ensure all essential tools have been called ────────────────────────────
    # Run any missing core tools directly, regardless of what model chose to call
    from tools import (parse_lab_values, calculate_severity_scores,
                       differential_diagnosis as run_ddx, flag_urgent_findings,
                       get_treatment_guidelines)

    full_text = message  # use original message for extraction

    if "parse_lab_values" not in tool_results_collected:
        tool_results_collected["parse_lab_values"] = parse_lab_values(full_text)

    if "flag_urgent_findings" not in tool_results_collected:
        tool_results_collected["flag_urgent_findings"] = flag_urgent_findings(
            clinical_text=full_text, lab_data=full_text, imaging=full_text)

    if "differential_diagnosis" not in tool_results_collected:
        tool_results_collected["differential_diagnosis"] = run_ddx(
            symptoms=full_text, lab_findings=full_text,
            imaging_findings=full_text, patient_history=full_text)

    # Auto-compute severity scores from extracted lab values
    if "calculate_severity_scores" not in tool_results_collected:
        labs_extracted = {v["name"].lower(): v["value"]
                         for v in tool_results_collected["parse_lab_values"].get("values", [])}
        def _extract(keys):
            for k in keys:
                for lk, lv in labs_extracted.items():
                    if k in lk:
                        try: return float(lv)
                        except: pass
            return None
        bili = _extract(["bilirubin","bili"])
        alb  = _extract(["albumin"])
        inr  = _extract(["inr"])
        cr   = _extract(["creatinine","cr"])
        if any(v is not None for v in [bili, alb, inr]):
            tool_results_collected["calculate_severity_scores"] = calculate_severity_scores(
                bilirubin=bili, albumin=alb, inr=inr, creatinine=cr,
                ascites=1 if "ascites" in full_text.lower() else 0,
                encephalopathy=1 if any(w in full_text.lower() for w in ["confusion","encephalopathy","he "]) else 0,
            )

    # Auto-get treatment guidelines based on primary DDx
    if "get_treatment_guidelines" not in tool_results_collected:
        primary = tool_results_collected.get("differential_diagnosis", {}).get("primary_diagnosis", "")
        if primary:
            tool_results_collected["get_treatment_guidelines"] = get_treatment_guidelines(
                diagnosis=primary, patient_context=full_text[:300])

    # Build comprehensive clinical report from all collected tool results
    if True:  # always build report
        tr = tool_results_collected

        # 1. Abnormal lab values — include high, low, critical, abnormal
        labs = tr.get("parse_lab_values", {})
        abnormal_values = []
        for v in labs.get("values", []):
            st = v.get("status", "normal")
            if st == "normal":
                continue
            arrow = "⚠️" if "critical" in st else ("↑" if st in ("high","abnormal") else "↓")
            abnormal_values.append(
                f"{arrow} {v['name']}: {v['value']} {v.get('unit','')} — {v.get('interpretation','')}"
            )

        # 2. Scores — aggregate all scoring tools
        scores_summary = []
        sev = tr.get("calculate_severity_scores", {})
        cp = sev.get("child_pugh") if sev else None
        meld = sev.get("meld") if sev else None
        albi = sev.get("albi") if sev else None
        if cp:
            scores_summary.append(f"Child-Pugh Grade {cp.get('grade','?')} (score {cp.get('total_score','?')}/15) — 1-yr survival {cp.get('one_year_survival','?')}")
        if meld:
            scores_summary.append(f"MELD score: {meld.get('score','?')} [{meld.get('category','?')}] — 3-month mortality {meld.get('three_month_mortality','?')}")
        if albi:
            scores_summary.append(f"ALBI Grade {albi.get('grade','?')} (score {albi.get('score','?')})")
        adv = tr.get("calculate_advanced_scores", {}).get("scores", {})
        for key, label in [("fib4","FIB-4"),("meld_3","MELD 3.0"),("maddrey_df","Maddrey DF"),("page_b","PAGE-B")]:
            if key in adv:
                sc = adv[key]
                val = sc.get("score","?")
                risk = sc.get("risk") or sc.get("category") or sc.get("interpretation","")
                if risk: risk = str(risk)[:60]
                scores_summary.append(f"{label}: {val} — {risk}" if risk else f"{label}: {val}")
        amap = tr.get("calculate_amap_hcc_risk", {})
        if amap.get("amap_score"):
            scores_summary.append(f"aMAP score: {amap['amap_score']} [{amap.get('risk_category','?')}] — annual HCC risk {amap.get('annual_hcc_risk','?')}")
        masld = tr.get("predict_masld_advanced_fibrosis", {})
        if masld.get("probability_advanced_fibrosis"):
            scores_summary.append(f"MASLD Advanced Fibrosis: {masld['probability_advanced_fibrosis']} [{masld.get('risk_category','?')}]")

        # 3 & 4. Diagnosis and DDx
        ddx = tr.get("differential_diagnosis", {})
        primary_dx = ddx.get("primary_diagnosis", "")
        differentials = ddx.get("differentials", [])
        urgent_considerations = ddx.get("urgent_considerations", [])

        # 5. Treatment
        tx = tr.get("get_treatment_guidelines", {})
        treatment_first = tx.get("first_line_treatment", [])[:5]
        treatment_second = tx.get("second_line_treatment", [])[:2]
        tx_goals = tx.get("treatment_goals", [])[:3]
        tx_source = tx.get("guideline_source", "")

        # 6. Future management / monitoring
        monitoring = tx.get("monitoring_parameters", [])[:4]
        special = tx.get("special_considerations", [])[:2]
        baveno = tr.get("assess_baveno_csph", {})
        if baveno.get("egd_recommendation"):
            monitoring.insert(0, f"EGD: {baveno['egd_recommendation'][:100]}")
        if baveno.get("nsbb_recommendation"):
            monitoring.append(f"NSBB: {baveno['nsbb_recommendation'][:100]}")

        # 7. Summary — use the AI final text (clean of markdown)
        summary_text = final_text.strip() if final_text else ""

        # Urgent findings
        urgent = tr.get("flag_urgent_findings", {})
        urgent_findings = []
        if urgent.get("findings"):
            urgent_findings = [
                {"level": f["urgency_level"], "finding": f["finding"],
                 "action": f["recommended_action"], "timeframe": f["timeframe"]}
                for f in urgent["findings"]
            ]

        report = {
            "urgent_findings": urgent_findings,
            "abnormal_values": abnormal_values,
            "scores": scores_summary,
            "primary_diagnosis": primary_dx,
            "differentials": differentials,
            "urgent_considerations": urgent_considerations,
            "treatment_first_line": treatment_first,
            "treatment_second_line": treatment_second,
            "treatment_goals": tx_goals,
            "guideline_source": tx_source,
            "future_management": monitoring,
            "special_considerations": special,
            "summary": summary_text,
        }
        yield send("clinical_report", report)

    yield send("done", {"final_text": final_text})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        stream_agent(req.message, req.session_id, req.mode, req.lang, req.image_b64, req.image_type),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/models")
def list_models():
    """Return current model info."""
    return {"model": MODEL, "backend": "Ollama (local)", "base_url": OLLAMA_BASE_URL}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")
