# HepatoAI — AI-Powered Liver Disease Clinical Decision Support

A locally-deployed clinical decision support agent for hepatology, built at the **Icahn School of Medicine at Mount Sinai**. Powered by a dual-model architecture: **Ollama qwen2.5:7b** (free, local) for clinical reasoning + **GPT-4o** for medical image analysis.

---

## Features

- **12 specialized clinical tools** grounded in AASLD, EASL, and Baveno VII guidelines
- **27 liver diseases** covered with evidence-based treatment protocols
- **47 embedded guidelines** from AASLD, EASL, Baveno VII, SMFM, WHO, RCOG, APASL, ICA
- **Structured Clinical Report** — 7 modules: abnormal values, scores, diagnosis, DDx, treatment, management, summary
- **GPT-4o image analysis** — CT, MRI, pathology, ultrasound
- **Runs 100% locally** — free, private, no data leaves your machine (except images sent to GPT-4o)
- **Web interface** — real-time streaming via Server-Sent Events (SSE)

---

## Quick Start

```bash
cd "liver agent"
pip install -r requirements.txt

# Set your OpenAI API key (for image analysis only)
export OPENAI_API_KEY=your_key_here

# Make sure Ollama is running with qwen2.5:7b
ollama pull qwen2.5:7b
ollama serve

# Launch the web app
uvicorn web_app:app --port 8000
# Then open http://localhost:8000
```

---

## System Architecture

```
Browser (index.html)
    ↕  SSE / HTTP
FastAPI Backend (web_app.py)
    ├── Ollama qwen2.5:7b   — tool calling, clinical reasoning (local · free)
    └── GPT-4o              — image analysis (CT/MRI/pathology · ~$0.01–0.03/image)
        ↓
12 Clinical Tools (tools.py)
        ↓
Knowledge Base — 47 guidelines · 27 diseases
        ↓
Clinical Report Output — 7 structured modules
```

---

## 12 Clinical Tools

| # | Tool | Description |
|---|------|-------------|
| 1 | `parse_lab_values` | Extract & flag abnormal lab results from free text |
| 2 | `flag_urgent_findings` | Critical alerts: ALF, variceal bleed, HCC, sepsis |
| 3 | `calculate_severity_scores` | Child-Pugh, MELD 3.0, ALBI, FIB-4, APRI |
| 4 | `differential_diagnosis` | Ranked differential diagnosis for liver diseases |
| 5 | `get_treatment_guidelines` | AASLD/EASL treatment protocols (27 diseases) |
| 6 | `assess_fibrosis_stage` | Metavir F0–F4 staging |
| 7 | `generate_clinical_summary` | Structured clinical summary |
| 8 | `calculate_advanced_scores` | MDF, GAHS, ABIC, Lille, PAGE-B, MELD-Na |
| 9 | `calculate_additional_clinical_scores` | eGFR (CKD-EPI), HAS-BLED, SOFA (via medcalc) |
| 10 | `calculate_amap_hcc_risk` | aMAP HCC risk score — all etiologies (Fan et al. J Hepatol 2020) |
| 11 | `assess_baveno_csph` | Baveno VII CSPH criteria — EGD decision support |
| 12 | `predict_masld_advanced_fibrosis` | MASLD fibrosis ML predictor — AUROC 0.83 (GitHub: laithomari) |

---

## Diseases Covered (27)

| Category | Diseases |
|----------|----------|
| Viral hepatitis | HBV · HCV · HDV · HEV · HAV |
| Metabolic / Immune | MASLD/NASH · ALD · AIH · DILI |
| Cholestatic | PBC · PSC · Overlap syndrome |
| Genetic | Wilson's disease · Hemochromatosis · Alpha-1 antitrypsin deficiency |
| Cirrhosis & complications | Cirrhosis · ACLF · Variceal hemorrhage · Ascites/SBP · HRS · Hepatic encephalopathy |
| Vascular | Budd-Chiari syndrome · Portal vein thrombosis |
| Malignancy | Hepatocellular carcinoma (BCLC staging) |
| Acute / Pregnancy | Acute liver failure · AFLP · Intrahepatic cholestasis of pregnancy |

---

## Clinical Report — 7 Modules

1. **Abnormal Lab Values** — flagged with severity (high / critical)
2. **Predictive Scores** — Child-Pugh, MELD 3.0, FIB-4, aMAP, SOFA, etc.
3. **Primary Diagnosis** — with supporting evidence
4. **Differential Diagnosis** — ranked alternatives
5. **Treatment Plan** — first-line, second-line, contraindications
6. **Future Management** — monitoring, surveillance, follow-up intervals
7. **Clinical Summary** — AI-generated narrative

---

## Guidelines Embedded (47)

| Organization | Count | Key Guidelines |
|---|---|---|
| AASLD | 16 | HBV 2023, HCV 2023, NAFLD 2023, HCC 2023, Cirrhosis 2021, ALF 2021 … |
| EASL | 20 | HBV 2017, HCV 2020, ALD 2018, AIH 2015, ACLF 2023, PBC 2017, PSC 2022 … |
| Other | 11 | Baveno VII 2022, SMFM 2020, RCOG, WHO, APASL 2023, ICA 2020, Alpha-1 Foundation 2020 … |

---

## Tech Stack

- **Backend:** Python 3.11 · FastAPI · uvicorn
- **LLM (text):** Ollama qwen2.5:7b — local, free, no API key required
- **LLM (vision):** GPT-4o via OpenAI API — images only
- **Clinical calculators:** medcalc library (validated Child-Pugh, MELD 3.0, FIB-4, eGFR)
- **Frontend:** Vanilla HTML/CSS/JavaScript · Server-Sent Events streaming
- **Platform:** macOS · runs fully offline (except image analysis)

---

## File Structure

```
web_app.py        FastAPI backend — agentic loop, SSE streaming, report builder
tools.py          12 clinical tools — medical logic, scoring formulas, guidelines
static/index.html Web frontend — Clinical Report display, image upload
agent.py          Legacy CLI agent
requirements.txt  Python dependencies
```

---

## Disclaimer

HepatoAI is for **clinical decision support only**. It does not replace clinical judgment, direct patient evaluation, or physician consultation. Always verify recommendations against current guidelines and individual patient circumstances. De-identify patient data before uploading images.
