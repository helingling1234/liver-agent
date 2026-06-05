"""Tool implementations for liver disease diagnosis and treatment agent."""

from __future__ import annotations
import math
import re
import json
from typing import Any

from models import (
    LabValue, ParsedLabResults, ChildPughScore, MeldScore, AlbiScore,
    SeverityScores, Diagnosis, DifferentialDiagnosis, FibrosisAssessment,
    TreatmentGuideline, ClinicalSummary, UrgentFinding, UrgentFindings,
)

# ─── Reference Ranges ────────────────────────────────────────────────────────

REFERENCE_RANGES: dict[str, dict] = {
    # Liver function
    "alt": {"low": 0, "high": 40, "unit": "U/L", "critical_high": 1000, "full_name": "ALT (Alanine Aminotransferase)"},
    "ast": {"low": 0, "high": 40, "unit": "U/L", "critical_high": 1000, "full_name": "AST (Aspartate Aminotransferase)"},
    "alp": {"low": 44, "high": 147, "unit": "U/L", "critical_high": 1000, "full_name": "ALP (Alkaline Phosphatase)"},
    "ggt": {"low": 0, "high": 60, "unit": "U/L", "full_name": "GGT (Gamma-Glutamyl Transferase)"},
    "total_bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL", "critical_high": 15, "full_name": "Total Bilirubin"},
    "direct_bilirubin": {"low": 0, "high": 0.3, "unit": "mg/dL", "full_name": "Direct Bilirubin"},
    "indirect_bilirubin": {"low": 0, "high": 0.9, "unit": "mg/dL", "full_name": "Indirect Bilirubin"},
    "albumin": {"low": 3.5, "high": 5.0, "unit": "g/dL", "critical_low": 2.0, "full_name": "Albumin"},
    "total_protein": {"low": 6.0, "high": 8.3, "unit": "g/dL", "full_name": "Total Protein"},
    # Coagulation
    "pt": {"low": 11, "high": 13.5, "unit": "seconds", "critical_high": 30, "full_name": "PT (Prothrombin Time)"},
    "inr": {"low": 0.8, "high": 1.2, "unit": "", "critical_high": 2.5, "full_name": "INR"},
    "ptt": {"low": 25, "high": 35, "unit": "seconds", "full_name": "PTT"},
    # CBC
    "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL", "critical_low": 7.0, "full_name": "Hemoglobin"},
    "platelets": {"low": 150, "high": 400, "unit": "×10³/μL", "critical_low": 50, "full_name": "Platelets"},
    "wbc": {"low": 4.5, "high": 11.0, "unit": "×10³/μL", "critical_high": 30, "full_name": "WBC"},
    # Renal
    "creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL", "critical_high": 5.0, "full_name": "Creatinine"},
    "bun": {"low": 7, "high": 25, "unit": "mg/dL", "full_name": "BUN"},
    "sodium": {"low": 136, "high": 145, "unit": "mEq/L", "critical_low": 125, "full_name": "Sodium"},
    # Viral markers
    "hbsag": {"type": "qualitative", "normal": "negative", "full_name": "HBsAg"},
    "hbsab": {"type": "qualitative", "normal": "negative_or_positive", "full_name": "HBsAb"},
    "hbeag": {"type": "qualitative", "normal": "negative", "full_name": "HBeAg"},
    "hbv_dna": {"low": 0, "high": 20, "unit": "IU/mL", "full_name": "HBV DNA"},
    "hcv_ab": {"type": "qualitative", "normal": "negative", "full_name": "HCV Antibody"},
    "hcv_rna": {"low": 0, "high": 15, "unit": "IU/mL", "full_name": "HCV RNA"},
    # Tumor markers
    "afp": {"low": 0, "high": 7.0, "unit": "ng/mL", "critical_high": 400, "full_name": "AFP (Alpha-Fetoprotein)"},
    # Special
    "ammonia": {"low": 15, "high": 45, "unit": "μg/dL", "critical_high": 100, "full_name": "Ammonia"},
    "ferritin": {"low": 12, "high": 300, "unit": "ng/mL", "full_name": "Ferritin"},
    "ceruloplasmin": {"low": 20, "high": 60, "unit": "mg/dL", "full_name": "Ceruloplasmin"},
    "ana": {"type": "qualitative", "normal": "negative", "full_name": "ANA"},
    "asma": {"type": "qualitative", "normal": "negative", "full_name": "ASMA (Anti-Smooth Muscle Ab)"},
    "ama": {"type": "qualitative", "normal": "negative", "full_name": "AMA (Anti-Mitochondrial Ab)"},
    "igm": {"low": 40, "high": 230, "unit": "mg/dL", "full_name": "IgM"},
    "igg": {"low": 700, "high": 1600, "unit": "mg/dL", "full_name": "IgG"},
}

# ─── Tool 1: Parse Lab Values ────────────────────────────────────────────────

def parse_lab_values(text: str) -> dict:
    """Extract and normalize lab values from free text."""
    results: list[LabValue] = []
    abnormal = 0
    critical = 0

    # Patterns to extract lab values
    patterns = [
        r"(ALT|AST|ALP|GGT|LDH)\s*[:=]?\s*([\d.]+)\s*(U/L|IU/L)?",
        r"(bilirubin|bili)\s*(total|direct|indirect)?\s*[:=]?\s*([\d.]+)\s*(mg/dL|μmol/L)?",
        r"(albumin|alb)\s*[:=]?\s*([\d.]+)\s*(g/dL|g/L)?",
        r"(INR|PT|PTT|prothrombin)\s*[:=]?\s*([\d.]+)",
        r"(creatinine|Cr)\s*[:=]?\s*([\d.]+)\s*(mg/dL)?",
        r"(AFP|alpha.fetoprotein)\s*[:=]?\s*([\d.]+)\s*(ng/mL|μg/L)?",
        r"(platelet|PLT)\s*[:=]?\s*([\d.]+)\s*(×10[³3]/[μu]L|k/[μu]L)?",
        r"(hemoglobin|Hgb|Hb)\s*[:=]?\s*([\d.]+)\s*(g/dL)?",
        r"(sodium|Na)\s*[:=]?\s*([\d.]+)\s*(mEq/L|mmol/L)?",
        r"(ammonia|NH3)\s*[:=]?\s*([\d.]+)\s*([μu]g/dL)?",
        r"(HBsAg|HBsAb|HBeAg|HCV.Ab|HCV.RNA|HBV.DNA)\s*[:=]?\s*(positive|negative|reactive|non-reactive|\d+[\d.]*)",
    ]

    # Normalize common abbreviations
    text_lower = text.lower()

    # Extract numeric values with context
    lab_extractions = []

    # ALT
    for m in re.finditer(r'\bALT\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("alt", float(m.group(1)), "U/L"))
    # AST
    for m in re.finditer(r'\bAST\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("ast", float(m.group(1)), "U/L"))
    # ALP
    for m in re.finditer(r'\bALP\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("alp", float(m.group(1)), "U/L"))
    # GGT
    for m in re.finditer(r'\bGGT\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("ggt", float(m.group(1)), "U/L"))
    # Total bilirubin
    for m in re.finditer(r'(?:total\s+)?bilirubin[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("total_bilirubin", float(m.group(1)), "mg/dL"))
    # Albumin
    for m in re.finditer(r'\balbumin\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("albumin", float(m.group(1)), "g/dL"))
    # INR
    for m in re.finditer(r'\bINR\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("inr", float(m.group(1)), ""))
    # PT
    for m in re.finditer(r'\bPT\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("pt", float(m.group(1)), "sec"))
    # Creatinine
    for m in re.finditer(r'\b(?:creatinine|Cr)\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("creatinine", float(m.group(1)), "mg/dL"))
    # AFP
    for m in re.finditer(r'\bAFP\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("afp", float(m.group(1)), "ng/mL"))
    # Platelets
    for m in re.finditer(r'\b(?:platelets?|PLT)\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("platelets", float(m.group(1)), "×10³/μL"))
    # Hemoglobin
    for m in re.finditer(r'\b(?:hemoglobin|Hgb?|Hb)\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("hemoglobin", float(m.group(1)), "g/dL"))
    # Sodium
    for m in re.finditer(r'\b(?:sodium|Na)\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("sodium", float(m.group(1)), "mEq/L"))
    # Ammonia
    for m in re.finditer(r'\b(?:ammonia|NH3)\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("ammonia", float(m.group(1)), "μg/dL"))
    # Sodium
    for m in re.finditer(r'\b(?:sodium|Na)\b[\s:=]*([\d.]+)', text, re.IGNORECASE):
        lab_extractions.append(("sodium", float(m.group(1)), "mEq/L"))
    # HBsAg
    for m in re.finditer(r'\bHBsAg\b[\s:=]*(positive|negative|reactive|non-reactive)', text, re.IGNORECASE):
        lab_extractions.append(("hbsag", m.group(1).lower(), ""))
    # HCV Ab
    for m in re.finditer(r'\bHCV[\s-]?Ab\b[\s:=]*(positive|negative|reactive|non-reactive)', text, re.IGNORECASE):
        lab_extractions.append(("hcv_ab", m.group(1).lower(), ""))
    # HBV DNA
    for m in re.finditer(r'\bHBV[\s-]?DNA\b[\s:=]*([\d.]+(?:\s*[×x]\s*10\^?\d+)?)', text, re.IGNORECASE):
        lab_extractions.append(("hbv_dna", m.group(1), "IU/mL"))
    # HCV RNA
    for m in re.finditer(r'\bHCV[\s-]?RNA\b[\s:=]*([\d.]+(?:\s*[×x]\s*10\^?\d+)?|(?:un)?detectable)', text, re.IGNORECASE):
        lab_extractions.append(("hcv_rna", m.group(1), "IU/mL"))

    for key, value, unit in lab_extractions:
        ref = REFERENCE_RANGES.get(key, {})
        full_name = ref.get("full_name", key.upper())
        status = "normal"
        interpretation = ""

        if ref.get("type") == "qualitative":
            if isinstance(value, str):
                if value in ("positive", "reactive"):
                    status = "abnormal"
                    interpretation = f"{full_name} is {value} — clinically significant"
                else:
                    interpretation = f"{full_name} is {value}"
            ref_range = f"Normal: {ref.get('normal', 'negative')}"
        else:
            try:
                v = float(value)
                low = ref.get("low", 0)
                high = ref.get("high", float('inf'))
                critical_high = ref.get("critical_high")
                critical_low = ref.get("critical_low")

                if critical_high and v > critical_high:
                    status = "critical_high"
                    critical += 1
                    interpretation = f"CRITICALLY elevated — requires immediate attention"
                elif critical_low and v < critical_low:
                    status = "critical_low"
                    critical += 1
                    interpretation = f"CRITICALLY low — requires immediate attention"
                elif v > high:
                    status = "high"
                    abnormal += 1
                    fold = round(v / high, 1) if high > 0 else "N/A"
                    interpretation = f"Elevated ({fold}× ULN)"
                elif v < low:
                    status = "low"
                    abnormal += 1
                    interpretation = f"Below normal range"
                else:
                    interpretation = "Within normal limits"

                ref_range = f"{low}–{high} {unit}".strip()
            except (ValueError, TypeError):
                ref_range = ""

        results.append(LabValue(
            name=full_name,
            value=value,
            unit=unit,
            reference_range=ref_range,
            status=status,
            interpretation=interpretation,
        ))

    # Generate summary
    if not results:
        summary = "No specific lab values detected in the text. Please provide values in format: 'ALT: 45 U/L'."
    else:
        summary_parts = [f"Extracted {len(results)} lab values."]
        if critical > 0:
            summary_parts.append(f"⚠️  {critical} CRITICAL value(s) detected.")
        if abnormal > 0:
            summary_parts.append(f"{abnormal} abnormal value(s) detected.")
        summary = " ".join(summary_parts)

    return ParsedLabResults(
        values=results,
        abnormal_count=abnormal,
        critical_count=critical,
        summary=summary,
    ).model_dump()


# ─── Tool 2: Calculate Severity Scores ──────────────────────────────────────

def _child_pugh(bilirubin: float, albumin: float, inr: float,
                ascites: int, encephalopathy: int) -> ChildPughScore:
    """Compute Child-Pugh score (ascites: 0=none,1=mild,2=moderate/severe; encephalopathy: 0-4)."""
    # Bilirubin (mg/dL) points
    if bilirubin < 2:
        bili_pts = 1
    elif bilirubin <= 3:
        bili_pts = 2
    else:
        bili_pts = 3

    # Albumin (g/dL) points
    if albumin > 3.5:
        alb_pts = 1
    elif albumin >= 2.8:
        alb_pts = 2
    else:
        alb_pts = 3

    # INR points
    if inr < 1.7:
        inr_pts = 1
    elif inr <= 2.3:
        inr_pts = 2
    else:
        inr_pts = 3

    # Ascites (0=none → 1pt, 1=mild → 2pts, 2=moderate/severe → 3pts)
    ascites_pts = min(ascites + 1, 3) if ascites >= 0 else 1

    # Encephalopathy (0=none → 1pt, 1-2=mild → 2pts, 3-4=severe → 3pts)
    if encephalopathy == 0:
        enc_pts = 1
    elif encephalopathy <= 2:
        enc_pts = 2
    else:
        enc_pts = 3

    total = bili_pts + alb_pts + inr_pts + ascites_pts + enc_pts

    if total <= 6:
        grade = "A"
        one_yr = "100%"
        two_yr = "85%"
    elif total <= 9:
        grade = "B"
        one_yr = "81%"
        two_yr = "57%"
    else:
        grade = "C"
        one_yr = "45%"
        two_yr = "35%"

    return ChildPughScore(
        total_score=total,
        grade=grade,
        bilirubin_points=bili_pts,
        albumin_points=alb_pts,
        inr_points=inr_pts,
        ascites_points=ascites_pts,
        encephalopathy_points=enc_pts,
        one_year_survival=one_yr,
        two_year_survival=two_yr,
        interpretation=(
            f"Child-Pugh {grade} (score {total}/15). "
            f"1-year survival: {one_yr}, 2-year survival: {two_yr}. "
            f"{'Compensated cirrhosis — relatively preserved liver function.' if grade == 'A' else 'Decompensated cirrhosis — significant hepatic dysfunction.' if grade == 'C' else 'Moderately decompensated cirrhosis.'}"
        ),
    )


def _meld(bilirubin: float, inr: float, creatinine: float) -> MeldScore:
    """MELD score = 3.78×ln(bilirubin) + 11.2×ln(INR) + 9.57×ln(creatinine) + 6.43"""
    bili = max(bilirubin, 1.0)
    inr_v = max(inr, 1.0)
    cr = max(min(creatinine, 4.0), 1.0)  # cap creatinine at 4.0

    score = 3.78 * math.log(bili) + 11.2 * math.log(inr_v) + 9.57 * math.log(cr) + 6.43
    score = round(score)
    score = max(score, 6)

    if score < 10:
        category = "Low"
        mortality = "<2%"
    elif score < 20:
        category = "Moderate"
        mortality = "6–20%"
    elif score < 30:
        category = "High"
        mortality = "20–52%"
    elif score < 40:
        category = "Very High"
        mortality = "52–71%"
    else:
        category = "Critical"
        mortality = ">71%"

    return MeldScore(
        score=score,
        category=category,
        three_month_mortality=mortality,
        interpretation=(
            f"MELD score: {score} ({category} risk). "
            f"3-month mortality: {mortality}. "
            f"{'Transplant listing should be considered.' if score >= 15 else 'Monitor closely; reassess in 3–6 months.'}"
        ),
    )


def _albi(albumin_g_dl: float, bilirubin_umol: float | None = None,
          bilirubin_mg_dl: float | None = None) -> AlbiScore:
    """ALBI = (log10[bilirubin in μmol/L] × 0.66) + (albumin in g/L × -0.085)"""
    # Convert if needed
    if bilirubin_umol is None and bilirubin_mg_dl is not None:
        bilirubin_umol = bilirubin_mg_dl * 17.1
    if bilirubin_umol is None:
        raise ValueError("Bilirubin required")

    albumin_g_l = albumin_g_dl * 10
    score = (math.log10(max(bilirubin_umol, 0.1)) * 0.66) + (albumin_g_l * -0.085)
    score = round(score, 2)

    if score <= -2.60:
        grade = 1
        desc = "Good hepatic function — low risk"
    elif score <= -1.39:
        grade = 2
        desc = "Intermediate hepatic function — moderate risk"
    else:
        grade = 3
        desc = "Poor hepatic function — high risk"

    return AlbiScore(
        score=score,
        grade=grade,
        interpretation=f"ALBI score: {score} (Grade {grade}). {desc}.",
    )


def calculate_severity_scores(
    bilirubin: float | None = None,
    albumin: float | None = None,
    inr: float | None = None,
    creatinine: float | None = None,
    ascites: int = 0,
    encephalopathy: int = 0,
) -> dict:
    """Calculate Child-Pugh, MELD, and ALBI scores."""
    missing = []
    cp = None
    meld = None
    albi = None

    if bilirubin is None:
        missing.append("bilirubin (mg/dL)")
    if albumin is None:
        missing.append("albumin (g/dL)")
    if inr is None:
        missing.append("INR")
    if creatinine is None:
        missing.append("creatinine (mg/dL)")

    recs = []

    if bilirubin is not None and albumin is not None and inr is not None:
        cp = _child_pugh(bilirubin, albumin, inr, ascites, encephalopathy)

    if bilirubin is not None and inr is not None and creatinine is not None:
        meld = _meld(bilirubin, inr, creatinine)

    if bilirubin is not None and albumin is not None:
        albi = _albi(albumin_g_dl=albumin, bilirubin_mg_dl=bilirubin)

    if cp is None:
        recs.append("Provide bilirubin, albumin, and INR for Child-Pugh score")
    if meld is None:
        recs.append("Provide bilirubin, INR, and creatinine for MELD score")

    return SeverityScores(
        child_pugh=cp,
        meld=meld,
        albi=albi,
        missing_values=missing,
        recommendations=recs,
    ).model_dump()


# ─── Tool 3: Differential Diagnosis ─────────────────────────────────────────

DISEASE_PATTERNS: dict[str, dict] = {
    "HBV Hepatitis": {
        "markers": ["hbsag positive", "hbv dna", "hbeag"],
        "labs": {"alt": (">40", "hepatocellular"), "ast": (">40", "hepatocellular")},
        "keywords": ["hepatitis b", "hbv", "hbsag", "cirrhosis"],
    },
    "HCV Hepatitis": {
        "markers": ["hcv ab positive", "hcv rna detectable"],
        "labs": {"alt": (">40", "hepatocellular")},
        "keywords": ["hepatitis c", "hcv", "iv drug", "transfusion"],
    },
    "NAFLD/NASH": {
        "keywords": ["obesity", "diabetes", "metabolic syndrome", "fatty liver", "steatosis", "nafld", "nash"],
        "labs": {"alt": (">40",), "ast": (">40",), "ggt": (">60",)},
        "imaging": ["steatosis", "fatty", "echogenic"],
    },
    "Alcoholic Liver Disease": {
        "keywords": ["alcohol", "drinking", "etoh", "ald"],
        "labs": {"ast": (">40",), "ggt": (">120",)},
        "patterns": ["ast:alt ratio >2"],
    },
    "Autoimmune Hepatitis": {
        "keywords": ["autoimmune", "ana", "asma", "young women"],
        "markers": ["ana positive", "asma positive", "elevated igg"],
        "labs": {"igg": (">1600",), "alt": (">40",)},
    },
    "Primary Biliary Cholangitis (PBC)": {
        "keywords": ["pbc", "cholestasis", "pruritus", "middle-aged women"],
        "markers": ["ama positive"],
        "labs": {"alp": (">147",), "ggt": (">60",)},
    },
    "Primary Sclerosing Cholangitis (PSC)": {
        "keywords": ["psc", "ibd", "ulcerative colitis", "crohn", "biliary strictures"],
        "labs": {"alp": (">147",), "ggt": (">60",)},
        "imaging": ["beaded appearance", "strictures", "mrcp"],
    },
    "Liver Cirrhosis": {
        "keywords": ["cirrhosis", "portal hypertension", "varices", "ascites", "splenomegaly"],
        "labs": {"platelets": ("<150",), "albumin": ("<3.5",), "inr": (">1.2",)},
        "imaging": ["nodular liver", "splenomegaly", "ascites", "varices"],
    },
    "Hepatocellular Carcinoma (HCC)": {
        "keywords": ["hcc", "hepatoma", "liver cancer", "mass"],
        "markers": ["afp elevated", "afp >400"],
        "labs": {"afp": (">7",)},
        "imaging": ["arterial enhancement", "washout", "mass", "nodule"],
    },
    "Acute Liver Failure": {
        "keywords": ["acute liver failure", "alf", "encephalopathy", "coagulopathy"],
        "labs": {"inr": (">1.5",), "alt": (">1000",), "bilirubin": (">3",)},
        "patterns": ["rapid onset", "no prior liver disease"],
    },
    "Drug-Induced Liver Injury (DILI)": {
        "keywords": ["medication", "drug", "acetaminophen", "antibiotics", "herbal", "dili"],
        "labs": {"alt": (">40",)},
        "patterns": ["temporal relationship with drug"],
    },
    "Wilson's Disease": {
        "keywords": ["wilson", "copper", "neurological", "young patient", "kayser-fleischer"],
        "markers": ["low ceruloplasmin", "elevated urine copper"],
        "labs": {"ceruloplasmin": ("<20",)},
    },
    "Hemochromatosis": {
        "keywords": ["hemochromatosis", "iron overload", "bronze diabetes", "hfe"],
        "labs": {"ferritin": (">300",)},
        "markers": ["transferrin saturation >45%"],
    },
}


def differential_diagnosis(
    symptoms: str = "",
    lab_findings: str = "",
    imaging_findings: str = "",
    patient_history: str = "",
) -> dict:
    """Generate ranked differential diagnosis for liver disease."""
    all_text = f"{symptoms} {lab_findings} {imaging_findings} {patient_history}".lower()
    scores: dict[str, int] = {}

    for disease, patterns in DISEASE_PATTERNS.items():
        score = 0
        for kw in patterns.get("keywords", []):
            if kw in all_text:
                score += 3
        for marker in patterns.get("markers", []):
            if marker in all_text:
                score += 5
        for img_term in patterns.get("imaging", []):
            if img_term in all_text:
                score += 2
        scores[disease] = score

    # Sort by score
    sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_diseases = [(d, s) for d, s in sorted_diseases if s > 0][:5]

    if not top_diseases:
        # Return generic differentials
        top_diseases = [
            ("NAFLD/NASH", 2),
            ("Viral Hepatitis (HBV/HCV)", 1),
            ("Drug-Induced Liver Injury (DILI)", 1),
        ]

    differentials = []
    for rank, (disease, score) in enumerate(top_diseases, 1):
        if score >= 8:
            prob = "high"
        elif score >= 4:
            prob = "moderate"
        else:
            prob = "low"

        supporting = []
        against = []
        next_steps = []

        if "HBV" in disease or "Hepatitis B" in disease:
            supporting.append("HBsAg positive / HBV DNA detectable") if "hbsag" in all_text else None
            next_steps = ["HBV DNA quantification", "HBe Ag/Ab", "Liver biopsy if needed", "Fibroscan"]
        elif "HCV" in disease or "Hepatitis C" in disease:
            supporting.append("HCV Ab positive / HCV RNA detectable") if "hcv" in all_text else None
            next_steps = ["HCV RNA quantification", "Genotyping", "Fibroscan", "Consider DAA therapy"]
        elif "NAFLD" in disease:
            supporting.append("Metabolic risk factors present") if any(k in all_text for k in ["obese", "diabetes", "fatty"]) else None
            next_steps = ["Liver ultrasound", "Fibroscan", "Metabolic workup", "FIB-4 score"]
        elif "Alcoholic" in disease:
            next_steps = ["AUDIT-C questionnaire", "GGT", "AST:ALT ratio", "Liver biopsy"]
        elif "Autoimmune" in disease:
            next_steps = ["ANA", "ASMA", "Anti-LKM-1", "IgG levels", "Liver biopsy"]
        elif "HCC" in disease:
            next_steps = ["Multiphasic CT/MRI", "AFP trend", "BCLC staging", "MDT referral"]
        elif "Cirrhosis" in disease:
            next_steps = ["Upper endoscopy for varices", "Child-Pugh/MELD scoring", "Ascites workup", "HCC surveillance"]
        elif "Acute Liver Failure" in disease:
            next_steps = ["IMMEDIATE ICU assessment", "Transplant center contact", "N-acetylcysteine if acetaminophen", "Toxicology screen"]
        else:
            next_steps = ["Additional testing required"]

        supporting = [s for s in supporting if s]
        differentials.append(Diagnosis(
            rank=rank,
            condition=disease,
            probability=prob,
            supporting_evidence=supporting,
            against_evidence=against,
            next_steps=next_steps,
        ))

    urgent = []
    if "acute liver failure" in all_text or "alf" in all_text:
        urgent.append("⚠️  Possible Acute Liver Failure — URGENT evaluation required")
    if "hcc" in all_text or "hepatoma" in all_text or "liver cancer" in all_text:
        urgent.append("⚠️  HCC suspected — urgent multiphase imaging and MDT referral")

    return DifferentialDiagnosis(
        primary_diagnosis=top_diseases[0][0] if top_diseases else "Undetermined",
        differentials=differentials,
        key_distinguishing_features=[
            "Viral serology (HBsAg, HCV Ab) to distinguish viral from non-viral",
            "AST:ALT ratio >2 suggests alcoholic liver disease",
            "Elevated ALP/GGT with normal ALT suggests cholestatic pattern",
            "AFP >400 ng/mL strongly suggests HCC",
            "Rapid onset of encephalopathy + coagulopathy suggests ALF",
        ],
        urgent_considerations=urgent,
    ).model_dump()


# ─── Tool 4: Assess Fibrosis Stage ──────────────────────────────────────────

def assess_fibrosis_stage(
    lab_data: str = "",
    imaging_findings: str = "",
    pathology: str = "",
    platelets: float | None = None,
    alt: float | None = None,
    age: int | None = None,
) -> dict:
    """Estimate Metavir fibrosis stage (F0–F4)."""
    all_text = f"{lab_data} {imaging_findings} {pathology}".lower()
    stage = "F0"
    confidence = "low"
    evidence = []
    confirmatory = []

    # Direct pathology evidence (highest confidence)
    if "f4" in all_text or "cirrhosis" in all_text or "cirrhotic" in all_text:
        stage = "F4"
        confidence = "high"
        evidence.append("Biopsy/pathology indicates cirrhosis (F4)")
    elif "f3" in all_text or "bridging fibrosis" in all_text:
        stage = "F3"
        confidence = "high"
        evidence.append("Biopsy/pathology indicates bridging fibrosis (F3)")
    elif "f2" in all_text or "periportal fibrosis" in all_text:
        stage = "F2"
        confidence = "high"
        evidence.append("Biopsy/pathology indicates periportal fibrosis (F2)")
    elif "f1" in all_text or "portal fibrosis" in all_text or "mild fibrosis" in all_text:
        stage = "F1"
        confidence = "high"
        evidence.append("Biopsy/pathology indicates mild fibrosis (F1)")
    elif "f0" in all_text or "no fibrosis" in all_text:
        stage = "F0"
        confidence = "high"
        evidence.append("Biopsy/pathology: no significant fibrosis")
    else:
        # Estimate from indirect markers
        fib4 = None
        if platelets is not None and alt is not None and age is not None:
            ast_match = re.search(r'AST[\s:=]*([\d.]+)', lab_data, re.IGNORECASE)
            if ast_match:
                ast_val = float(ast_match.group(1))
                fib4 = (age * ast_val) / (platelets * math.sqrt(alt))
                evidence.append(f"FIB-4 index: {round(fib4, 2)}")

                if fib4 < 1.30:
                    stage = "F0-F1"
                    confidence = "moderate"
                    evidence.append("FIB-4 <1.30 — low probability of advanced fibrosis")
                elif fib4 <= 2.67:
                    stage = "F1-F2"
                    confidence = "low"
                    evidence.append("FIB-4 1.30–2.67 — indeterminate; biopsy recommended")
                else:
                    stage = "F3-F4"
                    confidence = "moderate"
                    evidence.append("FIB-4 >2.67 — high probability of advanced fibrosis")

        # Imaging clues
        if "splenomegaly" in all_text:
            evidence.append("Splenomegaly suggests portal hypertension (≥F3-F4)")
            if stage in ("F0", "F1", "F0-F1", "F1-F2"):
                stage = "F3-F4"
                confidence = "moderate"
        if "ascites" in all_text:
            evidence.append("Ascites indicates decompensated cirrhosis (F4)")
            stage = "F4"
            confidence = "moderate"
        if "nodular" in all_text:
            evidence.append("Nodular liver surface suggests cirrhosis (F4)")
            stage = "F4"
            confidence = "moderate"
        if any(t in all_text for t in ["kpa", "fibroscan", "lse"]):
            for m in re.finditer(r'(\d+(?:\.\d+)?)\s*kpa', all_text):
                lse = float(m.group(1))
                if lse < 7.0:
                    stage = "F0-F1"; evidence.append(f"Fibroscan {lse} kPa — no/minimal fibrosis")
                elif lse < 9.5:
                    stage = "F2"; evidence.append(f"Fibroscan {lse} kPa — significant fibrosis")
                elif lse < 12.5:
                    stage = "F3"; evidence.append(f"Fibroscan {lse} kPa — advanced fibrosis")
                else:
                    stage = "F4"; evidence.append(f"Fibroscan {lse} kPa — cirrhosis")
                confidence = "moderate"

        if not evidence:
            stage = "F0-F2"
            evidence.append("Insufficient data for precise staging")
            confirmatory = ["Liver biopsy (gold standard)", "Fibroscan/Transient Elastography", "FIB-4 calculation", "APRI score"]

    stage_descriptions = {
        "F0": "No fibrosis",
        "F1": "Mild fibrosis — portal fibrosis without septa",
        "F2": "Significant fibrosis — portal fibrosis with few septa",
        "F3": "Severe fibrosis — numerous septa without cirrhosis",
        "F4": "Cirrhosis — complete fibrosis with regenerative nodules",
        "F0-F1": "No to minimal fibrosis",
        "F1-F2": "Mild to moderate fibrosis",
        "F3-F4": "Advanced fibrosis or cirrhosis",
        "F0-F2": "No to moderate fibrosis",
    }

    confirmatory = confirmatory or (["Confirm with Fibroscan or liver biopsy"] if confidence != "high" else [])

    return FibrosisAssessment(
        stage=stage,
        description=stage_descriptions.get(stage, "See description"),
        confidence=confidence,
        supporting_evidence=evidence,
        recommended_confirmatory_tests=confirmatory,
        clinical_significance=(
            "Significant fibrosis/cirrhosis — implement surveillance and prophylaxis protocols"
            if any(f in stage for f in ["F3", "F4", "3-4"])
            else "Early-stage — focus on treating underlying cause and preventing progression"
        ),
    ).model_dump()


# ─── Tool 5: Treatment Guidelines ───────────────────────────────────────────

TREATMENT_DB: dict[str, dict] = {
    "hepatitis b": {
        "first_line": [
            "Tenofovir disoproxil fumarate (TDF) 300 mg/day OR Tenofovir alafenamide (TAF) 25 mg/day",
            "Entecavir 0.5 mg/day (1 mg/day if lamivudine-resistant)",
            "For HBeAg-positive with immune-tolerant phase: consider watchful waiting",
            "Pegylated interferon-alpha 180 μg/week × 48 weeks (selected patients)",
        ],
        "second_line": [
            "If TDF failure: TAF; if TAF failure: switch/add entecavir",
            "For resistance: combination therapy or switch to alternative nucleoside",
        ],
        "goals": [
            "Suppress HBV DNA to undetectable levels",
            "Normalize ALT",
            "Prevent cirrhosis and HCC",
            "Achieve HBeAg seroconversion (if HBeAg+)",
            "Consider HBsAg loss as ultimate treatment goal",
        ],
        "monitoring": [
            "HBV DNA and ALT every 3–6 months",
            "HBeAg/HBsAg status every 6–12 months",
            "Renal function (TDF): creatinine/phosphate",
            "HCC surveillance if cirrhosis or family history: ultrasound ± AFP every 6 months",
        ],
        "special": [
            "Pregnancy: TDF preferred (Category B); entecavir contraindicated",
            "HIV coinfection: use TDF+FTC or TAF+FTC as part of HAART",
            "Cirrhosis: avoid interferon; use nucleoside analogues indefinitely",
            "Pre-chemotherapy/immunosuppression: prophylactic antiviral therapy",
        ],
        "source": "AASLD HBV Guidelines 2023; EASL HBV Guidelines 2017",
    },
    "hepatitis c": {
        "first_line": [
            "Genotype 1a/1b, 2, 3, 4, 5, 6 — Pan-genotypic regimens preferred:",
            "Sofosbuvir/Velpatasvir (Epclusa) 400/100 mg/day × 12 weeks",
            "Glecaprevir/Pibrentasvir (Mavyret) 100/40 mg 3 tabs/day × 8–12 weeks",
            "Sofosbuvir/Velpatasvir/Voxilaprevir (Vosevi) for retreatment × 12 weeks",
        ],
        "second_line": [
            "Sofosbuvir + Daclatasvir (if pan-genotypic not available)",
            "Ledipasvir/Sofosbuvir (Harvoni) for GT 1, 4, 5, 6",
        ],
        "goals": [
            "Achieve SVR12 (HCV RNA undetectable 12 weeks post-treatment)",
            "SVR = 'cure' in >95% of patients",
            "Prevent cirrhosis progression and HCC",
            "HCC surveillance still required after SVR if cirrhosis present",
        ],
        "monitoring": [
            "HCV RNA at treatment week 4, end of treatment, 12 weeks post-treatment (SVR12)",
            "LFTs during treatment",
            "Renal function with sofosbuvir-based regimens",
            "Check for drug interactions (rifampin, carbamazepine, PPI interactions)",
        ],
        "special": [
            "Cirrhosis: Child-Pugh A — treat with standard DAA; B/C — avoid NS3/4A protease inhibitors",
            "Renal impairment (eGFR <30): avoid sofosbuvir; use glecaprevir/pibrentasvir",
            "HIV coinfection: check drug interactions with ARVs",
            "Pregnancy: defer treatment until after delivery if possible",
        ],
        "source": "AASLD/IDSA HCV Guidance 2023; EASL HCV Guidelines 2020",
    },
    "nafld": {
        "first_line": [
            "Weight loss: 7–10% body weight reduction (primary intervention)",
            "Mediterranean diet",
            "Aerobic exercise: 150–300 min/week moderate intensity",
            "Control metabolic risk factors: diabetes, dyslipidemia, hypertension",
            "Vitamin E 800 IU/day (non-diabetic adults with biopsy-proven NASH)",
            "Pioglitazone 30 mg/day for patients with T2DM and NASH",
        ],
        "second_line": [
            "GLP-1 agonists (semaglutide, liraglutide) for obese patients with T2DM",
            "Bariatric surgery for BMI >35 with metabolic complications",
            "Avoid alcohol, hepatotoxic drugs",
            "Consider clinical trials for advanced NASH",
        ],
        "goals": [
            "Improve liver histology (reduce steatosis, inflammation, ballooning)",
            "Prevent progression to fibrosis and cirrhosis",
            "Reduce cardiovascular risk",
            "Normalize liver enzymes",
        ],
        "monitoring": [
            "LFTs every 6–12 months",
            "Fibroscan or FIB-4 annually for fibrosis staging",
            "Metabolic panel (HbA1c, lipids, BP)",
            "HCC surveillance if cirrhosis: US ± AFP every 6 months",
        ],
        "special": [
            "No approved pharmacotherapy for NASH in 2024 (resmetirom approved 2024 for F2-F3)",
            "Avoid statins with very elevated transaminases (>3× ULN)",
            "Metformin does not improve NASH but safe to continue for T2DM",
        ],
        "source": "AASLD NAFLD Practice Guidance 2023; EASL-EASD-EASO Guidelines 2016",
    },
    "alcoholic liver disease": {
        "first_line": [
            "Abstinence from alcohol (cornerstone of treatment)",
            "Nutritional support: high-calorie, high-protein diet",
            "Thiamine 100 mg IV/IM for 3–5 days (prevent Wernicke encephalopathy)",
            "Severe AH (MDF ≥32 or MELD ≥21): Prednisolone 40 mg/day × 28 days",
            "N-acetylcysteine as adjunct to prednisolone in severe AH",
        ],
        "second_line": [
            "Pentoxifylline 400 mg TID × 4 weeks (alternative if steroids contraindicated)",
            "Early liver transplantation for selected non-responders",
            "Naltrexone 50 mg/day or acamprosate 666 mg TID for alcohol dependence",
            "Baclofen for alcohol dependence in cirrhotics",
        ],
        "goals": [
            "Achieve and maintain alcohol abstinence",
            "Prevent and treat nutritional deficiencies",
            "Treat complications (ascites, variceal bleeding, HE)",
            "Liver transplantation if end-stage liver disease",
        ],
        "monitoring": [
            "LFTs weekly in acute phase, then monthly",
            "Lille score at day 7 (predicts steroid response): >0.45 = non-responder",
            "Nutritional status assessment",
            "Alcohol use: biochemical markers (CDT, GGT, PEth)",
        ],
        "special": [
            "Do NOT use steroids if infection, GI bleeding, or renal failure",
            "Lille model: score >0.45 at day 7 = stop steroids",
            "6-month abstinence rule for transplantation (varies by center)",
        ],
        "source": "AASLD Alcoholic Hepatitis Guidelines 2018; EASL ALD Guidelines 2018",
    },
    "autoimmune hepatitis": {
        "first_line": [
            "Prednisone 40–60 mg/day + Azathioprine 50–150 mg/day",
            "OR Budesonide 9 mg/day + Azathioprine 1–2 mg/kg/day (mild-moderate AIH)",
            "Taper prednisone over 6–12 weeks to maintenance 5–10 mg/day",
        ],
        "second_line": [
            "Mycophenolate mofetil 1–3 g/day (azathioprine intolerant)",
            "Tacrolimus or cyclosporine (refractory cases)",
            "Rituximab for treatment-resistant disease",
        ],
        "goals": [
            "Complete biochemical remission (normal LFTs and IgG)",
            "Histological remission",
            "Prevent cirrhosis",
            "Minimize steroid side effects",
        ],
        "monitoring": [
            "LFTs and IgG levels monthly for 3 months, then every 3 months",
            "CBC (azathioprine): monitor for cytopenias",
            "TPMT genotyping before azathioprine (risk of myelotoxicity)",
            "Bone density (steroid-associated osteoporosis)",
            "Annual ophthalmology (steroid-induced cataracts)",
        ],
        "special": [
            "Check TPMT activity or genotype before azathioprine",
            "Pregnancy: azathioprine generally safe; prednisone acceptable",
            "Avoid azathioprine in acute severe AIH with jaundice",
        ],
        "source": "AASLD AIH Guidelines 2019; EASL AIH Guidelines 2015",
    },
    "liver cirrhosis": {
        "first_line": [
            "ASCITES: Na restriction (<2g/day) + spironolactone 100 mg/day ± furosemide 40 mg/day",
            "Large-volume paracentesis for tense ascites + albumin 6–8 g/L removed",
            "VARICES: Non-selective beta-blockers (propranolol or carvedilol) for primary prophylaxis",
            "HEPATIC ENCEPHALOPATHY: Lactulose 30–45 mL BID-TID titrated to 2–3 soft stools/day",
            "HE: Rifaximin 550 mg BID for secondary prophylaxis",
            "SBP: Cefotaxime 2g IV q8h × 5 days; Albumin 1.5 g/kg Day 1, 1 g/kg Day 3",
        ],
        "second_line": [
            "Refractory ascites: TIPS (transjugular intrahepatic portosystemic shunt)",
            "Acute variceal bleeding: banding + terlipressin/octreotide + antibiotics",
            "HRS (hepatorenal syndrome): Norepinephrine + albumin or terlipressin",
            "Liver transplantation for end-stage disease",
        ],
        "goals": [
            "Prevent decompensation",
            "Treat complications as they arise",
            "HCC surveillance: US ± AFP every 6 months",
            "Evaluate for liver transplantation (MELD ≥15)",
        ],
        "monitoring": [
            "LFTs, INR, albumin, creatinine every 3–6 months",
            "Child-Pugh and MELD scoring",
            "EGD for varices at diagnosis, then every 1–3 years",
            "Abdominal US for HCC every 6 months",
            "Nutritional assessment",
        ],
        "special": [
            "NSAIDs contraindicated (worsen renal function and HRS)",
            "Aminoglycosides avoid (nephrotoxic)",
            "Sedatives with caution (precipitate HE)",
            "Na+ <125 mEq/L: consider tolvaptan or fluid restriction",
        ],
        "source": "AASLD Cirrhosis Practice Guidance 2021; EASL Decompensated Cirrhosis Guidelines 2018",
    },
    "hcc": {
        "first_line": [
            "BCLC 0/A (Very early/Early): Surgical resection OR Liver transplantation (Milan criteria) OR Ablation (RFA/MWA)",
            "BCLC B (Intermediate): TACE (trans-arterial chemoembolization)",
            "BCLC C (Advanced): Sorafenib 400 mg BID OR Lenvatinib 12 mg/day (first-line systemic)",
            "Atezolizumab + Bevacizumab (IMbrave150): preferred first-line systemic therapy",
            "Durvalumab + Tremelimumab (HIMALAYA): alternative first-line option",
        ],
        "second_line": [
            "Regorafenib, cabozantinib, ramucirumab (post-sorafenib)",
            "Pembrolizumab (post-sorafenib)",
            "Nivolumab ± ipilimumab (post-sorafenib)",
        ],
        "goals": [
            "Curative intent if BCLC 0/A",
            "Extend survival and maintain quality of life",
            "Milan criteria for transplant: single ≤5cm or ≤3 nodules each ≤3cm",
            "HCC surveillance post-resection/ablation",
        ],
        "monitoring": [
            "AFP and multiphasic CT/MRI every 3 months post-treatment",
            "LFTs and liver function",
            "Systemic therapy toxicities (hypertension, hand-foot syndrome, etc.)",
        ],
        "special": [
            "All HCC patients with HBV/HCV: treat underlying viral hepatitis",
            "Child-Pugh C/BCLC D: best supportive care; palliative approach",
            "LI-RADS classification for imaging characterization",
        ],
        "source": "AASLD HCC Guidance 2023; BCLC Staging System 2022",
    },
    "acute liver failure": {
        "first_line": [
            "IMMEDIATE ICU admission and monitoring",
            "Acetaminophen-induced: N-acetylcysteine (NAC) IV — 150 mg/kg over 1h, then 50 mg/kg over 4h, then 100 mg/kg over 16h",
            "Airway management: intubate if GCS <8 or deteriorating",
            "Cerebral edema: Mannitol 0.5–1 g/kg IV bolus; Target ICP <20 mmHg",
            "Coagulopathy: Vitamin K 10 mg IV; FFP only if invasive procedure",
            "Hypoglycemia: Dextrose infusion; target glucose 140–180 mg/dL",
            "Antibiotics: prophylactic if high infection risk",
        ],
        "second_line": [
            "Emergency liver transplantation (King's College Criteria or MELD >30)",
            "Molecular adsorbent recirculating system (MARS) as bridge to transplant",
            "Continuous renal replacement therapy for AKI",
        ],
        "goals": [
            "Prevent/treat complications: cerebral edema, infection, renal failure, coagulopathy",
            "Early transplant evaluation",
            "Identify and treat specific etiology",
        ],
        "monitoring": [
            "INR, bilirubin, creatinine, glucose every 4–6 hours",
            "Neurological assessment (Glasgow Coma Scale) hourly",
            "ICP monitoring if grade III/IV encephalopathy",
            "Continuous hemodynamic monitoring",
        ],
        "special": [
            "King's College Criteria for transplant listing",
            "Contact transplant center immediately",
            "Avoid sedatives (worsen HE), NSAIDs (worsen AKI)",
            "Hemodynamic instability: vasopressors (norepinephrine first-line)",
        ],
        "source": "AASLD ALF Guidelines 2021; EASL Clinical Practice Guidelines: ALF 2017",
    },
    "pbc": {
        "first_line": [
            "Ursodeoxycholic acid (UDCA) 13–15 mg/kg/day (split doses)",
            "Obeticholic acid 5 mg/day (titrate to 10 mg at 6 months if adequate response) for UDCA inadequate response",
        ],
        "second_line": [
            "Bezafibrate 400 mg/day or fenofibrate (in combination with UDCA)",
            "Seladelpar or elafibranor (investigational, recently approved in some regions)",
            "Cholestyramine for pruritus",
            "Rifampicin 150–300 mg BID for pruritus",
            "Sertraline for pruritus",
            "Naltrexone for refractory pruritus",
        ],
        "goals": [
            "Normalize ALP and bilirubin",
            "Prevent disease progression to cirrhosis",
            "Control symptoms (fatigue, pruritus)",
            "Liver transplantation for end-stage disease",
        ],
        "monitoring": [
            "LFTs every 3–6 months",
            "Lipid profile (UDCA may improve lipids)",
            "Bone density: DEXA scan every 2–3 years",
            "Fat-soluble vitamin levels annually",
            "AMA and IgM levels",
        ],
        "special": [
            "Overlap syndrome (AIH + PBC): requires immunosuppression + UDCA",
            "Fatigue is common — no specific treatment; lifestyle modification",
            "Calcium + Vitamin D supplementation routinely",
        ],
        "source": "EASL PBC Clinical Practice Guidelines 2017; AASLD PBC Guidance 2019",
    },
    "dili": {
        "first_line": [
            "Discontinue the offending drug (most important intervention)",
            "Acetaminophen toxicity: N-acetylcysteine (see ALF protocol)",
            "Supportive care: hydration, nutrition",
            "Cholestyramine for cholestatic DILI",
        ],
        "second_line": [
            "Corticosteroids for hypersensitivity DILI (anecdotal benefit)",
            "UDCA for cholestatic DILI",
            "Liver transplantation for fulminant DILI",
        ],
        "goals": [
            "Identify and discontinue the causative agent",
            "Allow liver recovery (usually 1–3 months)",
            "Prevent rechallenge with offending drug",
            "Hy's Law: ALT >3× ULN + bilirubin >2× ULN = high risk of mortality",
        ],
        "monitoring": [
            "LFTs weekly until resolution",
            "INR, bilirubin",
            "Monitor for progression to liver failure",
            "Document in medical record for future drug avoidance",
        ],
        "special": [
            "Report to pharmacovigilance authority",
            "Some immune-mediated DILI may require steroids",
            "Avoid re-exposure to causative drug",
        ],
        "source": "AASLD DILI Guidelines 2014; EASL DILI Guidelines 2019",
    },
}


def get_treatment_guidelines(diagnosis: str, patient_context: str = "") -> dict:
    """Retrieve evidence-based treatment guidelines for a liver disease diagnosis."""
    diagnosis_lower = diagnosis.lower()
    matched_key = None

    # Match diagnosis to database
    for key in TREATMENT_DB:
        if key in diagnosis_lower or any(k in diagnosis_lower for k in key.split()):
            matched_key = key
            break

    if not matched_key:
        # Try broader matching
        if "hbv" in diagnosis_lower or "hepatitis b" in diagnosis_lower:
            matched_key = "hepatitis b"
        elif "hcv" in diagnosis_lower or "hepatitis c" in diagnosis_lower:
            matched_key = "hepatitis c"
        elif "nash" in diagnosis_lower or "nafld" in diagnosis_lower or "fatty" in diagnosis_lower:
            matched_key = "nafld"
        elif "alcohol" in diagnosis_lower or "ald" in diagnosis_lower:
            matched_key = "alcoholic liver disease"
        elif "autoimmune" in diagnosis_lower or "aih" in diagnosis_lower:
            matched_key = "autoimmune hepatitis"
        elif "cirrhosis" in diagnosis_lower or "cirrhotic" in diagnosis_lower:
            matched_key = "liver cirrhosis"
        elif "hcc" in diagnosis_lower or "hepatocellular" in diagnosis_lower:
            matched_key = "hcc"
        elif "acute liver failure" in diagnosis_lower or "alf" in diagnosis_lower:
            matched_key = "acute liver failure"
        elif "pbc" in diagnosis_lower or "primary biliary" in diagnosis_lower:
            matched_key = "pbc"
        elif "dili" in diagnosis_lower or "drug-induced" in diagnosis_lower:
            matched_key = "dili"

    if not matched_key:
        return TreatmentGuideline(
            diagnosis=diagnosis,
            first_line_treatment=["No specific guidelines found for this diagnosis"],
            second_line_treatment=["Please consult a hepatologist"],
            treatment_goals=["Evaluate with specialist"],
            monitoring_parameters=["As clinically indicated"],
            special_considerations=["Consider referral to hepatology center"],
            guideline_source="N/A",
        ).model_dump()

    db = TREATMENT_DB[matched_key]
    return TreatmentGuideline(
        diagnosis=diagnosis,
        first_line_treatment=db.get("first_line", []),
        second_line_treatment=db.get("second_line", []),
        treatment_goals=db.get("goals", []),
        monitoring_parameters=db.get("monitoring", []),
        special_considerations=db.get("special", []),
        guideline_source=db.get("source", ""),
    ).model_dump()


# ─── Tool 6: Generate Clinical Summary ──────────────────────────────────────

def generate_clinical_summary(
    chief_complaint: str = "",
    clinical_data: str = "",
    assessment: str = "",
    plan: str = "",
    mode: str = "physician",
) -> dict:
    """Generate a structured clinical summary."""
    all_text = f"{chief_complaint} {clinical_data} {assessment} {plan}".lower()

    key_findings = []
    urgent_actions = []
    follow_up = []
    patient_education = []

    # Extract key findings from text
    if "cirrhosis" in all_text:
        key_findings.append("Evidence of liver cirrhosis")
        follow_up.append("HCC surveillance: US ± AFP every 6 months")
        follow_up.append("Upper endoscopy for variceal screening")
    if "hbsag" in all_text and "positive" in all_text:
        key_findings.append("Chronic HBV infection confirmed")
        follow_up.append("Monitor HBV DNA and ALT every 3–6 months")
    if "hcv" in all_text and ("positive" in all_text or "detectable" in all_text):
        key_findings.append("HCV infection — treatment indicated")
        follow_up.append("Initiate DAA therapy after genotyping")
    if "afp" in all_text:
        for m in re.finditer(r'AFP[\s:=]*([\d.]+)', clinical_data, re.IGNORECASE):
            afp_val = float(m.group(1))
            if afp_val > 400:
                urgent_actions.append(f"AFP critically elevated ({afp_val} ng/mL) — urgent multiphasic CT/MRI for HCC")
                key_findings.append(f"AFP critically elevated: {afp_val} ng/mL")
    if "acute liver failure" in all_text or "alf" in all_text:
        urgent_actions.append("URGENT: ICU admission and immediate transplant center contact")
        key_findings.append("Acute liver failure — life-threatening condition")

    if mode == "patient":
        patient_education = [
            "Your liver is being carefully evaluated and treated",
            "Take all medications as prescribed — do not stop without consulting your doctor",
            "Avoid alcohol completely",
            "Report any new symptoms immediately: increased yellowing, confusion, leg swelling, black stools",
            "Maintain a healthy diet with adequate protein and calories",
            "Attend all follow-up appointments",
        ]

    return ClinicalSummary(
        chief_complaint=chief_complaint,
        clinical_presentation=clinical_data[:500] if clinical_data else "",
        key_findings=key_findings or ["See clinical data above"],
        assessment=assessment or "Assessment pending review of clinical data",
        differential_diagnoses=[],
        severity_assessment="",
        plan=plan.split("\n") if plan else ["See treatment guidelines"],
        follow_up=follow_up or ["As clinically indicated"],
        urgent_actions=urgent_actions,
        patient_education=patient_education,
        mode=mode,
    ).model_dump()


# ─── Tool 7: Flag Urgent Findings ───────────────────────────────────────────

URGENT_CRITERIA = {
    "Acute Liver Failure": {
        "keywords": ["acute liver failure", "alf", "encephalopathy with coagulopathy"],
        "lab_trigger": {"inr": (">1.5", "INR >1.5"), "bilirubin": (">3.0", "bilirubin >3 mg/dL")},
        "level": "CRITICAL",
        "action": "IMMEDIATE ICU admission + transplant center contact",
        "timeframe": "Within 1 hour",
    },
    "Variceal Hemorrhage": {
        "keywords": ["hematemesis", "melena", "black stool", "variceal bleeding", "upper gi bleed"],
        "level": "CRITICAL",
        "action": "Immediate resuscitation + urgent endoscopy + vasoactive drugs + antibiotics",
        "timeframe": "Within 2 hours",
    },
    "Spontaneous Bacterial Peritonitis": {
        "keywords": ["sbp", "fever with ascites", "abdominal pain with ascites", "infected ascites"],
        "level": "URGENT",
        "action": "Paracentesis for diagnostic tap + empirical cefotaxime 2g IV q8h + albumin",
        "timeframe": "Within 4 hours",
    },
    "Hepatorenal Syndrome": {
        "keywords": ["hrs", "hepatorenal", "renal failure in cirrhosis"],
        "lab_trigger": {"creatinine": (">1.5", "creatinine >1.5 mg/dL")},
        "level": "URGENT",
        "action": "Discontinue diuretics + albumin challenge + consider terlipressin/norepinephrine",
        "timeframe": "Within 12 hours",
    },
    "Critical AFP": {
        "lab_trigger": {"afp": (">400", "AFP >400 ng/mL")},
        "keywords": ["afp elevated"],
        "level": "URGENT",
        "action": "Multiphasic CT/MRI within 48 hours; HCC multidisciplinary team referral",
        "timeframe": "Within 48 hours",
    },
    "Severe Coagulopathy": {
        "lab_trigger": {"inr": (">2.5", "INR >2.5")},
        "level": "URGENT",
        "action": "Assess for acute liver failure; Vitamin K; monitor closely",
        "timeframe": "Within 12 hours",
    },
    "Hepatic Encephalopathy": {
        "keywords": ["confusion", "encephalopathy", "hepatic coma", "asterixis", "flapping tremor"],
        "level": "URGENT",
        "action": "Identify and treat precipitating cause + lactulose + consider rifaximin",
        "timeframe": "Within 4 hours",
    },
    "Critical Bilirubin": {
        "lab_trigger": {"bilirubin": (">15", "Total bilirubin >15 mg/dL")},
        "level": "URGENT",
        "action": "Evaluate for acute liver failure, biliary obstruction, or massive hemolysis",
        "timeframe": "Within 24 hours",
    },
    "Critical Thrombocytopenia": {
        "lab_trigger": {"platelets": ("<50", "Platelets <50 ×10³/μL")},
        "level": "URGENT",
        "action": "Risk of bleeding — transfuse if active bleeding; review medications",
        "timeframe": "Within 24 hours",
    },
    "Critical Sodium Hyponatremia": {
        "lab_trigger": {"sodium": ("<125", "Sodium <125 mEq/L")},
        "level": "URGENT",
        "action": "Fluid restriction; consider tolvaptan; evaluate for hyponatremic encephalopathy",
        "timeframe": "Within 24 hours",
    },
}


def flag_urgent_findings(
    clinical_text: str = "",
    lab_data: str = "",
    imaging: str = "",
    labs_dict: dict | None = None,
) -> dict:
    """Identify urgent or critical findings requiring immediate attention."""
    all_text = f"{clinical_text} {lab_data} {imaging}".lower()
    findings: list[UrgentFinding] = []
    has_critical = False
    has_urgent = False
    emergency_actions = []

    # Parse numeric labs from text for threshold checking
    lab_values = {}
    for m in re.finditer(r'\bINR\b[\s:=]*([\d.]+)', lab_data, re.IGNORECASE):
        lab_values["inr"] = float(m.group(1))
    for m in re.finditer(r'(?:total\s+)?bilirubin[\s:=]*([\d.]+)', lab_data, re.IGNORECASE):
        lab_values["bilirubin"] = float(m.group(1))
    for m in re.finditer(r'\b(?:creatinine|Cr)\b[\s:=]*([\d.]+)', lab_data, re.IGNORECASE):
        lab_values["creatinine"] = float(m.group(1))
    for m in re.finditer(r'\bAFP\b[\s:=]*([\d.]+)', lab_data, re.IGNORECASE):
        lab_values["afp"] = float(m.group(1))
    for m in re.finditer(r'\b(?:platelets?|PLT)\b[\s:=]*([\d.]+)', lab_data, re.IGNORECASE):
        lab_values["platelets"] = float(m.group(1))
    for m in re.finditer(r'\b(?:sodium|Na)\b[\s:=]*([\d.]+)', lab_data, re.IGNORECASE):
        lab_values["sodium"] = float(m.group(1))
    for m in re.finditer(r'\bALT\b[\s:=]*([\d.]+)', lab_data, re.IGNORECASE):
        lab_values["alt"] = float(m.group(1))

    if labs_dict:
        lab_values.update(labs_dict)

    for finding_name, criteria in URGENT_CRITERIA.items():
        triggered = False
        rationale_parts = []

        # Keyword check
        for kw in criteria.get("keywords", []):
            if kw in all_text:
                triggered = True
                rationale_parts.append(f"Keyword detected: '{kw}'")

        # Lab threshold check
        for lab, (threshold, desc) in criteria.get("lab_trigger", {}).items():
            val = lab_values.get(lab)
            if val is not None:
                op = threshold[0]
                threshold_val = float(threshold[1:])
                if op == ">" and val > threshold_val:
                    triggered = True
                    rationale_parts.append(f"{desc} (actual: {val})")
                elif op == "<" and val < threshold_val:
                    triggered = True
                    rationale_parts.append(f"{desc} (actual: {val})")

        if triggered:
            level = criteria["level"]
            findings.append(UrgentFinding(
                finding=finding_name,
                urgency_level=level,
                recommended_action=criteria["action"],
                timeframe=criteria["timeframe"],
                rationale="; ".join(rationale_parts) if rationale_parts else "Clinical presentation",
            ))
            if level == "CRITICAL":
                has_critical = True
                emergency_actions.append(f"🚨 {finding_name}: {criteria['action']}")
            else:
                has_urgent = True

    overall = ""
    if has_critical:
        overall = "⚠️  CRITICAL findings detected. Immediate emergency action required."
    elif has_urgent:
        overall = "⚠️  Urgent findings detected. Prompt evaluation required within hours."
    elif not findings:
        overall = "No critical or urgent findings identified at this time. Continue routine monitoring."

    return UrgentFindings(
        has_critical=has_critical,
        has_urgent=has_urgent,
        findings=findings,
        overall_recommendation=overall,
        emergency_actions=emergency_actions,
    ).model_dump()


# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "parse_lab_values",
        "description": (
            "Extract and normalize laboratory values from free text. "
            "Identifies ALT, AST, ALP, GGT, bilirubin, albumin, INR, creatinine, AFP, CBC, "
            "viral markers (HBsAg, HCV Ab, HBV DNA), and other liver-related labs. "
            "Flags abnormal and critical values with reference ranges. "
            "Call this when the user provides lab results or a clinical note with lab values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Free text containing laboratory values (e.g., 'ALT: 85 U/L, AST: 72 U/L, bilirubin: 2.1 mg/dL')",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "calculate_severity_scores",
        "description": (
            "Calculate liver disease severity scores: Child-Pugh (class A/B/C), "
            "MELD score (3-month mortality), and ALBI score. "
            "Use when a patient has known cirrhosis and you have liver function values. "
            "All parameters are optional — will calculate whichever scores are possible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bilirubin": {"type": "number", "description": "Total bilirubin in mg/dL"},
                "albumin": {"type": "number", "description": "Albumin in g/dL"},
                "inr": {"type": "number", "description": "INR (International Normalized Ratio)"},
                "creatinine": {"type": "number", "description": "Creatinine in mg/dL"},
                "ascites": {
                    "type": "integer",
                    "description": "Ascites: 0=none, 1=mild/controlled, 2=moderate-severe/refractory",
                    "enum": [0, 1, 2],
                },
                "encephalopathy": {
                    "type": "integer",
                    "description": "Hepatic encephalopathy grade: 0=none, 1=grade 1, 2=grade 2, 3=grade 3, 4=grade 4",
                    "enum": [0, 1, 2, 3, 4],
                },
            },
        },
    },
    {
        "name": "differential_diagnosis",
        "description": (
            "Generate a ranked differential diagnosis list for liver disease based on clinical presentation. "
            "Covers HBV, HCV, NAFLD/NASH, alcoholic liver disease, autoimmune hepatitis, PBC, PSC, "
            "cirrhosis, HCC, acute liver failure, DILI, Wilson's disease, and hemochromatosis. "
            "Call this when you have a clinical presentation and want to systematically evaluate possible diagnoses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symptoms": {"type": "string", "description": "Patient symptoms and presenting complaints"},
                "lab_findings": {"type": "string", "description": "Summary of abnormal lab results"},
                "imaging_findings": {"type": "string", "description": "Imaging report findings (CT/MRI/US)"},
                "patient_history": {"type": "string", "description": "Relevant medical and social history"},
            },
        },
    },
    {
        "name": "assess_fibrosis_stage",
        "description": (
            "Estimate liver fibrosis/cirrhosis stage (Metavir F0–F4) from available data. "
            "Uses pathology results (gold standard), Fibroscan/LSE values, imaging findings, "
            "or FIB-4 calculation (requires age, platelets, ALT, AST). "
            "Call when evaluating disease severity or treatment urgency."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lab_data": {"type": "string", "description": "Lab values text (for FIB-4 calculation)"},
                "imaging_findings": {"type": "string", "description": "Imaging/Fibroscan findings"},
                "pathology": {"type": "string", "description": "Liver biopsy pathology report"},
                "platelets": {"type": "number", "description": "Platelet count ×10³/μL"},
                "alt": {"type": "number", "description": "ALT in U/L"},
                "age": {"type": "integer", "description": "Patient age in years"},
            },
        },
    },
    {
        "name": "get_treatment_guidelines",
        "description": (
            "Retrieve evidence-based treatment guidelines for a specific liver disease diagnosis. "
            "Covers HBV, HCV, NAFLD/NASH, alcoholic liver disease, autoimmune hepatitis, "
            "PBC, liver cirrhosis, HCC, acute liver failure, and DILI. "
            "Based on AASLD and EASL guidelines. "
            "Call after establishing a diagnosis to get treatment recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "The diagnosis (e.g., 'Hepatitis B', 'NAFLD', 'Liver Cirrhosis', 'HCC')",
                },
                "patient_context": {
                    "type": "string",
                    "description": "Relevant patient context (e.g., 'cirrhotic, Child-Pugh B', 'pregnant', 'HIV coinfection')",
                },
            },
            "required": ["diagnosis"],
        },
    },
    {
        "name": "generate_clinical_summary",
        "description": (
            "Generate a structured clinical summary with assessment and management plan. "
            "Produces a physician-style SOAP note or patient-friendly explanation. "
            "Call at the end of a consultation to synthesize all findings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chief_complaint": {"type": "string", "description": "Patient's chief complaint"},
                "clinical_data": {"type": "string", "description": "All clinical data (labs, imaging, history)"},
                "assessment": {"type": "string", "description": "Clinical assessment/diagnosis"},
                "plan": {"type": "string", "description": "Management plan"},
                "mode": {
                    "type": "string",
                    "description": "Output mode: 'physician' for clinical language or 'patient' for simplified",
                    "enum": ["physician", "patient"],
                },
            },
        },
    },
    {
        "name": "flag_urgent_findings",
        "description": (
            "Identify critical or urgent findings that require immediate medical attention. "
            "Detects acute liver failure, variceal hemorrhage, SBP, HRS, critical lab values, "
            "and other hepatological emergencies. "
            "ALWAYS call this when evaluating a new patient or when new concerning data is presented."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "clinical_text": {"type": "string", "description": "Clinical symptoms and presentation"},
                "lab_data": {"type": "string", "description": "Laboratory values text"},
                "imaging": {"type": "string", "description": "Imaging findings"},
            },
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict) -> Any:
    """Dispatch tool calls to their implementations."""
    tool_map = {
        "parse_lab_values": lambda i: parse_lab_values(i["text"]),
        "calculate_severity_scores": lambda i: calculate_severity_scores(**i),
        "differential_diagnosis": lambda i: differential_diagnosis(**i),
        "assess_fibrosis_stage": lambda i: assess_fibrosis_stage(**i),
        "get_treatment_guidelines": lambda i: get_treatment_guidelines(**i),
        "generate_clinical_summary": lambda i: generate_clinical_summary(**i),
        "flag_urgent_findings": lambda i: flag_urgent_findings(**i),
    }
    fn = tool_map.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn(tool_input)
    except Exception as e:
        return {"error": str(e)}
