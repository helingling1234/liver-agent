"""Tool implementations for liver disease diagnosis and treatment agent."""

from __future__ import annotations
import math
import re
import json
from typing import Any

# medcalc: validated clinical calculators (pip install medcalc)
try:
    import medcalc.calculator as _mc
    _MEDCALC = True
except ImportError:
    _MEDCALC = False

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
    """Compute Child-Pugh score. Uses medcalc validated library when available."""
    # Use medcalc validated formula if installed
    if _MEDCALC:
        ascites_map = {0: "none", 1: "mild", 2: "moderate"}
        ascites_str = ascites_map.get(ascites, "none")
        total = _mc.child_pugh_score(
            bilirubin=bilirubin, albumin=albumin, inr=inr,
            ascites=ascites_str, encephalopathy_grade=min(encephalopathy, 4)
        )
        # Reconstruct individual points (medcalc returns only total)
        bili_pts   = 1 if bilirubin < 2 else (2 if bilirubin <= 3 else 3)
        alb_pts    = 1 if albumin > 3.5 else (2 if albumin >= 2.8 else 3)
        inr_pts    = 1 if inr < 1.7 else (2 if inr <= 2.3 else 3)
        ascites_pts = min(ascites + 1, 3) if ascites >= 0 else 1
        enc_pts    = 1 if encephalopathy == 0 else (2 if encephalopathy <= 2 else 3)
    else:
        # Fallback: hand-coded formula
        bili_pts   = 1 if bilirubin < 2 else (2 if bilirubin <= 3 else 3)
        alb_pts    = 1 if albumin > 3.5 else (2 if albumin >= 2.8 else 3)
        inr_pts    = 1 if inr < 1.7 else (2 if inr <= 2.3 else 3)
        ascites_pts = min(ascites + 1, 3) if ascites >= 0 else 1
        enc_pts    = 1 if encephalopathy == 0 else (2 if encephalopathy <= 2 else 3)
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


def _meld(bilirubin: float, inr: float, creatinine: float,
          sodium: float = 137, albumin: float = 3.5,
          sex: str = "M", dialysis: bool = False) -> MeldScore:
    """MELD 3.0 score (2022 UNOS standard). Uses medcalc validated library when available."""
    if _MEDCALC:
        score = _mc.meld_3(
            age=50,  # age not used in MELD 3.0 formula but required by medcalc
            female=(sex.upper() in ("F", "FEMALE")),
            bilirubin=bilirubin,
            inr=inr,
            creatinine=creatinine,
            albumin=albumin,
            sodium=max(min(sodium, 137), 125),
            dialysis=dialysis,
        )
        score = max(score, 6)
    else:
        # Fallback: classic MELD formula
        bili = max(bilirubin, 1.0)
        inr_v = max(inr, 1.0)
        cr = max(min(creatinine, 4.0), 1.0)
        score = round(3.78 * math.log(bili) + 11.2 * math.log(inr_v) + 9.57 * math.log(cr) + 6.43)
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

    # ── NEW DISEASES ──────────────────────────────────────────────────────────

    "hepatitis d": {
        "first_line": [
            "Pegylated interferon alpha-2a (Peg-IFN-α) 180 mcg SC weekly × 48 weeks",
            "Bulevirtide 2 mg SC daily (EU approved 2020; entry inhibitor) — long-term/indefinite",
            "Bulevirtide 2 mg SC daily + Peg-IFN-α 180 mcg SC weekly × 48 weeks (combination — best SVR)",
            "All patients with HDV must also receive HBV treatment: TDF 300 mg/day or ETV 0.5 mg/day",
        ],
        "second_line": [
            "Lonafarnib (farnesyl transferase inhibitor) + ritonavir — investigational",
            "Peg-IFN-lambda — investigational trials",
        ],
        "goals": [
            "Undetectable HDV RNA at 6 months post-treatment (sustained virological response)",
            "ALT normalization",
            "HBsAg loss (rare but possible)",
            "Prevent cirrhosis progression and decompensation",
        ],
        "monitoring": [
            "HDV RNA quantification at baseline, weeks 12, 24, 48, end of treatment, and 24 weeks post-treatment",
            "ALT, AST, bilirubin every 3 months",
            "Serum bile acids (bulevirtide causes physiological bile acid increase — monitor for symptoms)",
            "HBV DNA and HBsAg every 6 months",
            "HCC surveillance if cirrhotic: US ± AFP every 6 months",
        ],
        "special": [
            "Screen ALL HBsAg-positive patients for HDV (EASL 2023 recommendation)",
            "Anti-HDV antibodies → confirm with HDV RNA if positive",
            "Bulevirtide available in EU/UK; limited access elsewhere — compassionate use programs",
            "HDV coinfection markedly accelerates cirrhosis — treat urgently",
            "Avoid liver biopsy in coagulopathic patients",
        ],
        "source": "EASL HDV CPG 2023; AASLD guidance",
    },

    "hepatitis e": {
        "first_line": [
            "Immunocompetent patients: Supportive care — most self-limiting (4–6 weeks)",
            "Immunocompromised patients: Reduce immunosuppression (if transplant/immunosuppressed) first",
            "If no spontaneous clearance after immunosuppression reduction: Ribavirin 600–1000 mg/day PO × 3–6 months",
            "Pregnancy (especially 3rd trimester): Supportive care; ICU monitoring; anti-emetics; IV fluids",
        ],
        "second_line": [
            "Ribavirin dose adjustment for renal impairment (eGFR-based dosing)",
            "Pegylated interferon (off-label; limited data in immunocompromised)",
            "Liver transplantation if acute liver failure develops",
        ],
        "goals": [
            "Immunocompetent: Spontaneous recovery in 4–6 weeks",
            "Immunocompromised: HEV RNA negativity at 3 months on ribavirin (SVR)",
            "Prevent progression to acute liver failure",
            "Prevent chronic HEV (immunocompromised at risk)",
        ],
        "monitoring": [
            "HEV RNA at 3 and 6 months on ribavirin therapy",
            "CBC every 4 weeks (ribavirin causes hemolytic anemia — hold if Hgb drops <10 g/dL)",
            "Renal function (ribavirin dose adjust if eGFR <50)",
            "LFTs weekly until resolution of hepatitis",
            "Fetal surveillance (non-stress test, BPP) if pregnant",
        ],
        "special": [
            "Genotype 3/4: Zoonotic (pork, deer); chronic HEV only in immunocompromised",
            "Genotype 1/2: Waterborne; epidemic in developing countries; severe in pregnancy",
            "HEV in pregnancy: 20–25% mortality (GT1); deliver if unstable",
            "Recurrence after ribavirin: Longer course (12 months) or pegIFN",
            "Report to public health authorities",
        ],
        "source": "EASL HEV CPG 2018",
    },

    "hepatitis a": {
        "first_line": [
            "Supportive care: IV hydration, antiemetics, analgesics (avoid acetaminophen/NSAIDs)",
            "Rest; adequate nutrition",
            "Hospitalize if: INR >1.5, bilirubin >15 mg/dL, encephalopathy, dehydration, immunocompromised",
            "No antiviral therapy available or indicated",
        ],
        "second_line": [
            "Liver transplantation if acute liver failure develops (King's College Criteria)",
        ],
        "goals": [
            "Spontaneous resolution (nearly universal in immunocompetent)",
            "Prevention: HAV vaccination (2-dose series: HAVRIX or VAQTA)",
            "Post-exposure prophylaxis: HAV vaccine ± HAV immunoglobulin within 2 weeks",
        ],
        "monitoring": [
            "LFTs weekly until downtrend confirmed, then monthly until normalization",
            "INR if jaundice present (synthetic function marker)",
            "Return-to-work when anicteric and ALT <3× ULN (typically 4–6 weeks)",
        ],
        "special": [
            "Relapsing HAV: 10–15% of patients; manage supportively; resolves within 3 months",
            "Cholestatic HAV: Prolonged jaundice; consider UDCA 13 mg/kg/day",
            "Avoid alcohol for 6 months",
            "Vaccinate household contacts and sexual partners",
        ],
        "source": "AASLD/ACG/EASL general guidance; WHO vaccination recommendations",
    },

    "wilson": {
        "first_line": [
            "D-Penicillamine 750–1500 mg/day PO in 2–4 divided doses (start 250 mg/day, increase weekly)",
            "Trientine dihydrochloride 750–1500 mg/day PO in 2–3 divided doses (preferred for neurological disease — less neurological worsening risk than D-penicillamine)",
            "Zinc acetate 50 mg elemental zinc TID (25 mg TID in children) — presymptomatic or maintenance",
            "Always supplement pyridoxine 25 mg/day with D-penicillamine",
        ],
        "second_line": [
            "Trientine for D-penicillamine intolerance/toxicity (rash, proteinuria, lupus-like syndrome)",
            "Zinc salts as maintenance after initial chelation (D-penicillamine/trientine × 2 years then switch)",
            "Tetrathiomolybdate — investigational (neurological Wilson's with rapid copper removal)",
            "Liver transplantation: Acute liver failure (Wilson's + Coombs-negative hemolytic anemia); decompensated cirrhosis failing chelation",
        ],
        "goals": [
            "24-hr urinary copper 200–500 mcg/24h on chelation therapy",
            "24-hr urinary copper <75 mcg/24h on zinc maintenance",
            "Exchangeable/non-ceruloplasmin-bound copper (NCC) <0.8 mcmol/L",
            "Neurological symptom stabilization/improvement",
            "Prevent further copper accumulation",
        ],
        "monitoring": [
            "24-hr urine copper every 3–6 months (primary monitoring parameter)",
            "Serum ceruloplasmin, non-ceruloplasmin-bound copper every 6–12 months",
            "CBC, urinalysis, renal function every 3 months (D-penicillamine toxicity monitoring)",
            "Neurological assessment annually",
            "Ophthalmology (Kayser-Fleischer ring regression): every 1–2 years",
            "LFTs every 3–6 months",
        ],
        "special": [
            "Leipzig Score ≥4 = Wilson's diagnosis confirmed; perform in all suspected cases",
            "Pregnancy: Zinc salts preferred (safest); continue chelation at 25–50% reduced dose if needed; penicillamine teratogenic at high doses",
            "Screen first-degree relatives: serum ceruloplasmin, urine copper, ATP7B genotyping",
            "Never stop treatment — withdrawal can cause acute fulminant liver failure",
            "Liver transplantation corrects the genetic defect (cures Wilson's)",
        ],
        "source": "EASL-ERN Wilson's Disease CPG 2023; AASLD 2008/2011 updated guidance",
    },

    "hemochromatosis": {
        "first_line": [
            "Therapeutic phlebotomy (venesection): 450–500 mL whole blood (removes ~250 mg iron) weekly or every 2 weeks",
            "Induction phase: Continue until ferritin 50–100 mcg/L + transferrin saturation <50%",
            "Check Hgb/Hct before each phlebotomy — hold if Hgb <11 g/dL",
            "Maintenance phase (after iron depletion): 2–4 phlebotomies/year to keep ferritin 50–100 mcg/L",
        ],
        "second_line": [
            "Erythrocytapheresis: More efficient iron removal (2–3× blood volume per session); some centers preferred",
            "Iron chelation (desferrioxamine SC/IV or deferasirox PO): Reserved for patients who cannot tolerate phlebotomy (severe anemia, heart failure, poor venous access)",
            "Deferasirox 10–40 mg/kg/day PO (rarely used in hemochromatosis; better established for transfusional iron overload)",
        ],
        "goals": [
            "Ferritin 50–100 mcg/L (not <20 mcg/L — risk of iron deficiency)",
            "Transferrin saturation <50%",
            "Prevent end-organ damage: cirrhosis, diabetes, cardiomyopathy, arthropathy",
            "Regression of fibrosis possible if caught before cirrhosis",
        ],
        "monitoring": [
            "Ferritin + transferrin saturation: every 10 phlebotomies during induction, then every 3–6 months maintenance",
            "CBC before each phlebotomy",
            "LFTs, glucose, HbA1c every 6–12 months",
            "HCC surveillance if cirrhotic: US + AFP every 6 months",
            "Cardiac function (echo/ECG) if cardiac iron overload suspected",
            "Bone density (DEXA) — hypogonadism common",
        ],
        "special": [
            "HFE genotyping: C282Y homozygous = classical hemochromatosis (most common Caucasian); H63D/C282Y compound heterozygous = mild",
            "Screen first-degree relatives: HFE genotype + ferritin + transferrin saturation",
            "Avoid vitamin C supplements >500 mg/day (enhances iron absorption)",
            "Avoid raw shellfish (Vibrio vulnificus lethal with iron overload)",
            "Liver biopsy if ferritin >1000 mcg/L or elevated ALT or hepatomegaly",
        ],
        "source": "EASL Hemochromatosis CPG 2022; AASLD 2011",
    },

    "alpha1 antitrypsin": {
        "first_line": [
            "No proven disease-modifying therapy for liver disease in AATD",
            "Avoid hepatotoxic agents: limit alcohol, avoid excess acetaminophen, avoid hepatotoxic herbal supplements",
            "Manage cirrhosis complications per standard guidelines (ascites, varices, HE)",
            "IV A1AT augmentation therapy (60 mg/kg/week): Addresses lung disease ONLY — no liver benefit",
        ],
        "second_line": [
            "Liver transplantation: Definitive treatment for end-stage liver disease (corrects A1AT genotype in recipient)",
            "Clinical trials: Gene therapy (FAZILi), RNA interference therapies (siRNA) — in development",
        ],
        "goals": [
            "Prevent progression to end-stage liver disease",
            "Optimize co-morbidities (avoid additional liver insults)",
            "Lung protection: A1AT augmentation + smoking cessation",
            "Transplant evaluation when MELD ≥15 or decompensation",
        ],
        "monitoring": [
            "LFTs, CBC, INR, albumin every 6 months",
            "Liver ultrasound every 6–12 months (cirrhotic patients)",
            "HCC surveillance: US + AFP every 6 months if cirrhotic",
            "Pulmonary function tests annually (FEV1, DLCO)",
            "Screen children with PIZZ phenotype for liver disease at birth and annually",
        ],
        "special": [
            "PIZZ = most severe genotype (serum A1AT <57 mg/dL); liver and lung disease",
            "Diagnosis: Serum A1AT level <80 mg/dL → confirm with phenotyping/genotyping",
            "Screen first-degree relatives of PIZZ patients",
            "Neonatal cholestasis most common liver presentation in infancy",
            "Adults: may present with cryptogenic cirrhosis — always check A1AT level",
        ],
        "source": "AASLD/EASL guidance; Alpha-1 Foundation guidelines 2020",
    },

    "psc": {
        "first_line": [
            "No pharmacotherapy proven to alter PSC disease course or improve transplant-free survival",
            "UDCA: EASL does NOT recommend routinely (conflicting evidence; high-dose ≥28 mg/kg/day associated with harm in CRC trial). Some centers use 13–23 mg/kg/day off-label for biochemical improvement",
            "Dominant biliary strictures: ERCP with balloon dilation (preferred over stenting)",
            "Pre-ERCP antibiotics: Ciprofloxacin 500 mg BID × 3–5 days to prevent cholangitis",
            "Management of pruritus: Cholestyramine 4–16 g/day → Rifampicin 150–300 mg BID → Naltrexone 12.5–50 mg/day → Sertraline 75–100 mg/day",
        ],
        "second_line": [
            "Antibiotics for recurrent cholangitis: Ciprofloxacin 500 mg BID × 7–14 days (acute); long-term prophylaxis ciprofloxacin 500 mg/day if ≥2 episodes/year",
            "Fat-soluble vitamin supplementation (A, D, E, K) if malabsorption",
            "Liver transplantation: End-stage PSC, recurrent bacterial cholangitis, intractable pruritus, suspected CCA without confirmed diagnosis",
            "Bezafibrate 400 mg/day: Some data for ALP reduction (off-label)",
        ],
        "goals": [
            "Prevent and manage biliary strictures and cholangitis",
            "Early detection of cholangiocarcinoma (CCA) — 10–15% lifetime risk",
            "Monitor for colorectal cancer (IBD-associated PSC: 5× higher CRC risk)",
            "Liver transplantation for end-stage disease (5-year survival ~80%)",
        ],
        "monitoring": [
            "LFTs (ALP, GGT, bilirubin, albumin, INR) every 6–12 months",
            "CA 19-9 annually (CCA surveillance; limited specificity)",
            "MRCP every 1–2 years (dominant stricture surveillance)",
            "Colonoscopy every 1–2 years (PSC-IBD patients — high colorectal cancer risk)",
            "HCC surveillance if cirrhotic: US + AFP every 6 months",
            "IgG4 at baseline (rule out IgG4-sclerosing cholangitis — responds to steroids)",
        ],
        "special": [
            "Rule out IgG4-SC (serum IgG4 >2× ULN) — steroid-responsive, unlike PSC",
            "PSC-IBD: predominantly UC; pancolitis pattern; regular surveillance colonoscopy",
            "Small-duct PSC: Normal MRCP but liver biopsy changes; better prognosis",
            "PSC recurs in 20–25% of liver grafts; re-transplantation may be required",
            "Liver transplantation listing: MELD exception points for recurrent cholangitis",
        ],
        "source": "EASL Sclerosing Cholangitis CPG 2022; ESGE/EASL 2017; AASLD guidelines",
    },

    "overlap syndrome": {
        "first_line": [
            "AIH/PBC Overlap (Paris Criteria): UDCA 13–15 mg/kg/day + immunosuppression (prednisolone 30–40 mg/day tapering + azathioprine 1–2 mg/kg/day)",
            "AIH/PSC Overlap: Immunosuppression (prednisolone + azathioprine) as per AIH protocol ± UDCA (controversial for PSC component)",
            "Dominant condition guides primary therapy; dual therapy for overlap",
        ],
        "second_line": [
            "Refractory AIH component: MMF 1–1.5 g BID, tacrolimus 2–3 mg BID",
            "Inadequate UDCA response (PBC component): Add obeticholic acid 5–10 mg/day or bezafibrate 400 mg/day",
            "Liver transplantation for end-stage disease",
        ],
        "goals": [
            "Biochemical remission of both components (normal ALT, IgG, ALP, bilirubin)",
            "Histological remission of AIH component",
            "Prevention of disease progression",
        ],
        "monitoring": [
            "ALT, AST, ALP, GGT, bilirubin, IgG every 3–6 months",
            "CBC (azathioprine monitoring)",
            "Bone density annually (chronic steroid use)",
            "MRCP if PSC overlap suspected (biliary abnormalities)",
        ],
        "special": [
            "Paris Criteria for AIH/PBC: ≥2 AIH criteria + ≥2 PBC criteria",
            "AIH/PSC overlap more common in children and young adults",
            "Autoimmune hepatitis scoring (IAIHG simplified) useful to quantify AIH component",
            "Respond to immunosuppression better than pure PSC",
        ],
        "source": "EASL AIH CPG 2015; EASL PSC CPG 2022; AASLD guidance",
    },

    "budd chiari": {
        "first_line": [
            "Anticoagulation IMMEDIATELY (all patients unless contraindicated):",
            "LMWH: Enoxaparin 1 mg/kg SC BID → bridge to VKA (warfarin, target INR 2–3) or DOAC",
            "DOACs (rivaroxaban, apixaban, dabigatran): Increasingly preferred over VKA; avoid in antiphospholipid syndrome (triple-positive) — use VKA",
            "Treat underlying prothrombotic condition: JAK2 V617F → myeloproliferative neoplasm treatment",
            "Angioplasty ± stenting: For focal IVC/hepatic vein stenosis or short-segment occlusion",
        ],
        "second_line": [
            "TIPS (transjugular intrahepatic portosystemic shunt): Anticoagulation + angioplasty failure; PTFE-covered stent preferred",
            "Liver transplantation: Failed TIPS, acute liver failure, cirrhosis not amenable to TIPS",
            "Systemic thrombolysis (urokinase, tPA): Acute BCS <3 weeks; specialist-only",
        ],
        "goals": [
            "Establish hepatic venous drainage (recanalization or shunting)",
            "Prevent progression to cirrhosis",
            "Treat underlying thrombophilia",
            "Indefinite anticoagulation if thrombophilia present",
        ],
        "monitoring": [
            "LFTs, INR (if on VKA), CBC every 1–3 months",
            "Doppler ultrasound every 3–6 months (TIPS patency; hepatic vein flow)",
            "Thrombophilia workup at diagnosis: JAK2, Factor V Leiden, prothrombin G20210A, protein C/S/antithrombin, antiphospholipid panel",
            "Bone marrow biopsy if JAK2-positive",
            "HCC surveillance if cirrhotic",
        ],
        "special": [
            "Most common thrombophilias: MPN (40%), antiphospholipid syndrome, PNH, Factor V Leiden",
            "TIPS occlusion: Revise/redilate; 1-year patency >80% with PTFE stents",
            "Acute BCS: Tender hepatomegaly + ascites + abdominal pain — diagnosis by Doppler",
            "Subacute/chronic BCS: Insidious onset; caudate lobe hypertrophy (direct hepatic vein drainage)",
        ],
        "source": "EASL Vascular Liver Diseases CPG 2016 (update 2025); APASL 2023",
    },

    "portal vein thrombosis": {
        "first_line": [
            "Acute non-cirrhotic PVT: Anticoagulation immediately — LMWH → VKA (INR 2–3) or DOAC × 6 months minimum",
            "Cirrhotic PVT: Anticoagulation recommended — LMWH (safest data) or DOAC; VKA difficult to manage (variable INR in cirrhosis)",
            "Splanchnic vein thrombosis: Same anticoagulation strategy",
            "Goal: Recanalization (up to 50% with anticoagulation) and prevention of extension",
        ],
        "second_line": [
            "TIPS: For PVT with complications (variceal bleeding, refractory ascites) or pre-transplant to maximize recanalization",
            "Catheter-directed thrombolysis: Acute PVT with bowel ischemia (superior mesenteric vein thrombosis)",
            "Indefinite anticoagulation: If underlying thrombophilia, recurrent thrombosis, or cancer-associated",
        ],
        "goals": [
            "Complete or partial recanalization of portal vein",
            "Prevention of extension to mesenteric veins",
            "Management of portal hypertension complications",
            "Optimize hepatic blood flow before liver transplantation",
        ],
        "monitoring": [
            "Doppler ultrasound or CT/MRI venography every 3–6 months",
            "LFTs, CBC, INR every 1–3 months",
            "Thrombophilia screen: JAK2, Factor V Leiden, prothrombin mutation, antiphospholipid panel",
            "Upper endoscopy if portal hypertension (variceal surveillance)",
        ],
        "special": [
            "Non-cirrhotic PVT: Consider local causes (pancreatitis, cholecystitis, appendicitis, IBD)",
            "Cirrhotic PVT: AASLD 2021 recommends anticoagulation even in cirrhosis if renal function preserved",
            "Cavernous transformation: Collateral vessels; conservative management; treat complications",
            "HCC can cause PVT (tumor thrombus) — distinguish from benign PVT by imaging",
        ],
        "source": "EASL Vascular Liver Diseases CPG 2016/2025; AASLD CPG 2021",
    },

    "aclf": {
        "first_line": [
            "IDENTIFY AND TREAT PRECIPITANT (most important step):",
            "Bacterial infection: Broad-spectrum antibiotics (piperacillin-tazobactam 4.5 g IV q8h or meropenem 1 g IV q8h based on local resistance)",
            "GI bleeding: Vasoactive drugs + EBL + antibiotics (see variceal hemorrhage protocol)",
            "Alcohol: Abstinence + corticosteroids if severe AH (prednisolone 40 mg/day if MDF ≥32)",
            "Reactivation of viral hepatitis: Antiviral therapy",
            "Organ support in ICU: Vasopressors (norepinephrine first-line) for circulatory failure",
            "Renal replacement therapy (CRRT preferred) for AKI or HRS",
            "Albumin infusion 20–40 g/day IV (may reduce organ failure progression)",
            "Early enteral nutrition: 35–40 kcal/kg/day; 1.2–1.5 g protein/kg/day",
        ],
        "second_line": [
            "Liver transplantation: Most effective treatment for ACLF-2 and ACLF-3 (high-priority listing)",
            "ACLF-3 with CLIF-C ACLF >64: Consider futility (60% 28-day mortality; transplant may not improve outcome)",
            "G-CSF (filgrastim): NOT routinely recommended (EASL 2023 — conflicting evidence)",
            "MARS/PROMETHEUS albumin dialysis: Bridge to transplant; individualized use",
            "TIPS: Generally contraindicated in ACLF (worsens HE, hemodynamic instability); use only in selected cases",
        ],
        "goals": [
            "ACLF Grade 1: 28-day mortality ~22%; aggressive treatment may downgrade to AD",
            "ACLF Grade 2: 28-day mortality ~32%; transplant evaluation urgent",
            "ACLF Grade 3: 28-day mortality ~60%; immediate transplant listing; futility assessment",
            "Remove/treat precipitant — ACLF may resolve if precipitant treated adequately",
        ],
        "monitoring": [
            "CLIF-C OF score (SOFA-based) daily for organ failure assessment",
            "CLIF-C ACLF score at diagnosis and day 3–7 (dynamic prognosis)",
            "Blood cultures, procalcitonin, CRP daily",
            "Bilirubin, creatinine, INR, sodium, WBC daily",
            "MAP, urine output hourly in ICU",
            "Lactate, arterial blood gas every 6–12 hours",
        ],
        "special": [
            "EASL-CLIF definition: Acute decompensation + ≥1 organ failure (liver, kidney, brain, coagulation, circulation, respiration)",
            "APASL definition broader: hepatic insult + liver failure (bilirubin ≥5 mg/dL + INR ≥1.5) with or without prior cirrhosis",
            "Dynamic assessment: Re-assess ACLF grade at day 3–7; improvement or resolution possible",
            "ACLF from non-cirrhotic liver: Manage as acute liver failure protocol",
            "HCC with ACLF: Poor outcomes; LT not indicated if HCC outside criteria",
        ],
        "source": "EASL ACLF CPG 2023; AASLD Practice Guidance 2024; APASL ACLF 2019",
    },

    "variceal hemorrhage": {
        "first_line": [
            "RESUSCITATION: Large-bore IV access × 2; target Hgb 7–8 g/dL (restrictive transfusion reduces mortality)",
            "VASOACTIVE DRUGS (start immediately, before endoscopy): Terlipressin 2 mg IV q4h × 48h then 1 mg IV q4h × 3–5 days total; OR Somatostatin 250 mcg IV bolus + 250 mcg/hr × 5 days; OR Octreotide 50 mcg IV bolus + 50 mcg/hr × 5 days",
            "ANTIBIOTICS (mandatory): Ceftriaxone 1 g IV daily × 7 days (reduces bacterial infections and rebleeding; preferred in advanced cirrhosis)",
            "URGENT ENDOSCOPY within 12 hours: Endoscopic band ligation (EBL) first-line for esophageal varices; tissue adhesive (cyanoacrylate) for gastric varices",
            "AIRWAY: Intubate for grade III/IV HE before endoscopy to prevent aspiration",
        ],
        "second_line": [
            "Pre-emptive TIPS within 72h (ideally <24h): Child-Pugh C <14 OR Child-Pugh B + active bleeding at endoscopy OR HVPG ≥20 mmHg — PTFE-covered stent; reduces rebleeding and mortality (HINT/PREVENT trials)",
            "Rescue TIPS: Failure of endoscopic + pharmacological therapy (persistent bleeding)",
            "Balloon tamponade (Sengstaken-Blakemore or Linton-Nachlas): Temporary bridge max 24h awaiting TIPS",
            "Self-expanding metal stents (SEMS, SX-ELLA stent): Alternative to balloon tamponade",
        ],
        "goals": [
            "Hemostasis within first endoscopic session (>85% with EBL + vasoactive drugs)",
            "Prevent 5-day treatment failure (<10% with optimal management)",
            "Reduce 6-week mortality (remains ~10–15% despite advances)",
            "Secondary prophylaxis: NSBB + EBL every 2–4 weeks until variceal eradication",
        ],
        "monitoring": [
            "Hemodynamic monitoring continuous (HR, BP, MAP)",
            "CBC, coagulation, LFTs, creatinine every 6–12 hours",
            "Renal function (AKI common complication; monitor creatinine daily)",
            "Hepatic encephalopathy assessment every shift",
            "Post-EBL: Upper endoscopy every 2–4 weeks until variceal eradication, then every 3 months × 1 year, then every 6 months",
        ],
        "special": [
            "Baveno VII: Pre-emptive TIPS in Child-Pugh B + active bleeding OR Child-Pugh C <14 reduces rebleeding by 50%",
            "Avoid fresh frozen plasma (no evidence; increases portal pressure); proton pump inhibitors standard post-banding",
            "Gastric varices: GOV2 or IGV — cyanoacrylate injection OR TIPS; BRTO (balloon-occluded retrograde transvenous obliteration) for fundal varices",
            "NSBB contraindicated if SBP, HRS, or severe circulatory failure (EASL)",
            "Carvedilol 6.25–12.5 mg/day preferred NSBB (superior portal pressure reduction vs propranolol)",
        ],
        "source": "Baveno VII Portal Hypertension Consensus 2021; EASL Decompensated Cirrhosis CPG 2018; AASLD 2023",
    },

    "sbp": {
        "first_line": [
            "Diagnostic paracentesis if ascites fluid PMN ≥250 cells/mm³ = SBP (culture may be negative)",
            "Cefotaxime 2 g IV q8h × 5 days (community-acquired or uncomplicated)",
            "Ceftriaxone 1 g IV daily × 5 days (alternative; once-daily dosing)",
            "Albumin IV (MANDATORY with antibiotics): 1.5 g/kg on Day 1 + 1 g/kg on Day 3 — prevents HRS; reduces in-hospital mortality (Sort et al., NEJM 1999)",
            "Albumin indication: Creatinine >1 mg/dL OR BUN >30 mg/dL OR bilirubin >4 mg/dL",
        ],
        "second_line": [
            "Healthcare-associated/nosocomial SBP or prior fluoroquinolone prophylaxis: Broader coverage — piperacillin-tazobactam 4.5 g IV q6h OR meropenem 1 g IV q8h (based on local resistance patterns)",
            "Oral ofloxacin 400 mg BID × 8 days: Selected uncomplicated SBP (no encephalopathy, no vomiting, no renal failure — Naveau criteria)",
        ],
        "goals": [
            "PMN reduction >25% at repeat paracentesis (48 hours) = treatment response",
            "Prevention of HRS (most feared complication — albumin is key)",
            "SBP-associated in-hospital mortality 15–20% (higher with renal failure)",
        ],
        "monitoring": [
            "Repeat paracentesis at 48 hours: PMN count (treatment failure if <25% reduction — change antibiotics)",
            "Creatinine, sodium, urine output daily",
            "Blood cultures + ascites culture at diagnosis (BACTEC bottles — bedside inoculation increases yield)",
            "CBC, LFTs daily during treatment",
        ],
        "special": [
            "SBP prophylaxis — long-term secondary: Norfloxacin 400 mg PO daily (indefinite); alternative ciprofloxacin 500 mg/day",
            "SBP prophylaxis — primary (ascites protein <1.5 g/dL + renal/liver impairment): Norfloxacin 400 mg/day",
            "GI hemorrhage prophylaxis: Ceftriaxone 1 g IV daily × 7 days (all cirrhotics with GI bleeding)",
            "Bacterascites (positive culture, PMN <250): Treat if symptomatic or repeat PMN ≥250",
            "Trimethoprim-sulfamethoxazole DS 1 tablet daily: Alternative if norfloxacin unavailable",
        ],
        "source": "AASLD Ascites/SBP/HRS CPG 2021; EASL Decompensated Cirrhosis CPG 2018",
    },

    "hepatorenal syndrome": {
        "first_line": [
            "STOP nephrotoxic drugs immediately: NSAIDs, aminoglycosides, contrast agents, diuretics",
            "Volume expansion trial: Albumin 1 g/kg/day IV × 2 days (max 100 g/day) — rule out hypovolemia",
            "If no creatinine improvement after 2 days: HRS confirmed → start terlipressin",
            "Terlipressin (PREFERRED): 0.5–1 mg IV q4–6h; increase to 2 mg IV q4–6h if creatinine not decreasing by day 4 (maximum 14 days)",
            "Albumin with terlipressin: 20–40 g/day IV throughout treatment",
            "STOP terlipressin if: No response (creatinine not decreasing by ≥25% at maximum dose by day 4); or terlipressin side effects (ischemia, severe hyponatremia)",
        ],
        "second_line": [
            "Norepinephrine (ICU only): 0.5–3 mg/hour IV; titrate to increase MAP by ≥10 mmHg OR achieve CVP 10–15 cmH2O — equally effective to terlipressin in RCTs",
            "Midodrine 7.5–12.5 mg PO TID + Octreotide 100–200 mcg SC TID (or 50 mcg/hr IV): Outpatient option; less effective than terlipressin but safer",
            "TIPS: TIPS-eligible (Child-Pugh ≤13, no severe HE) if HRS not responding",
            "Renal replacement therapy: Bridge to transplant; does NOT treat HRS — only supportive",
            "Liver transplantation: DEFINITIVE treatment; SLKT (simultaneous liver-kidney transplant) if AKI ≥4–6 weeks or CKD with eGFR <25 mL/min",
        ],
        "goals": [
            "Complete response: Creatinine <1.5 mg/dL",
            "Partial response: ≥50% decrease from peak creatinine",
            "Terlipressin complete response rate ~30–40% (CONFIRM/REVERSE trials)",
            "Bridge to liver transplantation",
        ],
        "monitoring": [
            "Creatinine, urine output, MAP every 6–12 hours",
            "Daily weight, intake/output balance",
            "Electrolytes (watch for hyponatremia worsening on terlipressin)",
            "Cardiac monitoring (terlipressin: arrhythmia, ischemia risk)",
            "CBC daily (vasopressin analogue may cause thrombocytopenia)",
        ],
        "special": [
            "ICA 2019 criteria: Creatinine >1.5 mg/dL OR increase ≥0.3 mg/dL within 48h; no improvement after ≥2 days of diuretic hold + albumin; no shock; no nephrotoxins; no parenchymal kidney disease",
            "HRS-AKI staging: Stage 1A (<1.5 mg/dL but increase ≥0.3); Stage 1B (1.5–2.0); Stage 2 (2.0–3.5); Stage 3 (>3.5 or need for RRT)",
            "Terlipressin FDA-approved USA 2022 (Terlivaz); previously only available in Europe",
            "Avoid beta-blockers if MAP <65 mmHg (Baveno VII recommendation)",
            "Urinary sodium <10 mEq/L supports HRS (fractional excretion of sodium <0.1%)",
        ],
        "source": "AASLD HRS CPG 2021; EASL 2018/2023 update; ICA 2020 consensus",
    },

    "hepatic encephalopathy": {
        "first_line": [
            "IDENTIFY AND CORRECT PRECIPITANT FIRST: GI bleeding (endoscopy), infection (antibiotics), dehydration (IV fluids), constipation (lactulose), hypokalemia (KCl), benzodiazepines (flumazenil 0.2–1 mg IV if suspected)",
            "Lactulose 20–30 mL PO TID–QID; titrate to 2–3 soft bowel movements per day",
            "Lactulose enema (if unable to take PO): 300 mL lactulose in 700 mL water via rectal tube",
            "Rifaximin 550 mg PO BID: Add for secondary prophylaxis after ≥2nd episode within 6 months (adjunct to lactulose)",
            "Nutritional support: 35–40 kcal/kg/day; 1.2–1.5 g protein/kg/day (DO NOT restrict protein — harmful); late evening snack recommended",
        ],
        "second_line": [
            "L-Ornithine L-Aspartate (LOLA): 9 g TID PO or IV infusion — adjunct; some evidence for reduction of ammonia",
            "Zinc supplementation: 220 mg zinc sulfate BID (zinc deficiency common in cirrhosis; augments urea cycle)",
            "Branched-chain amino acids (BCAA): Oral supplements if protein-intolerant (rare — protein restriction obsolete)",
            "Flumazenil 0.5–1 mg IV: Short-term improvement in HE if benzodiazepines suspected precipitant",
            "Neomycin 250–500 mg QID: Limited use; nephrotoxic and ototoxic",
        ],
        "goals": [
            "Grade 0 target (no HE) for chronic management",
            "Covert/minimal HE: Improve quality of life and driving ability",
            "Secondary prophylaxis after 1st episode: Lactulose 15–45 mL BID–TID",
            "Secondary prophylaxis after ≥2nd episode: Lactulose + rifaximin 550 mg BID",
        ],
        "monitoring": [
            "West Haven grade assessment at each visit",
            "Serum ammonia (fasting venous): Not diagnostic but trend useful; >100 mcg/dL suggests HE",
            "Electrolytes (especially sodium, potassium) — correct hypokalemia",
            "Renal function (precipitant identification)",
            "Psychometric tests for minimal/covert HE (Psychometric HE Score, Stroop EncephalApp)",
        ],
        "special": [
            "West Haven Grades: 0 = none; 1 = asterixis, subtle; 2 = disorientation, flapping tremor; 3 = somnolence, confusion; 4 = coma",
            "Ammonia itself is NOT toxic in isolation — inflammatory cytokines + ammonia synergistic",
            "TIPS-related HE: Reduce TIPS shunt diameter; rifaximin + lactulose",
            "Minimal/Covert HE: Screen with psychometric tests; impairs driving — warn patients",
            "Number Connection Test (NCT), Psychometric Hepatic Encephalopathy Score (PHES) for covert HE",
        ],
        "source": "EASL-AASLD Joint CPG 2014; EASL HE CPG 2022",
    },

    "icp": {
        "first_line": [
            "Ursodeoxycholic acid (UDCA) 10–15 mg/kg/day PO in 2–3 divided doses (first-line; reduces pruritus, lowers bile acids, potential fetal benefit)",
            "Vitamin K supplementation if prolonged cholestasis (risk of maternal coagulopathy and neonatal bleeding)",
            "Cholestyramine 4–8 g/day for refractory pruritus (separate from other medications by 4 hours; reduces fat-soluble vitamin absorption)",
        ],
        "second_line": [
            "Rifampicin 150 mg BID (off-label; effective for pruritus; avoid in 1st trimester; monitor LFTs)",
            "Antihistamines (diphenhydramine, hydroxyzine) for sleep disruption",
            "Topical emollients and cooling for pruritus relief",
        ],
        "goals": [
            "Reduction of maternal pruritus",
            "Normalization or reduction of serum bile acids (<40 mcmol/L optimal)",
            "Prevention of adverse fetal outcomes (stillbirth, preterm delivery, fetal distress)",
            "Safe delivery at optimal gestational age",
        ],
        "monitoring": [
            "Serum bile acids every 1–2 weeks (fasting specimen preferred)",
            "LFTs (ALT, AST, bilirubin) every 2 weeks",
            "Fetal monitoring: Kick counts; Non-stress test/BPP from 32–34 weeks",
            "Umbilical artery Doppler for severe cases (bile acids ≥100 mcmol/L)",
        ],
        "special": [
            "Delivery timing (SMFM 2020): Bile acids <40 mcmol/L = 36–39 weeks; 40–99 = 36–39 weeks; ≥100 mcmol/L = 36 weeks",
            "Recurrence in subsequent pregnancies ~45–70%",
            "Associated with increased risk of pre-eclampsia, gestational diabetes",
            "Post-partum: LFTs/bile acids normalize within 2–6 weeks; screen for PBC/PSC",
            "Avoid OCP (estrogen/progesterone can trigger ICP) in future contraception planning",
        ],
        "source": "EASL Liver Diseases in Pregnancy CPG 2023; SMFM Consult Series #53 2020; RCOG guidelines",
    },

    "aflp": {
        "first_line": [
            "IMMEDIATE DELIVERY — cornerstone of management regardless of gestational age",
            "Stabilize mother before delivery if possible: IV glucose (dextrose 10–50%); correct hypoglycemia (maintain glucose >60 mg/dL)",
            "FFP, cryoprecipitate, platelets for coagulopathy (DIC common)",
            "ICU care: Monitor for multiorgan failure; vasopressors if hemodynamically unstable",
            "Renal replacement therapy if acute kidney injury",
            "Enteral nutrition post-delivery if encephalopathic; hepatic diet",
        ],
        "second_line": [
            "N-acetylcysteine IV: Some centers use as adjunct (limited evidence for non-acetaminophen DILI)",
            "Plasmapheresis: If not improving post-delivery (remove lipid metabolites)",
            "MARS albumin dialysis: Bridge to recovery or transplantation",
            "Liver transplantation: ~14% of cases require LT; AFLP may recur with LT if underlying LCHAD mutation",
        ],
        "goals": [
            "Maternal survival (mortality 18% historically; improves with aggressive management to <2%)",
            "Fetal survival (prematurity main complication)",
            "Recovery of liver function (69% spontaneous recovery post-delivery)",
        ],
        "monitoring": [
            "Glucose every 1–2 hours (severe hypoglycemia common)",
            "LFTs, INR, fibrinogen, D-dimer every 4–6 hours",
            "Creatinine, electrolytes every 4–6 hours",
            "CBC, blood smear (microangiopathic hemolytic anemia can overlap with HELLP)",
            "Neonatal screen: Long-chain 3-hydroxyacyl-CoA dehydrogenase (LCHAD) deficiency in infant and parents",
        ],
        "special": [
            "Swansea Criteria ≥6 criteria = AFLP diagnosis",
            "LCHAD deficiency in fetus: ~20% of AFLP cases — screen neonate and parents; autosomal recessive",
            "Differentiate from HELLP syndrome: HELLP has hypertension, more severe thrombocytopenia, less coagulopathy",
            "Recurrence in subsequent pregnancies ~20% (monitor closely with bile acids/LFTs in 3rd trimester)",
            "Post-delivery: LFTs peak at day 1–4, then begin improving; full recovery 4–8 weeks",
        ],
        "source": "EASL Liver Diseases in Pregnancy CPG 2023; RCOG Green-Top Guideline; ACG",
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
        # New diseases
        elif "hdv" in diagnosis_lower or "hepatitis d" in diagnosis_lower or "delta" in diagnosis_lower:
            matched_key = "hepatitis d"
        elif "hev" in diagnosis_lower or "hepatitis e" in diagnosis_lower:
            matched_key = "hepatitis e"
        elif "hav" in diagnosis_lower or "hepatitis a" in diagnosis_lower:
            matched_key = "hepatitis a"
        elif "wilson" in diagnosis_lower or "copper" in diagnosis_lower:
            matched_key = "wilson"
        elif "hemochromatosis" in diagnosis_lower or "iron overload" in diagnosis_lower or "haemochromatosis" in diagnosis_lower:
            matched_key = "hemochromatosis"
        elif "alpha" in diagnosis_lower and ("antitrypsin" in diagnosis_lower or "a1at" in diagnosis_lower or "aat" in diagnosis_lower):
            matched_key = "alpha1 antitrypsin"
        elif "psc" in diagnosis_lower or "sclerosing cholangitis" in diagnosis_lower:
            matched_key = "psc"
        elif "overlap" in diagnosis_lower:
            matched_key = "overlap syndrome"
        elif "budd" in diagnosis_lower or "chiari" in diagnosis_lower or "hepatic vein thrombosis" in diagnosis_lower:
            matched_key = "budd chiari"
        elif "portal vein thrombosis" in diagnosis_lower or "pvt" in diagnosis_lower:
            matched_key = "portal vein thrombosis"
        elif "aclf" in diagnosis_lower or "acute-on-chronic" in diagnosis_lower or "acute on chronic" in diagnosis_lower:
            matched_key = "aclf"
        elif "variceal" in diagnosis_lower or "variceal hemorrhage" in diagnosis_lower or "variceal bleed" in diagnosis_lower:
            matched_key = "variceal hemorrhage"
        elif "sbp" in diagnosis_lower or "spontaneous bacterial peritonitis" in diagnosis_lower:
            matched_key = "sbp"
        elif "hepatorenal" in diagnosis_lower or "hrs" in diagnosis_lower:
            matched_key = "hepatorenal syndrome"
        elif "hepatic encephalopathy" in diagnosis_lower or " he " in diagnosis_lower:
            matched_key = "hepatic encephalopathy"
        elif "ascites" in diagnosis_lower:
            matched_key = "ascites" if "ascites" in TREATMENT_DB else "liver cirrhosis"
        elif "intrahepatic cholestasis" in diagnosis_lower or "icp" in diagnosis_lower or "obstetric cholestasis" in diagnosis_lower:
            matched_key = "icp"
        elif "acute fatty liver" in diagnosis_lower or "aflp" in diagnosis_lower:
            matched_key = "aflp"

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
    "ACLF Grade 3": {
        "keywords": ["aclf", "acute-on-chronic liver failure", "multiple organ failure", "3 organ failures"],
        "lab_trigger": {"inr": (">2.5", "INR >2.5"), "bilirubin": (">12", "Bilirubin >12 mg/dL")},
        "level": "CRITICAL",
        "action": "IMMEDIATE transplant center referral; ICU admission; organ support; CLIF-C ACLF scoring",
        "timeframe": "Within 1 hour",
    },
    "Acute Liver Failure — King's Criteria": {
        "keywords": ["acute liver failure", "no prior liver disease", "encephalopathy coagulopathy"],
        "lab_trigger": {"inr": (">6.5", "INR >6.5 — King's College Criteria met")},
        "level": "CRITICAL",
        "action": "Emergency transplant listing (UNOS Status 1A); N-acetylcysteine if acetaminophen; ICP monitoring",
        "timeframe": "Immediate",
    },
    "Acute Variceal Hemorrhage": {
        "keywords": ["hematemesis", "blood vomiting", "coffee ground", "melena", "massive gi bleed"],
        "level": "CRITICAL",
        "action": "Blood transfusion (target Hgb 7-8); terlipressin/octreotide NOW; ceftriaxone 1g IV; endoscopy within 12h",
        "timeframe": "Within 1 hour",
    },
    "Wilson Disease Fulminant": {
        "keywords": ["wilson", "coombs negative hemolysis", "acute liver failure wilson", "kayser fleischer"],
        "lab_trigger": {"bilirubin": (">10", "Bilirubin >10 — suspect fulminant Wilson")},
        "level": "CRITICAL",
        "action": "Urgent transplant evaluation; plasmapheresis as bridge; copper chelation (trientine)",
        "timeframe": "Within 2 hours",
    },
    "Acute Fatty Liver of Pregnancy": {
        "keywords": ["aflp", "acute fatty liver pregnancy", "swansea criteria", "jaundice hypoglycemia pregnancy"],
        "level": "CRITICAL",
        "action": "IMMEDIATE DELIVERY; ICU admission; correct hypoglycemia; correct coagulopathy; obstetric emergency",
        "timeframe": "Immediate",
    },
    "Maddrey Discriminant Function ≥32": {
        "keywords": ["severe alcoholic hepatitis", "alcoholic hepatitis mdf", "discriminant function"],
        "lab_trigger": {"bilirubin": (">8", "Bilirubin >8 mg/dL in AH context")},
        "level": "URGENT",
        "action": "Rule out infection/GI bleeding; prednisolone 40 mg/day if no contraindications; Lille score at day 7",
        "timeframe": "Within 12 hours",
    },
    "Critical Ammonia": {
        "lab_trigger": {"ammonia": (">100", "Ammonia >100 mcg/dL")},
        "keywords": ["hyperammonemia", "ammonia elevation"],
        "level": "URGENT",
        "action": "Lactulose enema + PO; identify precipitant; evaluate Grade III/IV HE; consider ICU",
        "timeframe": "Within 4 hours",
    },
    "Budd-Chiari Acute": {
        "keywords": ["budd chiari", "hepatic vein thrombosis", "sudden ascites tender hepatomegaly"],
        "level": "URGENT",
        "action": "Anticoagulation immediately (LMWH); Doppler US confirmation; thrombophilia screen; consider TIPS",
        "timeframe": "Within 6 hours",
    },
    "HCC Rupture / Bleeding": {
        "keywords": ["hcc rupture", "liver mass rupture", "hemoperitoneum hcc", "bleeding hcc"],
        "level": "CRITICAL",
        "action": "Emergency CT; hemodynamic resuscitation; transarterial embolization (TAE); surgical consultation",
        "timeframe": "Immediate",
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


# ─── Tool 8: Calculate Advanced Hepatology Scores ───────────────────────────

def calculate_advanced_scores(
    # FIB-4 / APRI inputs
    age: int | None = None,
    ast: float | None = None,
    alt: float | None = None,
    platelets: float | None = None,
    # Alcoholic hepatitis
    pt_patient: float | None = None,       # PT in seconds (patient)
    pt_control: float | None = None,       # PT in seconds (control, typically ~12)
    wbc: float | None = None,              # WBC ×10⁹/L
    urea: float | None = None,             # Urea mmol/L
    bilirubin_mgdl: float | None = None,   # Bilirubin mg/dL
    bilirubin_umol: float | None = None,   # Bilirubin µmol/L
    creatinine: float | None = None,       # Creatinine mg/dL
    inr: float | None = None,
    albumin: float | None = None,          # g/dL
    # Lille score
    albumin_day0_gl: float | None = None,  # Albumin g/L at Day 0
    bilirubin_day0_umol: float | None = None,
    bilirubin_day7_umol: float | None = None,
    renal_failure: int = 0,                # 0=no, 1=yes (Cr >1.3 mg/dL)
    # PAGE-B
    sex: str = "",                         # "M" or "F"
    # MELD 3.0
    sodium: float | None = None,
) -> dict:
    """Calculate advanced hepatology scoring systems:
    FIB-4, APRI, Maddrey DF, GAHS, ABIC, Lille, MELD 3.0, PAGE-B.
    Provide only the variables you have; scores with insufficient data are skipped.
    """
    results = {}
    missing = []

    # ── FIB-4 ─────────────────────────────────────────────────────────────────
    if all(v is not None for v in [age, ast, alt, platelets]) and alt > 0 and platelets > 0:
        fib4 = (age * ast) / (platelets * math.sqrt(alt))
        fib4 = round(fib4, 2)
        if fib4 < 1.3:
            fib4_risk = "Low (F0-F1)"
            fib4_action = "No significant fibrosis likely; reassess in 1–2 years"
        elif fib4 <= 2.67:
            fib4_risk = "Indeterminate"
            fib4_action = "Proceed to Fibroscan or liver biopsy for definitive staging"
        else:
            fib4_risk = "High (F3-F4)"
            fib4_action = "Advanced fibrosis/cirrhosis likely; specialist referral; HCC surveillance"
        # Age-adjusted cutoffs (≥65 years: <2.0 = low)
        if age >= 65 and fib4 < 2.0:
            fib4_risk = "Low (age-adjusted cutoff ≥65)"
            fib4_action = "Low fibrosis risk by age-adjusted criteria"
        results["fib4"] = {
            "score": fib4,
            "risk": fib4_risk,
            "action": fib4_action,
            "formula": f"({age} × {ast}) / ({platelets} × √{alt})",
            "cutoffs": "Low <1.3 (or <2.0 if ≥65 yrs) | Indeterminate 1.3–2.67 | High >2.67",
            "source": "Sterling et al., Hepatology 2006",
        }
    else:
        missing.append("FIB-4: requires age, AST, ALT, platelets")

    # ── APRI ──────────────────────────────────────────────────────────────────
    ast_uln = 40  # Standard ULN for AST
    if ast is not None and platelets is not None and platelets > 0:
        apri = (ast / ast_uln) / platelets * 100
        apri = round(apri, 2)
        if apri < 0.5:
            apri_risk = "Low fibrosis risk (F0-F1)"
        elif apri < 1.5:
            apri_risk = "Indeterminate (F1-F2)"
        elif apri < 2.0:
            apri_risk = "Significant fibrosis (≥F2)"
        else:
            apri_risk = "Probable cirrhosis (F3-F4)"
        results["apri"] = {
            "score": apri,
            "risk": apri_risk,
            "formula": f"(AST/ULN) / Platelets × 100 = ({ast}/{ast_uln}) / {platelets} × 100",
            "cutoffs": "<0.5 = low | 0.5–1.5 = indeterminate | >1.5 = significant fibrosis | >2.0 = cirrhosis",
            "source": "WHO HCV treatment guidance; Wai et al., Hepatology 2003",
        }
    else:
        missing.append("APRI: requires AST, platelets")

    # ── Maddrey Discriminant Function (MDF) ───────────────────────────────────
    if pt_patient is not None and bilirubin_mgdl is not None:
        pt_ctrl = pt_control if pt_control else 12.0
        mdf = 4.6 * (pt_patient - pt_ctrl) + bilirubin_mgdl
        mdf = round(mdf, 1)
        if mdf >= 32:
            mdf_interp = "SEVERE alcoholic hepatitis — consider prednisolone 40 mg/day if no contraindications"
            mdf_action = "Rule out infection/GI bleeding → prednisolone 40 mg/day × 28 days; Lille score at day 7"
        else:
            mdf_interp = "Non-severe alcoholic hepatitis"
            mdf_action = "Supportive care; abstinence; nutrition; monitor closely"
        results["maddrey_df"] = {
            "score": mdf,
            "interpretation": mdf_interp,
            "action": mdf_action,
            "formula": f"4.6 × ({pt_patient} − {pt_ctrl}) + {bilirubin_mgdl}",
            "cutoff": "≥32 = severe AH; treat with prednisolone",
            "source": "Maddrey et al., Gastroenterology 1978; AASLD/EASL/ACG ALD guidelines",
        }
    else:
        missing.append("Maddrey DF: requires PT (patient), bilirubin (mg/dL)")

    # ── GAHS (Glasgow Alcoholic Hepatitis Score) ───────────────────────────────
    if all(v is not None for v in [age, wbc, urea, bilirubin_umol]) and inr is not None:
        # Age
        age_pts = 2 if age >= 50 else 1
        # WBC
        if wbc < 15:
            wbc_pts = 1
        elif wbc <= 25:
            wbc_pts = 2
        else:
            wbc_pts = 3
        # Urea (mmol/L)
        if urea < 5:
            urea_pts = 1
        elif urea <= 7.5:
            urea_pts = 2
        else:
            urea_pts = 3
        # PT ratio (patient PT / control PT)
        pt_ratio = inr  # INR ≈ PT ratio
        if pt_ratio < 1.5:
            pt_pts = 1
        elif pt_ratio <= 2.0:
            pt_pts = 2
        else:
            pt_pts = 3
        # Bilirubin µmol/L
        if bilirubin_umol < 125:
            bili_pts = 1
        elif bilirubin_umol <= 250:
            bili_pts = 2
        else:
            bili_pts = 3
        gahs = age_pts + wbc_pts + urea_pts + pt_pts + bili_pts
        gahs_interp = "SEVERE — treat with prednisolone" if gahs >= 9 else "Non-severe — supportive care"
        results["gahs"] = {
            "score": gahs,
            "interpretation": gahs_interp,
            "components": {
                "age": age_pts, "wbc": wbc_pts, "urea": urea_pts, "inr/pt_ratio": pt_pts, "bilirubin": bili_pts
            },
            "cutoff": "≥9 = severe AH; sensitivity ~80%, specificity ~81%",
            "source": "Forrest et al., Gut 2005; EASL ALD CPG 2018",
        }
    else:
        missing.append("GAHS: requires age, WBC, urea, bilirubin (µmol/L), INR")

    # ── ABIC Score ────────────────────────────────────────────────────────────
    if all(v is not None for v in [age, bilirubin_mgdl, creatinine, inr]):
        abic = (age * 0.1) + (bilirubin_mgdl * 0.08) + (creatinine * 0.3) + (inr * 0.8)
        abic = round(abic, 2)
        if abic < 6.71:
            abic_risk = "Low risk (0% 90-day mortality)"
        elif abic <= 9.0:
            abic_risk = "Intermediate risk (30% 90-day mortality)"
        else:
            abic_risk = "High risk (75% 90-day mortality)"
        results["abic"] = {
            "score": abic,
            "risk": abic_risk,
            "formula": f"({age}×0.1) + ({bilirubin_mgdl}×0.08) + ({creatinine}×0.3) + ({inr}×0.8)",
            "cutoffs": "<6.71 = low | 6.71–9.0 = intermediate | >9.0 = high risk",
            "source": "Dominguez et al., Hepatology 2008; EASL ALD guidelines",
        }
    else:
        missing.append("ABIC: requires age, bilirubin (mg/dL), creatinine, INR")

    # ── Lille Score (Day-7 steroid response in AH) ────────────────────────────
    if all(v is not None for v in [age, albumin_day0_gl, bilirubin_day0_umol, bilirubin_day7_umol, creatinine, inr]):
        bilirubin_evolution = bilirubin_day0_umol - bilirubin_day7_umol
        lille = (3.19
                 - 0.101 * age
                 + 0.147 * albumin_day0_gl
                 + 0.0165 * bilirubin_evolution
                 - 0.206 * renal_failure
                 - 0.0065 * bilirubin_day0_umol
                 - 0.0096 * (inr * 12))  # PT sec approximation
        lille = round(lille, 3)
        if lille <= 0.45:
            lille_interp = "RESPONDER — continue prednisolone full course (28 days)"
        elif lille <= 0.56:
            lille_interp = "PARTIAL RESPONDER — continue with close monitoring; transplant evaluation"
        else:
            lille_interp = "NULL RESPONDER (Lille >0.56) — STOP prednisolone; 6-month mortality ~75%"
        results["lille"] = {
            "score": lille,
            "interpretation": lille_interp,
            "bilirubin_evolution_umol": round(bilirubin_evolution, 1),
            "cutoffs": "≤0.45 = responder | 0.45–0.56 = partial | >0.56 = null responder (stop steroids)",
            "source": "Louvet et al., Hepatology 2007; EASL/AASLD ALD guidelines",
        }
    else:
        missing.append("Lille: requires age, albumin Day-0 (g/L), bilirubin Day-0 and Day-7 (µmol/L), creatinine, INR")

    # ── MELD 3.0 ──────────────────────────────────────────────────────────────
    if all(v is not None for v in [bilirubin_mgdl, inr, creatinine]):
        bili = max(bilirubin_mgdl, 1.0)
        inr_v = max(inr, 1.0)
        cr = max(min(creatinine, 3.0), 1.0)
        na = max(min(sodium, 137), 125) if sodium is not None else 137
        alb = max(min(albumin, 3.5), 1.5) if albumin is not None else 3.5
        sex_pts = 1.33 if sex.upper() == "F" else 0.0

        meld3 = (sex_pts
                 + 4.56 * math.log(bili)
                 + 0.82 * (137 - na)
                 - 0.24 * (137 - na) * math.log(bili)
                 + 9.09 * math.log(inr_v)
                 + 11.14 * math.log(cr)
                 + 1.85 * (3.5 - alb)
                 - 1.83 * (3.5 - alb) * math.log(cr)
                 + 6)
        meld3 = max(round(meld3), 6)

        if meld3 < 10:
            m3_category = "Low"
            m3_mortality = "<2%"
        elif meld3 < 20:
            m3_category = "Moderate"
            m3_mortality = "6–20%"
        elif meld3 < 30:
            m3_category = "High"
            m3_mortality = "20–52%"
        elif meld3 < 40:
            m3_category = "Very High"
            m3_mortality = "52–71%"
        else:
            m3_category = "Critical"
            m3_mortality = ">71%"

        results["meld_3"] = {
            "score": meld3,
            "category": m3_category,
            "three_month_mortality": m3_mortality,
            "sex_bonus_applied": sex.upper() == "F",
            "sodium_used": na,
            "albumin_used": alb,
            "interpretation": (
                f"MELD 3.0 score: {meld3} ({m3_category}). "
                f"3-month mortality: {m3_mortality}. "
                f"{'Transplant listing threshold met (MELD 3.0 ≥15).' if meld3 >= 15 else 'Below transplant listing threshold.'}"
            ),
            "source": "Kim et al., Hepatology 2021; UNOS/OPTN transplant allocation from 2022",
        }
    else:
        missing.append("MELD 3.0: requires bilirubin (mg/dL), INR, creatinine; optionally sodium, albumin, sex")

    # ── PAGE-B Score (HCC risk in HBV patients on antiviral therapy) ──────────
    if age is not None and platelets is not None and sex:
        # Age points
        if age <= 29:
            age_pg = 0
        elif age <= 39:
            age_pg = 5
        elif age <= 49:
            age_pg = 10
        elif age <= 59:
            age_pg = 15
        elif age <= 69:
            age_pg = 20
        else:
            age_pg = 25
        # Sex points
        sex_pg = 3 if sex.upper() == "M" else 0
        # Platelets (×10⁹/L)
        if platelets >= 200:
            plt_pg = 0
        elif platelets >= 100:
            plt_pg = 5
        else:
            plt_pg = 10
        pageb = age_pg + sex_pg + plt_pg
        if pageb <= 9:
            pageb_risk = "Low risk (5-year HCC incidence <1%)"
        elif pageb <= 17:
            pageb_risk = "Intermediate risk (5-year HCC incidence ~1–4%)"
        else:
            pageb_risk = "High risk (5-year HCC incidence >10%) — intensify HCC surveillance"
        results["page_b"] = {
            "score": pageb,
            "risk": pageb_risk,
            "components": {"age_points": age_pg, "sex_points": sex_pg, "platelet_points": plt_pg},
            "indication": "For HBV patients on antiviral therapy (non-cirrhotic)",
            "cutoffs": "≤9 = low | 10–17 = intermediate | ≥18 = high risk",
            "source": "Papatheodoridis et al., J Hepatol 2016; EASL HBV CPG 2017",
        }
    else:
        missing.append("PAGE-B: requires age, sex (M/F), platelets")

    return {
        "scores": results,
        "missing_data": missing,
        "scores_calculated": list(results.keys()),
        "clinical_note": (
            "Scores requiring bilirubin in µmol/L: GAHS (1 mg/dL = 17.1 µmol/L). "
            "Lille score requires Day-0 AND Day-7 bilirubin. "
            "MELD 3.0 is the current UNOS transplant allocation score (replaced MELD-Na in 2022). "
            "FIB-4 and APRI are non-invasive fibrosis markers validated across HCV, HBV, NAFLD."
        ),
    }


# ─── Tool 9: Additional Validated Calculators (medcalc) ─────────────────────

def calculate_additional_clinical_scores(
    # eGFR (Hepatorenal syndrome, TDF dosing)
    creatinine: float | None = None,
    age: int | None = None,
    sex: str = "M",
    # SOFA score (ACLF)
    pao2_fio2: float | None = None,
    platelets_sofa: float | None = None,
    bilirubin_sofa: float | None = None,
    map_mmhg: float | None = None,
    gcs_total: int | None = None,
    creatinine_sofa: float | None = None,
    urine_output_sofa: float | None = None,
    vasopressor: str = "none",
    # HAS-BLED (bleeding risk for anticoagulation — Budd-Chiari, PVT)
    hypertension: bool = False,
    renal_disease: bool = False,
    liver_disease_hasbled: bool = False,
    stroke_history: bool = False,
    prior_bleeding: bool = False,
    labile_inr: bool = False,
    elderly_hasbled: bool = False,
    antiplatelet: bool = False,
    alcohol_hasbled: bool = False,
    # FIB-4 (use medcalc validated)
    ast: float | None = None,
    alt: float | None = None,
    platelets_fib4: float | None = None,
) -> dict:
    """Calculate additional validated clinical scores using medcalc library.
    Includes: eGFR (CKD-EPI), SOFA score, HAS-BLED, FIB-4 (medcalc validated).
    """
    if not _MEDCALC:
        return {"error": "medcalc library not installed. Run: pip install medcalc"}

    results = {}
    missing = []

    # ── eGFR (CKD-EPI) — for HRS staging, TDF/ETV dose adjustment ────────────
    if creatinine is not None and age is not None:
        try:
            egfr = _mc.egfr_epi(
                scr=creatinine,
                age=age,
                male=(str(sex).upper() not in ("F","FEMALE")),
            )
            if egfr >= 90:
                ckd = "G1 — Normal or high (≥90)"
                tdf_dose = "TDF: standard 300 mg/day"
            elif egfr >= 60:
                ckd = "G2 — Mildly decreased (60–89)"
                tdf_dose = "TDF: standard 300 mg/day"
            elif egfr >= 30:
                ckd = "G3 — Moderately decreased (30–59)"
                tdf_dose = "TDF: 300 mg every 48h OR switch to TAF 25 mg/day"
            elif egfr >= 15:
                ckd = "G4 — Severely decreased (15–29)"
                tdf_dose = "TDF: 300 mg every 72–96h OR TAF 25 mg/day (preferred)"
            else:
                ckd = "G5 — Kidney failure (<15)"
                tdf_dose = "TAF 25 mg/day (TDF contraindicated). Entecavir dose adjust."
            results["egfr_ckd_epi"] = {
                "egfr_ml_min_1_73m2": round(egfr, 1),
                "ckd_stage": ckd,
                "tdf_dose_adjustment": tdf_dose,
                "clinical_note": "eGFR required for HRS-AKI staging and antiviral dose adjustment",
            }
        except Exception as e:
            missing.append(f"eGFR: {e}")
    else:
        missing.append("eGFR: requires creatinine + age")

    # ── FIB-4 (medcalc validated version) ─────────────────────────────────────
    if all(v is not None for v in [age, ast, alt, platelets_fib4]) and platelets_fib4 > 0 and alt > 0:
        try:
            fib4 = _mc.fib4_index(age=age, ast=ast, alt=alt, platelets=platelets_fib4)
            fib4 = round(fib4, 2)
            if age >= 65:
                cutoff_low, cutoff_high = 2.0, 2.67
                note = "Age-adjusted: low risk cutoff raised to 2.0 for age ≥65"
            else:
                cutoff_low, cutoff_high = 1.3, 2.67
                note = "Standard cutoffs: <1.3 low, 1.3–2.67 indeterminate, >2.67 high"
            risk = "Low" if fib4 < cutoff_low else ("Indeterminate" if fib4 <= cutoff_high else "High")
            results["fib4_medcalc"] = {
                "score": fib4,
                "risk": risk,
                "note": note,
                "source": "medcalc validated formula",
            }
        except Exception as e:
            missing.append(f"FIB-4: {e}")
    else:
        missing.append("FIB-4: requires age, AST, ALT, platelets")

    # ── HAS-BLED (bleeding risk for anticoagulation decisions) ─────────────────
    # Relevant when anticoagulating Budd-Chiari / PVT / AF in cirrhosis
    try:
        hasbled_score = sum([
            hypertension, renal_disease, liver_disease_hasbled, stroke_history,
            prior_bleeding, labile_inr, elderly_hasbled, antiplatelet, alcohol_hasbled
        ])
        if hasbled_score <= 2:
            bleed_risk = "Low (≤2%/year)"
            anticoag_rec = "Anticoagulation generally appropriate with monitoring"
        elif hasbled_score == 3:
            bleed_risk = "Moderate (~3.7%/year)"
            anticoag_rec = "Use caution; consider reversible risk factors; monitor closely"
        else:
            bleed_risk = "High (>8%/year)"
            anticoag_rec = "High bleeding risk — weigh against thrombosis risk; treat reversible factors first"
        results["has_bled"] = {
            "score": hasbled_score,
            "annual_bleeding_risk": bleed_risk,
            "anticoagulation_recommendation": anticoag_rec,
            "note": "For cirrhosis-associated anticoagulation (Budd-Chiari, PVT). Score ≥3 = high risk.",
        }
    except Exception as e:
        missing.append(f"HAS-BLED: {e}")

    # ── SOFA score (for ACLF severity assessment) ──────────────────────────────
    if all(v is not None for v in [platelets_sofa, bilirubin_sofa, map_mmhg, gcs_total, creatinine_sofa]):
        try:
            sofa = _mc.sofa_score(
                pao2_fio2=pao2_fio2 or 400,
                platelets=platelets_sofa,
                bilirubin=bilirubin_sofa,
                map=map_mmhg,
                gcs=gcs_total,
                creatinine=creatinine_sofa,
                urine_output=urine_output_sofa or 500,
                vasopressor=vasopressor,
            )
            if sofa < 6:
                mortality = "<10%"
            elif sofa < 9:
                mortality = "15–20%"
            elif sofa < 12:
                mortality = "40–50%"
            else:
                mortality = ">80%"
            results["sofa"] = {
                "score": sofa,
                "icu_mortality_estimate": mortality,
                "aclf_relevance": "SOFA ≥2 defines organ failure in EASL-CLIF ACLF criteria",
                "note": "Use CLIF-C OF (modified SOFA) for formal ACLF grading",
            }
        except Exception as e:
            missing.append(f"SOFA: {e}")
    else:
        missing.append("SOFA: requires platelets, bilirubin, MAP, GCS, creatinine")

    return {
        "scores": results,
        "missing_data": missing,
        "calculator_source": "medcalc library (pip install medcalc)",
    }


# ─── Tool 9 (original): Advanced Fibrosis Calculator (MASLD) ───────────────────────────
# Source: github.com/laithomari/advanced_fibrosis_calculator
# Model: L2-regularized logistic regression, trained on 1,581 biopsy-confirmed patients
# Validated: Internal AUROC 0.826, Asian cohort 0.737, NHANES 0.743
# Indication: MASLD/NAFLD patients with INDETERMINATE FIB-4 (1.3–2.67)

_AFC_MODEL = {
    "intercept": -1.2938195574226066,
    "features": ["age", "bmi", "ast_log", "alt_log", "ggt_log", "platelets", "diabetes", "ast_alt_ratio"],
    "coefficients": {
        "age":          0.47812706715595105,
        "bmi":          0.3785487071652713,
        "ast_log":      1.9170632746913627,
        "alt_log":     -1.7219456328868705,
        "ggt_log":      0.46040728958803795,
        "platelets":   -0.6822389111254026,
        "diabetes":     0.3331626014517234,
        "ast_alt_ratio":-0.4753960846614913,
    },
    "scaler_means": {
        "age":          49.951296647691336,
        "bmi":          34.46683744465528,
        "ast_log":      3.6748166929600568,
        "alt_log":      3.924569614865929,
        "ggt_log":      3.865424548612931,
        "platelets":    238320.05060088553,
        "diabetes":     0.33649588867805186,
        "ast_alt_ratio":0.8332148103668242,
    },
    "scaler_stds": {
        "age":          12.529465583430557,
        "bmi":          6.742973909605373,
        "ast_log":      0.5294716136649897,
        "alt_log":      0.627102084646885,
        "ggt_log":      0.7696293820814604,
        "platelets":    75728.73479814474,
        "diabetes":     0.4725107465241611,
        "ast_alt_ratio":0.35912402258659876,
    },
    "thresholds": {"rule_out": 0.2561, "youden": 0.3295, "rule_in": 0.5922},
}


# ─── Tool 10: aMAP Score (HCC Risk, All Etiologies) ─────────────────────────
# Fan et al., J Hepatol 2020;73:1368-1378. PMID: 32097765
# Validated across HBV, HCV, NAFLD, ALD — unlike PAGE-B (HBV only)
# Score 0-100: <50 = Low, 50-60 = Medium, ≥60 = High annual HCC risk

def calculate_amap_hcc_risk(
    age: int,
    ast: float | None = None,
    alt: float | None = None,
    afp: float = 5.0,            # AFP in ng/mL (default normal)
    albumin: float = 4.0,        # Albumin in g/dL
    platelets: float = 200.0,    # Platelets ×10³/μL
    bilirubin: float = 1.0,      # Total bilirubin in mg/dL
    sex: str = "M",
) -> dict:
    """Calculate aMAP score for HCC annual risk in cirrhotic patients.
    Works for ALL cirrhosis etiologies (HBV, HCV, NAFLD, ALD).
    Source: Fan et al., J Hepatol 2020;73:1368-1378 (aMAP score)
    """
    sex_m = 1 if str(sex).upper() in ("M", "MALE") else 0
    albumin_gl = albumin * 10  # g/dL → g/L

    lp = (0.0249 * age
          + 0.0647 * sex_m
          + 0.330 * math.log10(max(afp, 0) + 1)
          - 0.0216 * albumin_gl
          - 0.0148 * platelets
          + 0.273 * bilirubin)

    # Calibrated mapping to 0-100 scale (×12 + 54)
    score = max(0, min(100, round(lp * 12 + 54)))

    if score < 50:
        risk = "Low"
        annual_risk = "~0.4%/year"
        five_yr = "~2%"
        surveillance = "Ultrasound + AFP every 6–12 months"
        color = "green"
    elif score < 60:
        risk = "Medium"
        annual_risk = "~3.3%/year"
        five_yr = "~15%"
        surveillance = "Ultrasound + AFP every 6 months. Consider contrast-enhanced CT/MRI annually."
        color = "orange"
    else:
        risk = "High"
        annual_risk = "~18%/year"
        five_yr = "~60%"
        surveillance = "Ultrasound + AFP every 3–6 months. Annual contrast-enhanced CT or MRI. MDT review."
        color = "red"

    return {
        "amap_score": score,
        "risk_category": risk,
        "annual_hcc_risk": annual_risk,
        "five_year_hcc_risk": five_yr,
        "surveillance_recommendation": surveillance,
        "inputs": {
            "age": age, "sex": "Male" if sex_m else "Female",
            "afp_ngml": afp, "albumin_gdl": albumin,
            "platelets_k": platelets, "bilirubin_mgdl": bilirubin,
        },
        "note": "aMAP validated for HBV/HCV/NAFLD/ALD cirrhotics. Requires pre-existing cirrhosis/advanced fibrosis.",
        "source": "Fan R, et al. J Hepatol 2020;73:1368-1378. PMID: 32097765",
        "comparison_to_page_b": "PAGE-B (HBV only, on antiviral therapy) — use aMAP for non-HBV or combined etiologies.",
    }


# ─── Tool 11: Baveno VII CSPH Criteria ──────────────────────────────────────
# de Franchis R et al. (Baveno VII faculty). J Hepatol 2022;76:959-974
# Rule-out CSPH to avoid EGD for variceal screening

def assess_baveno_csph(
    lsm_kpa: float | None = None,          # Liver Stiffness Measurement (Fibroscan, kPa)
    platelets: float | None = None,         # Platelets ×10³/μL
    ssm_kpa: float | None = None,           # Spleen Stiffness Measurement (kPa, optional)
    viral_suppression: bool = False,        # HCV SVR or HBV viral suppression achieved
    clinical_context: str = "",
) -> dict:
    """Evaluate Baveno VII criteria for Clinically Significant Portal Hypertension (CSPH).
    Determines whether EGD for variceal screening can be safely avoided.
    Source: Baveno VII Consensus, J Hepatol 2022;76:959-974
    """
    result = {
        "lsm_kpa": lsm_kpa,
        "platelets_k": platelets,
        "ssm_kpa": ssm_kpa,
        "viral_suppression": viral_suppression,
        "csph_status": None,
        "egd_recommendation": None,
        "nsbb_recommendation": None,
        "rationale": [],
        "source": "Baveno VII Consensus Workshop, de Franchis R et al. J Hepatol 2022;76:959-974",
    }

    if lsm_kpa is None:
        result["csph_status"] = "Insufficient data"
        result["egd_recommendation"] = "LSM (Fibroscan) required for Baveno VII assessment"
        result["rationale"] = ["LSM not provided — cannot apply Baveno VII criteria"]
        return result

    rationale = []

    # ── Rule OUT CSPH (skip EGD) ──────────────────────────────────────────────
    # Standard (untreated / any etiology)
    if lsm_kpa < 15 and platelets is not None and platelets > 150:
        result["csph_status"] = "CSPH Ruled OUT"
        result["egd_recommendation"] = "EGD NOT required — Baveno VII rule-out criteria met"
        result["nsbb_recommendation"] = "NSBB not indicated (no CSPH)"
        rationale.append(f"LSM {lsm_kpa} kPa < 15 kPa AND Platelets {platelets}k > 150k → CSPH excluded")
        rationale.append("Risk of missing high-risk varices: <5% (Baveno VII validation)")
        result["rationale"] = rationale
        return result

    # Viral suppression (lower LSM threshold): HCV SVR or HBV on antiviral with suppression
    if viral_suppression and lsm_kpa < 12 and platelets is not None and platelets > 150:
        result["csph_status"] = "CSPH Ruled OUT (viral suppression criteria)"
        result["egd_recommendation"] = "EGD NOT required — lower threshold applies with viral suppression"
        result["nsbb_recommendation"] = "NSBB not indicated"
        rationale.append(f"Viral suppression + LSM {lsm_kpa} kPa < 12 kPa + Platelets {platelets}k > 150k")
        rationale.append("HCV SVR or HBV suppression lowers LSM threshold for CSPH rule-out")
        result["rationale"] = rationale
        return result

    # Very low LSM regardless of platelets
    if lsm_kpa < 10:
        result["csph_status"] = "CSPH Very Unlikely"
        result["egd_recommendation"] = "EGD likely not required — LSM < 10 kPa"
        result["nsbb_recommendation"] = "NSBB not indicated"
        rationale.append(f"LSM {lsm_kpa} kPa < 10 kPa — cACLD (compensated advanced CLD) unlikely")
        result["rationale"] = rationale
        return result

    # ── Rule IN CSPH (EGD definitely needed) ──────────────────────────────────
    csph_confirmed = False
    if lsm_kpa >= 25:
        csph_confirmed = True
        rationale.append(f"LSM {lsm_kpa} kPa ≥ 25 kPa → CSPH confirmed (rule-in threshold)")
    elif lsm_kpa >= 20 and platelets is not None and platelets < 150:
        csph_confirmed = True
        rationale.append(f"LSM {lsm_kpa} kPa ≥ 20 kPa AND Platelets {platelets}k < 150k → CSPH likely")
    if ssm_kpa is not None and ssm_kpa >= 21:
        csph_confirmed = True
        rationale.append(f"Spleen stiffness {ssm_kpa} kPa ≥ 21 kPa → CSPH confirmed regardless of LSM")

    if csph_confirmed:
        result["csph_status"] = "CSPH Confirmed"
        result["egd_recommendation"] = (
            "EGD REQUIRED for variceal screening. "
            "If high-risk varices found: start carvedilol 6.25 mg/day or propranolol 20 mg BID, "
            "OR endoscopic band ligation (EBL)."
        )
        result["nsbb_recommendation"] = (
            "Consider starting NSBB prophylaxis: "
            "Carvedilol 6.25 mg/day (preferred, Baveno VII) OR Propranolol 20 mg BID "
            "→ titrate to HR 55–60 bpm"
        )
        result["rationale"] = rationale
        return result

    # ── Gray zone ─────────────────────────────────────────────────────────────
    result["csph_status"] = "Indeterminate (Gray Zone)"
    if lsm_kpa < 15:
        rationale.append(f"LSM {lsm_kpa} kPa < 15 kPa but Platelets {'not provided' if platelets is None else f'{platelets}k ≤ 150k'} → cannot rule out CSPH")
    else:
        rationale.append(f"LSM {lsm_kpa} kPa in 15–25 kPa range — neither rule-out nor rule-in")

    result["egd_recommendation"] = (
        "EGD RECOMMENDED — criteria for safe omission not met. "
        "Consider HVPG measurement (gold standard) or additional non-invasive tests."
    )
    result["nsbb_recommendation"] = "Individualized decision — await EGD results before starting NSBB"

    # Additional guidance for gray zone
    if ssm_kpa is not None:
        if ssm_kpa < 40:
            rationale.append(f"Spleen stiffness {ssm_kpa} kPa < 40 kPa — may help rule out high-risk varices")
        else:
            rationale.append(f"Spleen stiffness {ssm_kpa} kPa ≥ 40 kPa — high-risk varices likely")

    rationale.append("Baveno VII gray zone: HVPG 10-15 mmHg most likely; EGD to confirm variceal status")
    result["rationale"] = rationale
    return result


def predict_masld_advanced_fibrosis(
    age: int,
    ast: float,
    alt: float,
    platelets: float,          # ×10³/μL  e.g. 185
    bmi: float | None = None,
    ggt: float | None = None,
    diabetes: int = 0,         # 1 = Yes, 0 = No
) -> dict:
    """Predict advanced fibrosis (F≥3) risk in MASLD patients with
    indeterminate FIB-4 (1.3–2.67).
    Source: github.com/laithomari/advanced_fibrosis_calculator
    """
    import math

    # Validate indication: compute FIB-4 first
    fib4 = None
    if alt and alt > 0 and platelets and platelets > 0:
        fib4 = (age * ast) / (platelets * math.sqrt(alt))

    indication_warning = None
    if fib4 is not None:
        if fib4 < 1.3:
            indication_warning = f"FIB-4 {round(fib4,2)} < 1.3 (low risk zone). This calculator is validated for FIB-4 1.3–2.67 only."
        elif fib4 > 2.67:
            indication_warning = f"FIB-4 {round(fib4,2)} > 2.67 (high risk zone). This calculator is validated for FIB-4 1.3–2.67 only."

    # Impute missing values with training means (fallback)
    means = _AFC_MODEL["scaler_means"]
    bmi_val     = bmi if bmi is not None else means["bmi"]
    ggt_val     = ggt if ggt is not None else math.exp(means["ggt_log"]) - 1
    platelets_v = platelets * 1000 if platelets < 500 else platelets  # harmonise to /μL

    # Derived features
    ast_log      = math.log1p(ast)
    alt_log      = math.log1p(alt)
    ggt_log      = math.log1p(ggt_val)
    ast_alt_r    = ast / alt if alt > 0 else means["ast_alt_ratio"]

    feature_vals = {
        "age": age, "bmi": bmi_val, "ast_log": ast_log,
        "alt_log": alt_log, "ggt_log": ggt_log,
        "platelets": platelets_v, "diabetes": diabetes,
        "ast_alt_ratio": ast_alt_r,
    }

    # Standardise + compute logit
    logit = _AFC_MODEL["intercept"]
    for feat in _AFC_MODEL["features"]:
        z = (feature_vals[feat] - _AFC_MODEL["scaler_means"][feat]) / _AFC_MODEL["scaler_stds"][feat]
        logit += _AFC_MODEL["coefficients"][feat] * z

    prob = 1 / (1 + math.exp(-logit))
    prob_pct = round(prob * 100, 1)

    th = _AFC_MODEL["thresholds"]
    if prob < th["rule_out"]:
        risk_category = "Low Risk"
        recommendation = "Advanced fibrosis (F≥3) unlikely. Monitor in primary care; reassess FIB-4 in 1–2 years."
        action = "Routine monitoring"
    elif prob >= th["rule_in"]:
        risk_category = "High Risk"
        recommendation = "High probability of advanced fibrosis. Refer to hepatologist. Consider Fibroscan/liver biopsy."
        action = "Hepatology referral + confirmatory testing"
    else:
        risk_category = "Intermediate Risk"
        recommendation = "Indeterminate result. Consider second-line testing: Fibroscan (LSM), ELF score, or liver biopsy."
        action = "Second-line fibrosis testing (Fibroscan/VCTE recommended)"

    result = {
        "probability_advanced_fibrosis": f"{prob_pct}%",
        "probability_raw": round(prob, 4),
        "risk_category": risk_category,
        "recommendation": recommendation,
        "action": action,
        "fib4_score": round(fib4, 2) if fib4 else None,
        "inputs_used": {
            "age": age, "ast": ast, "alt": alt,
            "platelets_per_ul": platelets_v,
            "bmi": round(bmi_val, 1),
            "ggt": round(ggt_val, 1),
            "diabetes": "Yes" if diabetes else "No",
            "ast_alt_ratio": round(ast_alt_r, 2),
        },
        "thresholds": {"rule_out": f"<{th['rule_out']*100:.1f}%", "rule_in": f"≥{th['rule_in']*100:.1f}%"},
        "model_info": "L2-regularized logistic regression | 1,581 biopsy-confirmed MASLD patients | AUROC 0.826",
        "source": "Laithomari et al. github.com/laithomari/advanced_fibrosis_calculator | MIT License",
        "indication": "MASLD/NAFLD patients with indeterminate FIB-4 (1.3–2.67)",
        "indication_warning": indication_warning,
    }
    return result


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
            "Detects acute liver failure (King's Criteria), variceal hemorrhage, SBP, HRS, ACLF Grade 3, "
            "severe alcoholic hepatitis (MDF≥32), Budd-Chiari, HCC rupture, AFLP, Wilson's fulminant, "
            "and critical lab values. "
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
    {
        "name": "calculate_advanced_scores",
        "description": (
            "Calculate advanced hepatology scoring systems beyond Child-Pugh/MELD/ALBI. "
            "Includes: FIB-4 (non-invasive fibrosis), APRI (AST-platelet ratio), "
            "Maddrey Discriminant Function (severe alcoholic hepatitis), "
            "GAHS (Glasgow Alcoholic Hepatitis Score), ABIC score (alcoholic hepatitis mortality), "
            "Lille score (steroid response in alcoholic hepatitis at Day 7), "
            "MELD 3.0 (current UNOS transplant allocation score, includes sex and albumin), "
            "PAGE-B (HCC risk in HBV patients on antiviral therapy). "
            "Provide only the values you have — scores with insufficient data are skipped with explanation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "Patient age in years"},
                "ast": {"type": "number", "description": "AST in U/L"},
                "alt": {"type": "number", "description": "ALT in U/L"},
                "platelets": {"type": "number", "description": "Platelets ×10³/μL"},
                "pt_patient": {"type": "number", "description": "Prothrombin time (patient) in seconds"},
                "pt_control": {"type": "number", "description": "Prothrombin time (control/normal) in seconds (default 12)"},
                "wbc": {"type": "number", "description": "WBC ×10⁹/L (for GAHS)"},
                "urea": {"type": "number", "description": "Urea in mmol/L (for GAHS)"},
                "bilirubin_mgdl": {"type": "number", "description": "Total bilirubin in mg/dL"},
                "bilirubin_umol": {"type": "number", "description": "Total bilirubin in µmol/L (for GAHS/Lille)"},
                "creatinine": {"type": "number", "description": "Creatinine in mg/dL"},
                "inr": {"type": "number", "description": "INR"},
                "albumin": {"type": "number", "description": "Albumin in g/dL"},
                "sodium": {"type": "number", "description": "Serum sodium in mEq/L (for MELD 3.0)"},
                "sex": {"type": "string", "description": "Patient sex: 'M' or 'F' (for MELD 3.0 and PAGE-B)"},
                "albumin_day0_gl": {"type": "number", "description": "Albumin at Day 0 in g/L (for Lille score — 1 g/dL = 10 g/L)"},
                "bilirubin_day0_umol": {"type": "number", "description": "Bilirubin at Day 0 in µmol/L (for Lille score)"},
                "bilirubin_day7_umol": {"type": "number", "description": "Bilirubin at Day 7 in µmol/L (for Lille score)"},
                "renal_failure": {"type": "integer", "description": "Renal failure 0=no 1=yes (Cr >1.3 mg/dL) for Lille score", "enum": [0, 1]},
            },
        },
    },
    {
        "name": "calculate_additional_clinical_scores",
        "description": (
            "Calculate additional validated clinical scores using the medcalc library. "
            "Includes: (1) eGFR CKD-EPI with TDF/antiviral dose adjustment recommendations — "
            "critical for hepatorenal syndrome staging and safe antiviral dosing; "
            "(2) FIB-4 index (medcalc validated version with age-adjusted cutoffs); "
            "(3) HAS-BLED score for bleeding risk when anticoagulation is being considered "
            "(Budd-Chiari, portal vein thrombosis, AF in cirrhosis); "
            "(4) SOFA score for ICU/ACLF severity. "
            "Call this when renal function, anticoagulation decision, or ICU severity assessment is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "creatinine":   {"type": "number",  "description": "Creatinine mg/dL (for eGFR)"},
                "age":          {"type": "integer", "description": "Age in years"},
                "sex":          {"type": "string",  "description": "Sex: M or F"},
                "ast":          {"type": "number",  "description": "AST U/L (for FIB-4)"},
                "alt":          {"type": "number",  "description": "ALT U/L (for FIB-4)"},
                "platelets_fib4": {"type": "number", "description": "Platelets ×10³/μL (for FIB-4)"},
                "hypertension": {"type": "boolean", "description": "Hypertension (for HAS-BLED)"},
                "renal_disease":{"type": "boolean", "description": "Renal disease (for HAS-BLED)"},
                "liver_disease_hasbled": {"type": "boolean", "description": "Liver disease (for HAS-BLED)"},
                "prior_bleeding":{"type": "boolean", "description": "Prior bleeding history (for HAS-BLED)"},
                "stroke_history":{"type": "boolean", "description": "Stroke history (for HAS-BLED)"},
                "elderly_hasbled":{"type": "boolean", "description": "Age >65 (for HAS-BLED)"},
                "antiplatelet": {"type": "boolean", "description": "Antiplatelet use (for HAS-BLED)"},
                "alcohol_hasbled":{"type": "boolean","description": "Alcohol use ≥8 drinks/week (for HAS-BLED)"},
            },
        },
    },
    {
        "name": "calculate_amap_hcc_risk",
        "description": (
            "Calculate aMAP score for annual HCC risk in cirrhotic patients. "
            "UNLIKE PAGE-B (HBV only), aMAP works for ALL cirrhosis etiologies: HBV, HCV, NAFLD, ALD. "
            "Requires confirmed cirrhosis or advanced fibrosis. "
            "Inputs: age, sex, AFP, albumin, platelets, bilirubin. "
            "Outputs: aMAP score 0-100, risk category (Low/Medium/High), "
            "annual HCC incidence estimate, and HCC surveillance frequency recommendation. "
            "Low (<50): ~0.4%/year. Medium (50-60): ~3.3%/year. High (≥60): ~18%/year. "
            "Source: Fan et al., J Hepatol 2020;73:1368-1378."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "age":       {"type": "integer", "description": "Age in years"},
                "sex":       {"type": "string",  "description": "Sex: 'M' or 'F'"},
                "afp":       {"type": "number",  "description": "AFP in ng/mL (default 5 if unknown)"},
                "albumin":   {"type": "number",  "description": "Albumin in g/dL"},
                "platelets": {"type": "number",  "description": "Platelets ×10³/μL"},
                "bilirubin": {"type": "number",  "description": "Total bilirubin in mg/dL"},
            },
            "required": ["age"],
        },
    },
    {
        "name": "assess_baveno_csph",
        "description": (
            "Apply Baveno VII criteria to assess Clinically Significant Portal Hypertension (CSPH) "
            "and determine whether EGD (endoscopy) for variceal screening can be safely avoided. "
            "Key criteria: LSM < 15 kPa AND Platelets > 150k → CSPH ruled out → skip EGD. "
            "LSM ≥ 25 kPa → CSPH confirmed → EGD required. "
            "Lower threshold (LSM < 12 kPa) applies for patients with viral suppression (HCV SVR or HBV). "
            "Also provides NSBB prophylaxis recommendations. "
            "Call this whenever a patient has Fibroscan (LSM) results and cirrhosis is suspected. "
            "Source: Baveno VII Consensus, de Franchis R et al. J Hepatol 2022;76:959-974."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lsm_kpa":            {"type": "number",  "description": "Liver stiffness (Fibroscan LSE) in kPa"},
                "platelets":          {"type": "number",  "description": "Platelets ×10³/μL"},
                "ssm_kpa":            {"type": "number",  "description": "Spleen stiffness in kPa (optional, if available)"},
                "viral_suppression":  {"type": "boolean", "description": "True if HCV SVR achieved or HBV virally suppressed on treatment"},
                "clinical_context":   {"type": "string",  "description": "Brief clinical context (optional)"},
            },
            "required": ["lsm_kpa"],
        },
    },
    {
        "name": "predict_masld_advanced_fibrosis",
        "description": (
            "Predict advanced fibrosis (F≥3) probability in MASLD/NAFLD patients using a validated "
            "L2-regularized logistic regression model (AUROC 0.826). "
            "IMPORTANT: This tool is specifically for patients with INDETERMINATE FIB-4 scores (1.3–2.67) "
            "where standard FIB-4 alone is insufficient. "
            "Inputs: age, AST, ALT, platelets. Optional: BMI, GGT, diabetes status. "
            "Outputs: probability of advanced fibrosis, risk category (Low/Intermediate/High), "
            "and specific clinical recommendation. "
            "Source: github.com/laithomari/advanced_fibrosis_calculator (MIT License, validated on 1,581 biopsies)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "age":      {"type": "integer", "description": "Patient age in years"},
                "ast":      {"type": "number",  "description": "AST in U/L"},
                "alt":      {"type": "number",  "description": "ALT in U/L"},
                "platelets":{"type": "number",  "description": "Platelet count ×10³/μL (e.g. 185)"},
                "bmi":      {"type": "number",  "description": "BMI in kg/m² (optional; uses cohort mean if missing)"},
                "ggt":      {"type": "number",  "description": "GGT in U/L (optional)"},
                "diabetes": {"type": "integer", "description": "Diabetes: 1=Yes, 0=No", "enum": [0, 1]},
            },
            "required": ["age", "ast", "alt", "platelets"],
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
        "calculate_advanced_scores": lambda i: calculate_advanced_scores(**i),
        "calculate_additional_clinical_scores": lambda i: calculate_additional_clinical_scores(**i),
        "calculate_amap_hcc_risk": lambda i: calculate_amap_hcc_risk(**i),
        "assess_baveno_csph": lambda i: assess_baveno_csph(**i),
        "predict_masld_advanced_fibrosis": lambda i: predict_masld_advanced_fibrosis(**i),
    }
    fn = tool_map.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn(tool_input)
    except Exception as e:
        return {"error": str(e)}
