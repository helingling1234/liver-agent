"""Pydantic models for structured data in the liver disease agent."""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class LabValue(BaseModel):
    name: str
    value: float | str
    unit: str = ""
    reference_range: str = ""
    status: str = "normal"  # normal, high, low, critical_high, critical_low
    interpretation: str = ""


class ParsedLabResults(BaseModel):
    values: list[LabValue] = []
    abnormal_count: int = 0
    critical_count: int = 0
    summary: str = ""


class ChildPughScore(BaseModel):
    total_score: int
    grade: str  # A, B, C
    bilirubin_points: int
    albumin_points: int
    inr_points: int
    ascites_points: int
    encephalopathy_points: int
    one_year_survival: str
    two_year_survival: str
    interpretation: str


class MeldScore(BaseModel):
    score: int
    category: str
    three_month_mortality: str
    interpretation: str


class AlbiScore(BaseModel):
    score: float
    grade: int  # 1, 2, 3
    interpretation: str


class SeverityScores(BaseModel):
    child_pugh: ChildPughScore | None = None
    meld: MeldScore | None = None
    albi: AlbiScore | None = None
    missing_values: list[str] = []
    recommendations: list[str] = []


class Diagnosis(BaseModel):
    rank: int
    condition: str
    probability: str  # high, moderate, low
    supporting_evidence: list[str] = []
    against_evidence: list[str] = []
    next_steps: list[str] = []


class DifferentialDiagnosis(BaseModel):
    primary_diagnosis: str
    differentials: list[Diagnosis] = []
    key_distinguishing_features: list[str] = []
    urgent_considerations: list[str] = []


class FibrosisAssessment(BaseModel):
    stage: str  # F0, F1, F2, F3, F4
    description: str
    confidence: str  # high, moderate, low
    supporting_evidence: list[str] = []
    recommended_confirmatory_tests: list[str] = []
    clinical_significance: str


class TreatmentGuideline(BaseModel):
    diagnosis: str
    first_line_treatment: list[str] = []
    second_line_treatment: list[str] = []
    contraindications: list[str] = []
    monitoring_parameters: list[str] = []
    treatment_goals: list[str] = []
    special_considerations: list[str] = []
    guideline_source: str = ""


class ClinicalSummary(BaseModel):
    chief_complaint: str = ""
    clinical_presentation: str = ""
    key_findings: list[str] = []
    assessment: str = ""
    differential_diagnoses: list[str] = []
    severity_assessment: str = ""
    plan: list[str] = []
    follow_up: list[str] = []
    urgent_actions: list[str] = []
    patient_education: list[str] = []
    mode: str = "physician"  # physician or patient


class UrgentFinding(BaseModel):
    finding: str
    urgency_level: str  # CRITICAL, URGENT, ROUTINE
    recommended_action: str
    timeframe: str
    rationale: str


class UrgentFindings(BaseModel):
    has_critical: bool = False
    has_urgent: bool = False
    findings: list[UrgentFinding] = []
    overall_recommendation: str = ""
    emergency_actions: list[str] = []
