from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class FrequencyPattern(str, Enum):
    ONCE_DAILY = "1-0-0"
    TWICE_DAILY = "1-0-1"
    THRICE_DAILY = "1-1-1"
    FOUR_TIMES_DAILY = "1-1-1-1"
    AS_NEEDED = "SOS"
    CUSTOM = "custom"


class Route(str, Enum):
    ORAL = "oral"
    TOPICAL = "topical"
    INJECTION = "injection"
    INHALATION = "inhalation"
    SUBLINGUAL = "sublingual"
    RECTAL = "rectal"
    OPHTHALMIC = "ophthalmic"
    OTIC = "otic"
    NASAL = "nasal"
    TRANSDERMAL = "transdermal"
    UNKNOWN = "unknown"


class Medicine(BaseModel):
    medicineName: str = Field(..., description="Name of the medicine")
    dosage: Optional[str] = Field(None, description="Dosage/strength (e.g., '500 mg', '10 ml')")
    frequency: Optional[str] = Field(None, description="Frequency (e.g., '1-0-1', 'SOS', 'BD')")
    duration: Optional[str] = Field(None, description="Duration (e.g., '5 days', '2 weeks')")
    route: Optional[Route] = Field(Route.UNKNOWN, description="Route of administration")
    instructions: Optional[str] = Field(None, description="Special instructions (e.g., 'After food')")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Extraction confidence score")
    
    @field_validator('frequency', mode='before')
    @classmethod
    def normalize_frequency(cls, v):
        if v is None:
            return v
        freq_map = {
            'od': '1-0-0', 'once daily': '1-0-0', 'once a day': '1-0-0',
            'bd': '1-0-1', 'twice daily': '1-0-1', 'twice a day': '1-0-1',
            'tid': '1-1-1', 'thrice daily': '1-1-1', 'three times': '1-1-1',
            'qid': '1-1-1-1', 'four times': '1-1-1-1',
            'sos': 'SOS', 'as needed': 'SOS', 'prn': 'SOS',
        }
        return freq_map.get(v.lower().strip(), v)


class Prescription(BaseModel):
    medicines: List[Medicine] = Field(default_factory=list, description="List of prescribed medicines")
    notes: Optional[str] = Field(None, description="Additional prescription notes")
    rawTranscript: Optional[str] = Field(None, description="Original speech-to-text transcript")


class ValidationIssue(BaseModel):
    field: str
    medicineIndex: int
    issue: str
    severity: Literal["error", "warning", "info"]


class ValidationResult(BaseModel):
    isValid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    missingFields: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    prescription: Prescription
    validation: ValidationResult
    processingTimeMs: float
    sttConfidence: Optional[float] = None


class PrescriptionResponse(BaseModel):
    success: bool
    data: Optional[ExtractionResult] = None
    error: Optional[str] = None


class AudioProcessRequest(BaseModel):
    audioBase64: Optional[str] = None
    audioUrl: Optional[str] = None
    language: str = "en"
    returnTranscript: bool = True