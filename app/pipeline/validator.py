from typing import List, Dict, Any
from dataclasses import dataclass

from app.models.schemas import Medicine, Prescription, ValidationResult, ValidationIssue, Route


@dataclass
class ValidationConfig:
    required_fields: List[str] = None
    
    def __post_init__(self):
        if self.required_fields is None:
            self.required_fields = ["medicineName"]


class PrescriptionValidator:
    """Validate extracted prescription data."""
    
    # Known medicine database (simplified - in production, use a proper drug database)
    KNOWN_MEDICINES = {
        "paracetamol", "ibuprofen", "amoxicillin", "metformin", "atorvastatin",
        "omeprazole", "levothyroxine", "amlodipine", "losartan", "metoprolol",
        "aspirin", "diclofenac", "naproxen", "cetirizine", "loratadine",
        "pantoprazole", "ranitidine", "domperidone", "ondansetron", "diazepam",
        "alprazolam", "clonazepam", "zolpidem", "sertraline", "fluoxetine",
        "escitalopram", "venlafaxine", "duloxetine", "gabapentin", "pregabalin",
        "insulin", "glipizide", "glimepiride", "pioglitazone", "sitagliptin",
        "lisinopril", "enalapril", "ramipril", "valsartan", "irbesartan",
        "hydrochlorothiazide", "furosemide", "spironolactone", "warfarin",
        "rivaroxaban", "apixaban", "dabigatran", "clopidogrel", "ticagrelor",
    }
    
    # Valid dosage units
    VALID_UNITS = {"mg", "g", "mcg", "µg", "ml", "cc", "units", "iu", "tablet", "tab", "capsule", "cap", "pill"}
    
    # Valid frequency patterns
    VALID_FREQUENCIES = {
        "1-0-0", "1-0-1", "1-1-1", "1-1-1-1", "SOS",
        "OD", "BD", "TID", "QID", "PRN"
    }
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
    
    def validate(self, prescription: Prescription) -> ValidationResult:
        """Validate a prescription and return issues."""
        issues = []
        missing_fields = []
        
        if not prescription.medicines:
            issues.append(ValidationIssue(
                field="medicines",
                medicineIndex=-1,
                issue="No medicines found in prescription",
                severity="error"
            ))
            return ValidationResult(isValid=False, issues=issues, missing_fields=["medicines"])
        
        for idx, med in enumerate(prescription.medicines):
            med_issues, med_missing = self._validate_medicine(med, idx)
            issues.extend(med_issues)
            missing_fields.extend(med_missing)
        
        # Check for duplicate medicines
        med_names = [m.medicineName.lower() for m in prescription.medicines]
        seen = set()
        for idx, name in enumerate(med_names):
            if name in seen:
                issues.append(ValidationIssue(
                    field="medicineName",
                    medicineIndex=idx,
                    issue=f"Duplicate medicine: {name}",
                    severity="warning"
                ))
            seen.add(name)
        
        is_valid = not any(i.severity == "error" for i in issues)
        
        return ValidationResult(
            isValid=is_valid,
            issues=issues,
            missingFields=list(set(missing_fields))
        )
    
    def _validate_medicine(self, med: Medicine, idx: int) -> tuple[List[ValidationIssue], List[str]]:
        issues = []
        missing = []
        
        # Required field: medicineName
        if not med.medicineName or not med.medicineName.strip():
            issues.append(ValidationIssue(
                field="medicineName",
                medicineIndex=idx,
                issue="Medicine name is required",
                severity="error"
            ))
            missing.append("medicineName")
        else:
            # Check if medicine is known
            if med.medicineName.lower() not in self.KNOWN_MEDICINES:
                issues.append(ValidationIssue(
                    field="medicineName",
                    medicineIndex=idx,
                    issue=f"Unknown medicine: {med.medicineName}. Please verify.",
                    severity="warning"
                ))
        
        # Validate dosage
        if med.dosage:
            dosage_issues = self._validate_dosage(med.dosage, idx)
            issues.extend(dosage_issues)
        else:
            missing.append("dosage")
            issues.append(ValidationIssue(
                field="dosage",
                medicineIndex=idx,
                issue="Dosage not specified",
                severity="warning"
            ))
        
        # Validate frequency
        if med.frequency:
            freq_issues = self._validate_frequency(med.frequency, idx)
            issues.extend(freq_issues)
        else:
            missing.append("frequency")
            issues.append(ValidationIssue(
                field="frequency",
                medicineIndex=idx,
                issue="Frequency not specified",
                severity="warning"
            ))
        
        # Validate duration
        if med.duration:
            dur_issues = self._validate_duration(med.duration, idx)
            issues.extend(dur_issues)
        else:
            missing.append("duration")
            issues.append(ValidationIssue(
                field="duration",
                medicineIndex=idx,
                issue="Duration not specified",
                severity="info"
            ))
        
        # Validate route
        if med.route and med.route != Route.UNKNOWN:
            pass  # Enum validation handled by Pydantic
        else:
            missing.append("route")
            issues.append(ValidationIssue(
                field="route",
                medicineIndex=idx,
                issue="Route not specified, defaulting to oral",
                severity="info"
            ))
        
        return issues, missing
    
    def _validate_dosage(self, dosage: str, idx: int) -> List[ValidationIssue]:
        issues = []
        dosage_lower = dosage.lower().strip()
        
        # Check for unit
        has_unit = any(unit in dosage_lower for unit in self.VALID_UNITS)
        if not has_unit:
            issues.append(ValidationIssue(
                field="dosage",
                medicineIndex=idx,
                issue=f"Dosage may be missing unit: {dosage}",
                severity="warning"
            ))
        
        # Check for numeric value
        import re
        if not re.search(r'\d', dosage):
            issues.append(ValidationIssue(
                field="dosage",
                medicineIndex=idx,
                issue=f"Dosage appears to lack numeric value: {dosage}",
                severity="warning"
            ))
        
        return issues
    
    def _validate_frequency(self, frequency: str, idx: int) -> List[ValidationIssue]:
        issues = []
        freq_upper = frequency.upper().strip()
        
        if freq_upper not in self.VALID_FREQUENCIES:
            issues.append(ValidationIssue(
                field="frequency",
                medicineIndex=idx,
                issue=f"Non-standard frequency format: {frequency}. Expected: 1-0-0, 1-0-1, 1-1-1, 1-1-1-1, SOS",
                severity="warning"
            ))
        
        return issues
    
    def _validate_duration(self, duration: str, idx: int) -> List[ValidationIssue]:
        issues = []
        duration_lower = duration.lower().strip()
        
        import re
        if not re.search(r'\d+\s*(days?|weeks?|months?)', duration_lower):
            issues.append(ValidationIssue(
                field="duration",
                medicineIndex=idx,
                issue=f"Duration format unclear: {duration}. Expected: '5 days', '2 weeks', etc.",
                severity="warning"
            ))
        
        return issues


# Singleton
_validator_instance = None


def get_validator(config: ValidationConfig = None) -> PrescriptionValidator:
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = PrescriptionValidator(config)
    return _validator_instance