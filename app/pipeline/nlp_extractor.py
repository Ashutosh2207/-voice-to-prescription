import re
from typing import List, Optional
from dataclasses import dataclass

from app.models.schemas import Route
from app.pipeline.normalizer import get_normalizer


@dataclass
class ExtractedMedicine:
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[Route] = None
    instructions: Optional[str] = None
    confidence: float = 0.0


class PrescriptionExtractor:
    """Extract prescription medicines using deterministic rules."""

    MEDICINE_NAMES = {
        "paracetamol",
        "ibuprofen",
        "amoxicillin",
        "metformin",
        "atorvastatin",
        "omeprazole",
        "levothyroxine",
        "amlodipine",
        "losartan",
        "metoprolol",
        "aspirin",
        "diclofenac",
        "naproxen",
        "cetirizine",
        "loratadine",
        "pantoprazole",
        "ranitidine",
        "domperidone",
        "ondansetron",
        "diazepam",
        "alprazolam",
        "clonazepam",
        "zolpidem",
        "sertraline",
        "fluoxetine",
        "escitalopram",
        "venlafaxine",
        "duloxetine",
        "gabapentin",
        "pregabalin",
        "insulin",
        "glipizide",
        "glimepiride",
        "pioglitazone",
        "sitagliptin",
        "lisinopril",
        "enalapril",
        "ramipril",
        "valsartan",
        "irbesartan",
        "hydrochlorothiazide",
        "furosemide",
        "spironolactone",
        "warfarin",
        "rivaroxaban",
        "apixaban",
        "dabigatran",
        "clopidogrel",
        "ticagrelor",
        "azithromycin",
        "ciprofloxacin",
        "doxycycline",
        "clarithromycin",
        "prednisolone",
        "dexamethasone",
        "montelukast",
        "salbutamol",
        "budesonide",
        "fluticasone",
        "esomeprazole",
        "lansoprazole",
        "rabeprazole",
        "famotidine",
        "cimetidine",
        "metoclopramide",
        "hyoscine",
        "dicyclomine",
        "mefenamic acid",
        "tranexamic acid",
        "calcium",
        "vitamin d3",
        "vitamin b12",
        "folic acid",
        "iron",
        "zinc",
        "multivitamin",
        "probiotic",
        "ors",
        "electral",
    }

    def __init__(self):
        self.normalizer = get_normalizer()
        self._compile_patterns()

    def _compile_patterns(self):

        med_pattern = "|".join(
            re.escape(m)
            for m in sorted(
                self.MEDICINE_NAMES,
                key=len,
                reverse=True
            )
        )

        self.med_regex = re.compile(
            rf"\b({med_pattern})\b",
            re.IGNORECASE
        )

        self.dosage_regex = re.compile(
            r"(\d+(?:\.\d+)?)\s*"
            r"(mg|g|mcg|µg|ml|cc|units?|iu|"
            r"tablet|tab|capsule|cap|pill|puffs|puff|drop)",
            re.IGNORECASE
        )

        self.frequency_patterns = [

            (
                re.compile(
                    r"\btwice\s+(?:daily|a day|per day)\b",
                    re.I
                ),
                "1-0-1"
            ),

            (
                re.compile(
                    r"\b2\s+times?\s+(?:a|per)\s+day\b",
                    re.I
                ),
                "1-0-1"
            ),

            (
                re.compile(
                    r"\bthrice\s+(?:daily|a day|per day)\b",
                    re.I
                ),
                "1-1-1"
            ),

            (
                re.compile(
                    r"\b3\s+times?\s+(?:a|per)\s+day\b",
                    re.I
                ),
                "1-1-1"
            ),

            (
                re.compile(
                    r"\bthree\s+times?\s+(?:a|per)\s+day\b",
                    re.I
                ),
                "1-1-1"
            ),

            (
                re.compile(
                    r"\bonce\s+(?:daily|a day|per day)\b",
                    re.I
                ),
                "1-0-0"
            ),

            (
                re.compile(
                    r"\bSOS\b",
                    re.I
                ),
                "SOS"
            ),

            (
                re.compile(
                    r"\bPRN\b",
                    re.I
                ),
                "SOS"
            ),

            (
                re.compile(
                    r"\bBD\b",
                    re.I
                ),
                "1-0-1"
            ),

            (
                re.compile(
                    r"\bTID\b",
                    re.I
                ),
                "1-1-1"
            ),

            (
                re.compile(
                    r"\bQID\b",
                    re.I
                ),
                "1-1-1-1"
            ),

            (
                re.compile(
                    r"\b(\d-\d-\d(?:-\d)?)\b",
                    re.I
                ),
                None
            ),

            (
                re.compile(
                    r"\bone\s+(?:tablet|cap|capsule|pill)\s+(?:in\s+the\s+)?morning\s+(?:and|&)\s+one\s+(?:tablet|cap|capsule|pill)?\s+(?:in\s+the\s+)?evening\b",
                    re.I
                ),
                "1-0-1"
            ),

            (
                re.compile(
                    r"\bone\s+(?:tablet|cap|capsule|pill)\s+(?:in\s+the\s+)?morning\s+(?:and|&)\s+one\s+(?:tablet|cap|capsule|pill)?\s+(?:in\s+the\s+)?afternoon\s+(?:and|&)\s+one\s+(?:tablet|cap|capsule|pill)?\s+(?:in\s+the\s+)?evening\b",
                    re.I
                ),
                "1-1-1"
            ),

            (
                re.compile(
                    r"\b(?:morning|am)\s+(?:and|&)\s+(?:evening|pm|night)\b",
                    re.I
                ),
                "1-0-1"
            ),

            (
                re.compile(
                    r"\b(?:morning|am)\s+(?:and|&)\s+(?:afternoon|noon)\s+(?:and|&)\s+(?:evening|pm|night)\b",
                    re.I
                ),
                "1-1-1"
            ),

            (
                re.compile(
                    r"\bone\s+time\s+(?:in\s+the\s+)?morning\s+(?:and|&)\s+one\s+time\s+(?:in\s+the\s+)?evening\b",
                    re.I
                ),
                "1-0-1"
            ),
        ]

        self.duration_regex = re.compile(
            r"(?:for\s+)?"
            r"(\d+)\s*"
            r"(days?|weeks?|months?)",
            re.IGNORECASE
        )

        self.route_patterns = [
            (
                re.compile(
                    r"\b(?:oral|by mouth|po|per os)\b",
                    re.I
                ),
                Route.ORAL
            ),
            (
                re.compile(
                    r"\b(?:topical|apply|cream|ointment|gel)\b",
                    re.I
                ),
                Route.TOPICAL
            ),
            (
                re.compile(
                    r"\b(?:inject|injection|iv|im|"
                    r"subcut|sc|intramuscular|intravenous)\b",
                    re.I
                ),
                Route.INJECTION
            ),
            (
                re.compile(
                    r"\b(?:inhale|inhalation|puff|inhaler)\b",
                    re.I
                ),
                Route.INHALATION
            ),
        ]

        self.instruction_patterns = [
            (
                re.compile(
                    r"\b(?:after|with)\s*"
                    r"(?:food|foods|meal|meals|eating)\b",
                    re.I
                ),
                "After food"
            ),
            (
                re.compile(
                    r"\b(?:before|empty)\s*"
                    r"(?:stomach|meal|food)\b",
                    re.I
                ),
                "Before food"
            ),
            (
                re.compile(
                    r"\b(?:at bedtime|before sleep)\b",
                    re.I
                ),
                "At bedtime"
            ),
        ]

    # ---------------------------------------------------------
    # Main extraction
    # ---------------------------------------------------------

    def extract(
        self,
        text: str
    ) -> List[ExtractedMedicine]:

        if not text or not text.strip():
            return []

        # Normalize STT mistakes
        norm_result = self.normalizer.normalize(text)

        normalized_text = norm_result[
            "normalized_text"
        ]

        print(
            f"[NLP] Original: {text}"
        )

        print(
            f"[NLP] Normalized: {normalized_text}"
        )

        # Find medicines
        med_matches = list(
            self.med_regex.finditer(
                normalized_text
            )
        )

        if not med_matches:
            print(
                "[NLP] No known medicines found."
            )
            return []

        medicines = []

        for index, match in enumerate(med_matches):

            med_name = match.group(1).lower()

            # -------------------------------------------------
            # IMPORTANT:
            # Context ends before next medicine.
            # This prevents medicine 1 from stealing
            # dosage/frequency of medicine 2.
            # -------------------------------------------------

            start_pos = match.end()

            if index + 1 < len(med_matches):

                next_start = med_matches[
                    index + 1
                ].start()

                context_end = next_start

            else:
                context_end = min(
                    len(normalized_text),
                    start_pos + 150
                )

            context = normalized_text[
                start_pos:context_end
            ]

            # -------------------------------------------------
            # Extract fields
            # -------------------------------------------------

            dosage = self._extract_dosage(
                context
            )

            frequency = self._extract_frequency(
                context
            )

            duration = self._extract_duration(
                context
            )

            route = self._extract_route(
                context,
                med_name
            )

            instructions = self._extract_instructions(
                context
            )

            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            fields_found = sum(
                bool(value)
                for value in [
                    dosage,
                    frequency,
                    duration,
                    instructions
                ]
            )

            confidence = min(
                0.70 + fields_found * 0.07,
                0.98
            )

            medicine = ExtractedMedicine(
                name=med_name.capitalize(),
                dosage=dosage,
                frequency=frequency,
                duration=duration,
                route=route or Route.ORAL,
                instructions=instructions,
                confidence=confidence
            )

            medicines.append(
                medicine
            )

        print(
            f"[NLP] Medicines extracted: "
            f"{len(medicines)}"
        )

        return medicines

    # ---------------------------------------------------------
    # Dosage
    # ---------------------------------------------------------

    def _extract_dosage(
        self,
        text: str
    ) -> Optional[str]:

        match = self.dosage_regex.search(
            text
        )

        if match:

            return (
                f"{match.group(1)} "
                f"{match.group(2)}"
            )

        return None

    # ---------------------------------------------------------
    # Frequency
    # ---------------------------------------------------------

    def _extract_frequency(
        self,
        text: str
    ) -> Optional[str]:

        for pattern, code in self.frequency_patterns:

            match = pattern.search(text)

            if match:

                if code is None:
                    return match.group(1)

                return code

        return None

    # ---------------------------------------------------------
    # Duration
    # ---------------------------------------------------------

    def _extract_duration(
        self,
        text: str
    ) -> Optional[str]:

        match = self.duration_regex.search(
            text
        )

        if match:

            return (
                f"{match.group(1)} "
                f"{match.group(2)}"
            )

        return None

    # ---------------------------------------------------------
    # Route
    # ---------------------------------------------------------

    def _extract_route(
        self,
        text: str,
        medicine_name: str = ""
    ) -> Optional[Route]:

        # Check medicine name first (e.g., "insulin" implies injection)
        combined_text = f"{medicine_name} {text}"

        for pattern, route in self.route_patterns:

            if pattern.search(combined_text):

                return route

        # Default oral
        return Route.ORAL

    # ---------------------------------------------------------
    # Instructions
    # ---------------------------------------------------------

    def _extract_instructions(
        self,
        text: str
    ) -> Optional[str]:

        found = []

        for pattern, instruction in self.instruction_patterns:

            if pattern.search(text):

                if instruction not in found:
                    found.append(instruction)

        if found:
            return ", ".join(found)

        return None


# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------

_extractor_instance = None


def get_extractor(
    use_llm: bool = False,
    api_key: str = None,
    model_name: str = "en_core_web_sm"
):

    global _extractor_instance

    if _extractor_instance is None:
        _extractor_instance = PrescriptionExtractor()

    return _extractor_instance