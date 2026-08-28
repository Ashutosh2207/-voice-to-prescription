from typing import List, Dict, Any
from app.models.schemas import Medicine, Prescription


class PrescriptionTransformer:
    """Transform internal prescription format to target Medhant Lite format."""

    # ------------------------------------------------------------------
    # Frequency mapping
    # ------------------------------------------------------------------

    FREQUENCY_MAP = {
        "1-0-1": "1-0-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",
        "1-1-1": "1-1-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",
        "1-0-0": "1-0-0-0 / MORNING/AFTERNOON/EVENING/NIGHT",
        "0-1-0": "0-1-0-0 / MORNING/AFTERNOON/EVENING/NIGHT",
        "0-0-1": "0-0-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",
        "SOS": "SOS / AS NEEDED",
        "PRN": "SOS / AS NEEDED",
    }

    TEXT_FREQUENCY_MAP = {
        "ONCE DAILY":
            "1-0-0-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "ONCE A DAY":
            "1-0-0-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "ONCE":
            "1-0-0-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "TWICE DAILY":
            "1-0-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "TWICE A DAY":
            "1-0-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "TWICE":
            "1-0-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "THRICE DAILY":
            "1-1-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "THREE TIMES A DAY":
            "1-1-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "THREE TIMES DAILY":
            "1-1-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "THRICE":
            "1-1-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",

        "FOUR TIMES A DAY":
            "1-1-1-1 / MORNING/AFTERNOON/EVENING/NIGHT",

        "FOUR TIMES DAILY":
            "1-1-1-1 / MORNING/AFTERNOON/EVENING/NIGHT",

        "FOUR TIMES":
            "1-1-1-1 / MORNING/AFTERNOON/EVENING/NIGHT",
    }

    # ------------------------------------------------------------------
    # Medicine form detection
    # ------------------------------------------------------------------

    FORM_DETECTION_KEYWORDS = {
        "CAPSULE": ["capsule", " cap "],
        "TABLET": ["tablet", " tab ", " pill "],
        "SYRUP": ["syrup", "suspension"],
        "INJECTION": ["injection", "inject ", " iv ", " im "],
        "CREAM": ["cream", "ointment"],
        "DROP": ["drop", "eye drop", "ear drop"],
        "INHALER": ["inhaler", "puff"],
        "PATCH": ["patch", "transdermal"],
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Form detection
    # ------------------------------------------------------------------

    def _detect_form(
        self,
        medicine_name: str,
        raw_transcript: str = "",
        all_medicine_names: list = None
    ) -> str:
        """Detect medicine form from transcript context around medicine name."""

        if not raw_transcript:
            return "TABLET"

        name_lower = medicine_name.lower()
        transcript_lower = raw_transcript.lower()

        med_index = transcript_lower.find(name_lower)

        if med_index == -1:
            return "TABLET"

        # Find the next medicine so context does not overlap
        next_med_index = len(transcript_lower)

        if all_medicine_names:

            for other_name in all_medicine_names:

                if other_name.lower() == name_lower:
                    continue

                other_idx = transcript_lower.find(
                    other_name.lower(),
                    med_index + len(name_lower)
                )

                if other_idx != -1 and other_idx < next_med_index:
                    next_med_index = other_idx

        context_start = max(0, med_index - 30)

        context_end = min(
            next_med_index,
            med_index + len(name_lower) + 80
        )

        context = transcript_lower[
            context_start:context_end
        ]

        for form, keywords in self.FORM_DETECTION_KEYWORDS.items():

            for keyword in keywords:

                if keyword in context:
                    return form

        return "TABLET"

    # ------------------------------------------------------------------
    # Frequency parsing
    # ------------------------------------------------------------------

    def _parse_frequency(self, freq: str) -> str:
        """
        Normalize medicine frequency to exactly four time slots.

        Examples:
            1-0-1     -> 1-0-1-0
            1-1-1     -> 1-1-1-0
            1-0-1-0   -> 1-0-1-0
            1-1-1-1   -> 1-1-1-1
            SOS       -> SOS / AS NEEDED
        """

        if not freq:
            return ""

        freq = str(freq).strip().upper()

        # Remove any existing description after " / "
        # Example:
        # 1-0-1-0 / MORNING/... -> 1-0-1-0
        if " / " in freq:
            freq = freq.split(" / ", 1)[0].strip()

        # SOS / PRN
        if freq in {"SOS", "PRN"}:
            return "SOS / AS NEEDED"

        # Already known textual frequency
        if freq in self.TEXT_FREQUENCY_MAP:
            return self.TEXT_FREQUENCY_MAP[freq]

        # Known numeric frequency map
        if freq in self.FREQUENCY_MAP:
            return self.FREQUENCY_MAP[freq]

        # --------------------------------------------------------------
        # Numeric frequency normalization
        # --------------------------------------------------------------

        parts = [part.strip() for part in freq.split("-")]

        # Only allow 0 / 1 frequency slots
        if parts and all(part in {"0", "1"} for part in parts):

            if len(parts) == 1:
                parts.extend(["0", "0", "0"])

            elif len(parts) == 2:
                parts.extend(["0", "0"])

            elif len(parts) == 3:
                parts.append("0")

            elif len(parts) > 4:
                parts = parts[:4]

            normalized = "-".join(parts)

            return (
                f"{normalized} / "
                "MORNING/AFTERNOON/EVENING/NIGHT"
            )

        # If frequency is something unexpected,
        # return it unchanged instead of corrupting it.
        return freq

    # ------------------------------------------------------------------
    # Duration parsing
    # ------------------------------------------------------------------

    def _parse_duration(self, duration: str) -> str:
        """Normalize duration format."""

        if not duration:
            return ""

        duration = str(duration).strip().upper()

        return duration

    # ------------------------------------------------------------------
    # Unit per time
    # ------------------------------------------------------------------

    def _get_unit_per_time(self, freq: str) -> str:
        """
        Extract unit per time from frequency.

        Example:
            1-0-1 -> 1
            2-0-2 -> 2
        """

        if not freq:
            return "1"

        freq = str(freq).strip()

        # Remove target description if present
        if " / " in freq:
            freq = freq.split(" / ", 1)[0].strip()

        parts = freq.split("-")

        if parts and parts[0].isdigit():
            return parts[0]

        return "1"

    # ------------------------------------------------------------------
    # Transform one medicine
    # ------------------------------------------------------------------

    def transform_medicine(
        self,
        medicine: Medicine,
        raw_transcript: str = "",
        all_medicine_names: list = None
    ) -> Dict[str, Any]:
        """Transform a single medicine to target Medhant Lite format."""

        form = self._detect_form(
            medicine.medicineName,
            raw_transcript,
            all_medicine_names
        )

        # Dosage
        dosage = medicine.dosage or ""
        dosage = dosage.upper().replace(" ", "")

        # Frequency
        frequency = self._parse_frequency(
            medicine.frequency or ""
        )

        # Duration
        duration = self._parse_duration(
            medicine.duration or ""
        )

        # Instructions / remarks
        remarks = (
            medicine.instructions or ""
        ).upper().strip()

        # Unit per time
        unit_per_time = self._get_unit_per_time(
            medicine.frequency or ""
        )

        return {
            "name": medicine.medicineName.upper(),
            "form": form,
            "dosage": dosage,
            "numberOfTime": frequency,
            "remarks": remarks,
            "timesPerDay": duration,
            "unitPerTime": unit_per_time,
        }

    # ------------------------------------------------------------------
    # Transform complete prescription
    # ------------------------------------------------------------------

    def transform(
        self,
        prescription: Prescription
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Transform complete prescription to target format."""

        raw_transcript = prescription.rawTranscript or ""

        medicine_names = [
            med.medicineName
            for med in prescription.medicines
        ]

        medicines = []

        for med in prescription.medicines:

            transformed = self.transform_medicine(
                med,
                raw_transcript,
                medicine_names
            )

            medicines.append(transformed)

        return {
            "prescription": medicines
        }


# ----------------------------------------------------------------------
# Singleton transformer instance
# ----------------------------------------------------------------------

_transformer_instance = None


def get_transformer() -> PrescriptionTransformer:
    """Return a shared PrescriptionTransformer instance."""

    global _transformer_instance

    if _transformer_instance is None:
        _transformer_instance = PrescriptionTransformer()

    return _transformer_instance