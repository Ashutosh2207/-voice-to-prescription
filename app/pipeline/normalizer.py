import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class NormalizedText:
    text: str
    replacements: List[Tuple[str, str]]


class TextNormalizer:
    """Normalize medical speech-to-text output."""

    # ---------------------------------------------------------
    # Common Whisper / Speech-to-Text medicine errors
    # ---------------------------------------------------------
    MEDICINE_CORRECTIONS = {

        "paracetamol": [
            "parasetamol",
            "paracitamol",
            "para setamol",
            "perasetamol",
            "parasitum",
            "parasitum all",
            "paracitamol",
            "paracetemol",
            "paracetomol",
            "paracetamol"
        ],

        "amoxicillin": [
            "amoxicilin",
            "amoxycillin",
            "amoxi cillin",
            "amoxidum",
            "amoxydum",
            "amoxilum",
            "amoxicilline",
            "amoxicilin"
        ],

        "morning": [
            "mroning",
            "mronin",
            "mornin"
        ],

        "in the morning": [
            "inthemorning",
            "inthe morning",
            "intomorning"
        ],

        "in the evening": [
            "intheevening",
            "inthe evening",
            "intoevening"
        ],

        "ibuprofen": [
            "ibuprophen",
            "ibu profen",
            "eyebuprofen",
            "ibuprofen"
        ],

        "metformin": [
            "met formin",
            "metfo rmin"
        ],

        "atorvastatin": [
            "atorva statin",
            "ator vastatin"
        ],

        "omeprazole": [
            "omeprazal",
            "ome pra zole"
        ],

        "levothyroxine": [
            "levo thyroxine",
            "levothyroxin"
        ],

        "amlodipine": [
            "amlo dipine",
            "amlodi pine"
        ],

        "losartan": [
            "lo sartan",
            "losartan potassium"
        ],

        "metoprolol": [
            "metroprolol",
            "meto prolol"
        ],

        "azithromycin": [
            "azithro mycin",
            "azithramycin"
        ],

        "cetirizine": [
            "cetrizine",
            "cetrazine"
        ],

        "pantoprazole": [
            "pantaprazole",
            "pantoprazol"
        ],

        "diclofenac": [
            "diclofinac",
            "diclofen"
        ],
    }

    # ---------------------------------------------------------
    # Dosage
    # ---------------------------------------------------------
    DOSAGE_PATTERNS = [
        (
            r'(\d+(?:\.\d+)?)\s*'
            r'(mg|g|mcg|µg|ml|cc|units?|iu)',
            r'\1 \2'
        ),
        (
            r'(\d+)\s*'
            r'(tablet|tab|cap|capsule|pill)',
            r'\1 \2'
        ),
        (
            r'(\d+)\s*'
            r'(puffs|puff)',
            r'\1 \2'
        ),
    ]

    # ---------------------------------------------------------
    # Frequency
    # ---------------------------------------------------------
    FREQUENCY_PATTERNS = {
        r'\b(?:once|one time)\s*(?:daily|a day|per day)\b':
            '1-0-0',

        r'\b(?:twice|two times)\s*(?:daily|a day|per day)\b':
            '1-0-1',

        r'\b(?:thrice|three times)\s*(?:daily|a day|per day)\b':
            '1-1-1',

        r'\b(?:four times)\s*(?:daily|a day|per day)\b':
            '1-1-1-1',

        r'\b(?:3|three)\s*times?\s*(?:a|per)\s*day\b':
            '1-1-1',

        r'\b(?:2|two)\s*times?\s*(?:a|per)\s*day\b':
            '1-0-1',

        r'\b(?:1|one)\s*time?\s*(?:a|per)\s*day\b':
            '1-0-0',

        r'\b(?:every|each)\s*(?:morning|am)\b':
            '1-0-0',

        r'\b(?:every|each)\s*(?:evening|pm|night)\b':
            '0-0-1',

        r'\b(?:morning|am)\s*(?:and|&)\s*(?:evening|pm|night)\b':
            '1-0-1',

        r'\b(?:morning|am)\s*(?:and|&)\s*'
        r'(?:afternoon|noon)\s*(?:and|&)\s*'
        r'(?:evening|pm|night)\b':
            '1-1-1',

        r'\bone\s+tablet\s+(?:in\s+the\s+)?morning\s+(?:and|&)\s+one\s+(?:tablet\s+)?(?:in\s+the\s+)?evening\b':
            '1-0-1',

        r'\bone\s+capsule\s+(?:in\s+the\s+)?morning\s+(?:and|&)\s+one\s+(?:capsule\s+)?(?:in\s+the\s+)?evening\b':
            '1-0-1',

        r'\bone\s+tablet\s+(?:in\s+the\s+)?morning\s+(?:and|&)\s+one\s+tablet\s+(?:in\s+the\s+)?afternoon\s+(?:and|&)\s+one\s+tablet\s+(?:in\s+the\s+)?evening\b':
            '1-1-1',

        r'\bone\s+tablet\s+(?:in\s+the\s+)?morning\s*[,]?\s*(?:and|&)?\s*afternoon\s+(?:and|&)\s+evening\b':
            '1-1-1',

        r'\bone\s+time\s+(?:in\s+the\s+)?morning\s+(?:and|&)\s+one\s+time\s+(?:in\s+the\s+)?evening\b':
            '1-0-1',

        r'\b(?:sos|as needed|prn|when needed)\b':
            'SOS',

        r'\b(?:bd|b\.?d\.?|bis in die)\b':
            '1-0-1',

        r'\b(?:tid|t\.?i\.?d\.?|ter in die)\b':
            '1-1-1',

        r'\b(?:qid|q\.?i\.?d\.?|quater in die)\b':
            '1-1-1-1',

        r'\b(?:od|o\.?d\.?|once daily)\b':
            '1-0-0',
    }

    # ---------------------------------------------------------
    # Duration
    # ---------------------------------------------------------
    DURATION_PATTERNS = {
        r'\bfor\s+(\d+)\s*(days?|weeks?|months?)\b':
            r'\1 \2',

        r'\b(\d+)\s*(days?|weeks?|months?)\b':
            r'\1 \2',

        r'\b(\d+)\s*(?:doses?|dose)\b':
            r'\1 doses',
    }

    # ---------------------------------------------------------
    # Route
    # ---------------------------------------------------------
    ROUTE_PATTERNS = {
        r'\b(?:oral|by mouth|po|per os)\b':
            'oral',

        r'\b(?:topical|apply|cream|ointment|gel)\b':
            'topical',

        r'\b(?:inject|injection|iv|im|subcut|sc|'
        r'intramuscular|intravenous)\b':
            'injection',

        r'\b(?:inhale|inhalation|puff|inhaler)\b':
            'inhalation',

        r'\b(?:sublingual|under tongue)\b':
            'sublingual',

        r'\b(?:rectal|suppository)\b':
            'rectal',

        r'\b(?:eye|ophthalmic|eye drop)\b':
            'ophthalmic',

        r'\b(?:ear|otic|ear drop)\b':
            'otic',

        r'\b(?:nasal|nose spray|nose drop)\b':
            'nasal',

        r'\b(?:patch|transdermal)\b':
            'transdermal',
    }

    # ---------------------------------------------------------
    # Instructions
    # ---------------------------------------------------------
    INSTRUCTION_PATTERNS = {
        r'\b(?:after|with)\s*'
        r'(?:food|foods|meal|meals|eating)\b':
            'After food',

        r'\b(?:before|empty)\s*'
        r'(?:stomach|meal|food)\b':
            'Before food',

        r'\b(?:at bedtime|night|before sleep)\b':
            'At bedtime',

        r'\b(?:as directed|as advised)\b':
            'As directed',
    }

    def __init__(self):
        self.compiled_freq = [
            (re.compile(k, re.IGNORECASE), v)
            for k, v in self.FREQUENCY_PATTERNS.items()
        ]

        self.compiled_dur = [
            (re.compile(k, re.IGNORECASE), v)
            for k, v in self.DURATION_PATTERNS.items()
        ]

        self.compiled_route = [
            (re.compile(k, re.IGNORECASE), v)
            for k, v in self.ROUTE_PATTERNS.items()
        ]

        self.compiled_inst = [
            (re.compile(k, re.IGNORECASE), v)
            for k, v in self.INSTRUCTION_PATTERNS.items()
        ]

    # ---------------------------------------------------------
    # Medicine normalization
    # ---------------------------------------------------------
    def normalize_medicine_names(
        self,
        text: str
    ) -> NormalizedText:

        replacements = []
        normalized = text.lower()

        # Longest variants first
        all_variants = []

        for correct, variants in self.MEDICINE_CORRECTIONS.items():
            for variant in variants:
                all_variants.append((variant, correct))

        all_variants.sort(
            key=lambda x: len(x[0]),
            reverse=True
        )

        for variant, correct in all_variants:

            pattern = r'\b' + re.escape(variant) + r'\b'

            if re.search(pattern, normalized):

                normalized = re.sub(
                    pattern,
                    correct,
                    normalized,
                    flags=re.IGNORECASE
                )

                replacements.append(
                    (variant, correct)
                )

        return NormalizedText(
            text=normalized,
            replacements=replacements
        )

    # ---------------------------------------------------------
    # Dosage normalization
    # ---------------------------------------------------------
    def normalize_dosage(
        self,
        text: str
    ) -> NormalizedText:

        replacements = []
        normalized = text

        for pattern, repl in self.DOSAGE_PATTERNS:

            matches = list(
                re.finditer(
                    pattern,
                    normalized,
                    re.IGNORECASE
                )
            )

            for match in reversed(matches):

                old = match.group(0)

                new = re.sub(
                    pattern,
                    repl,
                    old,
                    flags=re.IGNORECASE
                )

                normalized = (
                    normalized[:match.start()]
                    + new
                    + normalized[match.end():]
                )

                replacements.append(
                    (old, new)
                )

        return NormalizedText(
            text=normalized,
            replacements=replacements
        )

    # ---------------------------------------------------------
    # Frequency
    # ---------------------------------------------------------
    def extract_frequency(
        self,
        text: str
    ) -> Tuple[str, List[Tuple[str, str]]]:

        replacements = []
        normalized = text

        for pattern, freq_code in self.compiled_freq:

            matches = list(pattern.finditer(normalized))

            for match in reversed(matches):

                old = match.group(0)

                normalized = (
                    normalized[:match.start()]
                    + freq_code
                    + normalized[match.end():]
                )

                replacements.append(
                    (old, freq_code)
                )

        return normalized, replacements

    # ---------------------------------------------------------
    # Duration
    # ---------------------------------------------------------
    def extract_duration(
        self,
        text: str
    ) -> Tuple[str, List[Tuple[str, str]]]:

        replacements = []
        normalized = text

        for pattern, repl in self.compiled_dur:

            matches = list(
                pattern.finditer(normalized)
            )

            for match in reversed(matches):

                old = match.group(0)

                new = pattern.sub(
                    repl,
                    old
                )

                normalized = (
                    normalized[:match.start()]
                    + new
                    + normalized[match.end():]
                )

                replacements.append(
                    (old, new)
                )

        return normalized, replacements

    # ---------------------------------------------------------
    # Route
    # ---------------------------------------------------------
    def extract_route(
        self,
        text: str
    ) -> Tuple[str, List[Tuple[str, str]]]:

        replacements = []
        normalized = text

        for pattern, route in self.compiled_route:

            match = pattern.search(normalized)

            if match:

                old = match.group(0)

                normalized = (
                    normalized[:match.start()]
                    + route
                    + normalized[match.end():]
                )

                replacements.append(
                    (old, route)
                )

                break

        return normalized, replacements

    # ---------------------------------------------------------
    # Instructions
    # ---------------------------------------------------------
    def extract_instructions(
        self,
        text: str
    ) -> Tuple[str, List[Tuple[str, str]]]:

        replacements = []
        normalized = text

        for pattern, inst in self.compiled_inst:

            match = pattern.search(normalized)

            if match:

                old = match.group(0)

                normalized = (
                    normalized[:match.start()]
                    + inst
                    + normalized[match.end():]
                )

                replacements.append(
                    (old, inst)
                )

        return normalized, replacements

    # ---------------------------------------------------------
    # Full normalization
    # ---------------------------------------------------------
    def normalize(self, text: str) -> Dict:

        all_replacements = []

        # 1. Medicine names
        result = self.normalize_medicine_names(text)

        all_replacements.extend(
            result.replacements
        )

        text = result.text

        # 2. Dosage
        result = self.normalize_dosage(text)

        all_replacements.extend(
            result.replacements
        )

        text = result.text

        # 3. Frequency
        text, reps = self.extract_frequency(text)

        all_replacements.extend(reps)

        # 4. Duration
        text, reps = self.extract_duration(text)

        all_replacements.extend(reps)

        # 5. Route
        text, reps = self.extract_route(text)

        all_replacements.extend(reps)

        # 6. Instructions
        text, reps = self.extract_instructions(text)

        all_replacements.extend(reps)

        return {
            "normalized_text": text,
            "replacements": all_replacements
        }


# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------

_normalizer_instance = None


def get_normalizer() -> TextNormalizer:

    global _normalizer_instance

    if _normalizer_instance is None:
        _normalizer_instance = TextNormalizer()

    return _normalizer_instance