import json
import httpx

from typing import List, Optional
from dataclasses import dataclass

from app.models.schemas import Route
from app.pipeline.nlp_extractor import ExtractedMedicine


# ============================================================
# OLLAMA CONFIG
# ============================================================

@dataclass
class OllamaConfig:

    # Local Ollama server
    base_url: str = "http://localhost:11434"

    # Local LLM
    model: str = "ministral-3:3b"

    # Low temperature = more consistent extraction
    temperature: float = 0.1

    timeout: float = 120.0


# ============================================================
# OLLAMA PRESCRIPTION EXTRACTOR
# ============================================================

class OllamaPrescriptionExtractor:
    """
    Local LLM prescription extractor.

    Model:
        ministral-3:3b

    Ollama:
        http://localhost:11434

    Flow:

        Transcript
            ↓
        Ministral 3:3B
            ↓
        Structured medicine extraction
            ↓
        ExtractedMedicine objects
    """

    def __init__(
        self,
        config: OllamaConfig = None
    ):

        self.config = (
            config
            or OllamaConfig()
        )

        self.client = httpx.Client(
            timeout=self.config.timeout
        )

        self.system_prompt = (
            self._get_system_prompt()
        )

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def _get_system_prompt(self) -> str:

        return """
You are a medical prescription extraction engine.

Your job is to extract medicines from a doctor's
spoken/dictated prescription.

You MUST extract ONLY information that is present
or clearly implied by the doctor's statement.

DO NOT invent medicine information.

DO NOT add medicines that were not mentioned.

DO NOT change dosage.

DO NOT change duration.

DO NOT change food instructions.

DO NOT hallucinate missing information.

------------------------------------------------------------
INPUT
------------------------------------------------------------

The user will provide a medical prescription transcript.

Example:

"Give aspirin 100 mg tablet one tablet in morning
and evening after food for 2 days and paracetamol
500 mg tablet one tablet morning afternoon and evening
after food for 3 days."

------------------------------------------------------------
EXTRACTION
------------------------------------------------------------

For every medicine extract:

1. medicineName
2. dosage
3. frequency
4. duration
5. route
6. instructions
7. confidence

------------------------------------------------------------
FREQUENCY NORMALIZATION
------------------------------------------------------------

Convert frequency to these codes:

Once daily:
1-0-0

Twice daily:
1-0-1

Three times daily:
1-1-1

Four times daily:
1-1-1-1

Morning:
1-0-0-0

Morning + evening:
1-0-1-0

Morning + afternoon:
1-1-0-0

Morning + afternoon + evening:
1-1-1-0

Morning + afternoon + evening + night:
1-1-1-1

As needed:
SOS

------------------------------------------------------------
FOOD INSTRUCTIONS
------------------------------------------------------------

Normalize:

"after food"
"after meals"
"after eating"

to:

AFTER FOOD

Normalize:

"before food"
"before meals"
"before eating"

to:

BEFORE FOOD

If no food instruction is mentioned:

null

------------------------------------------------------------
DOSAGE
------------------------------------------------------------

Keep the numeric dosage and unit.

Examples:

100 mg -> 100MG
500 mg -> 500MG
5 ml -> 5ML

------------------------------------------------------------
DURATION
------------------------------------------------------------

Examples:

for 2 days -> 2 DAYS
for 5 days -> 5 DAYS
for 1 week -> 1 WEEK
for 2 weeks -> 2 WEEKS

If duration is not mentioned:

null

------------------------------------------------------------
UNIT PER TIME
------------------------------------------------------------

Examples:

one tablet -> 1
two tablets -> 2
one capsule -> 1
two capsules -> 2
one spoon -> 1

If not specified:

1

------------------------------------------------------------
FORM
------------------------------------------------------------

Determine form from the spoken prescription.

Examples:

tablet -> TABLET
tablets -> TABLET

capsule -> CAPSULE
capsules -> CAPSULE

syrup -> SYRUP
injection -> INJECTION

If form is not mentioned:

UNKNOWN

------------------------------------------------------------
ROUTE
------------------------------------------------------------

Allowed values:

oral
topical
injection
inhalation
sublingual
rectal
ophthalmic
otic
nasal
transdermal
unknown

For tablets/capsules taken normally:

oral

------------------------------------------------------------
OUTPUT
------------------------------------------------------------

Return ONLY valid JSON.

The JSON MUST have exactly this structure:

{
  "medicines": [
    {
      "medicineName": "ASPIRIN",
      "dosage": "100MG",
      "frequency": "1-0-1-0",
      "duration": "2 DAYS",
      "route": "oral",
      "instructions": "AFTER FOOD",
      "form": "TABLET",
      "unitPerTime": "1",
      "confidence": 0.95
    }
  ],
  "notes": null
}

IMPORTANT:

The internal extraction JSON above is used by the
application transformer.

The final application will convert it into:

{
  "prescription": [
    {
      "name": "ASPIRIN",
      "form": "TABLET",
      "dosage": "100MG",
      "numberOfTime": "1-0-1-0 / MORNING/AFTERNOON/EVENING/NIGHT",
      "remarks": "AFTER FOOD",
      "timesPerDay": "2 DAYS",
      "unitPerTime": "1"
    }
  ]
}

Never return markdown.

Never return ```json.

Return JSON only.

------------------------------------------------------------
IMPORTANT SAFETY RULE
------------------------------------------------------------

If a field is not present in the transcript,
use null where appropriate.

Never guess missing dosage, duration, or instructions.

Extract ALL medicines mentioned in the transcript.
"""

    # ========================================================
    # ASYNC EXTRACTION
    # ========================================================

    async def extract_async(
        self,
        text: str
    ) -> List[ExtractedMedicine]:

        if not text or not text.strip():

            return []

        try:

            async with httpx.AsyncClient(
                timeout=self.config.timeout
            ) as client:

                response = await client.post(

                    f"{self.config.base_url}/api/chat",

                    json={

                        "model": self.config.model,

                        "messages": [

                            {
                                "role": "system",
                                "content": self.system_prompt
                            },

                            {
                                "role": "user",
                                "content": (
                                    "Extract the prescription "
                                    "from this transcript:\n\n"
                                    f"{text}"
                                )
                            }

                        ],

                        "temperature": (
                            self.config.temperature
                        ),

                        "format": "json",

                        "stream": False
                    }
                )

                response.raise_for_status()

                result = response.json()

                content = (
                    result
                    .get("message", {})
                    .get("content", "{}")
                )

                # ------------------------------------------------
                # Parse JSON
                # ------------------------------------------------

                try:

                    parsed = json.loads(
                        content
                    )

                except json.JSONDecodeError:

                    print(
                        "Ollama returned invalid JSON:"
                    )

                    print(content)

                    return []

                # ------------------------------------------------
                # Extract medicines
                # ------------------------------------------------

                medicines = []

                for med_data in parsed.get(
                    "medicines",
                    []
                ):

                    if not isinstance(
                        med_data,
                        dict
                    ):
                        continue

                    name = (
                        med_data
                        .get(
                            "medicineName",
                            ""
                        )
                    )

                    if not name:
                        continue

                    # --------------------------------------------
                    # Route
                    # --------------------------------------------

                    route_value = (
                        med_data
                        .get(
                            "route",
                            "unknown"
                        )
                    )

                    try:

                        route = Route(
                            route_value
                        )

                    except ValueError:

                        route = Route.UNKNOWN

                    # --------------------------------------------
                    # Confidence
                    # --------------------------------------------

                    confidence = (
                        med_data
                        .get(
                            "confidence",
                            0.8
                        )
                    )

                    try:

                        confidence = float(
                            confidence
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        confidence = 0.8

                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            confidence
                        )
                    )

                    # --------------------------------------------
                    # Create ExtractedMedicine
                    # --------------------------------------------

                    med = ExtractedMedicine(

                        name=str(
                            name
                        ).upper(),

                        dosage=(
                            str(
                                med_data.get(
                                    "dosage"
                                )
                            ).upper()
                            if med_data.get(
                                "dosage"
                            ) is not None
                            else None
                        ),

                        frequency=(
                            med_data.get(
                                "frequency"
                            )
                        ),

                        duration=(
                            str(
                                med_data.get(
                                    "duration"
                                )
                            ).upper()
                            if med_data.get(
                                "duration"
                            ) is not None
                            else None
                        ),

                        route=route,

                        instructions=(
                            str(
                                med_data.get(
                                    "instructions"
                                )
                            ).upper()
                            if med_data.get(
                                "instructions"
                            ) is not None
                            else None
                        ),

                        confidence=confidence
                    )

                    # Store additional fields dynamically
                    # for transformer compatibility.
                    med.form = (
                        str(
                            med_data.get(
                                "form",
                                "UNKNOWN"
                            )
                        ).upper()
                    )

                    med.unitPerTime = str(
                        med_data.get(
                            "unitPerTime",
                            "1"
                        )
                    )

                    medicines.append(
                        med
                    )

                return medicines

        except Exception as e:

            print(
                "Ollama extraction error:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            return []

    # ========================================================
    # SYNCHRONOUS EXTRACTION
    # ========================================================

    def extract(
        self,
        text: str
    ) -> List[ExtractedMedicine]:

        import asyncio

        try:

            loop = asyncio.get_event_loop()

        except RuntimeError:

            loop = asyncio.new_event_loop()

            asyncio.set_event_loop(
                loop
            )

        return loop.run_until_complete(
            self.extract_async(text)
        )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    async def check_model_available(
        self
    ) -> bool:

        try:

            response = await self._async_get(
                f"{self.config.base_url}/api/tags"
            )

            response.raise_for_status()

            models = (
                response
                .json()
                .get(
                    "models",
                    []
                )
            )

            return any(
                self.config.model
                in model.get(
                    "name",
                    ""
                )
                for model in models
            )

        except Exception as e:

            print(
                f"Ollama model check failed: {e}"
            )

            return False

    # ========================================================
    # ASYNC GET HELPER
    # ========================================================

    async def _async_get(
        self,
        url: str
    ):

        async with httpx.AsyncClient(
            timeout=self.config.timeout
        ) as client:

            return await client.get(
                url
            )

    # ========================================================
    # PULL MODEL
    # ========================================================

    async def pull_model(
        self
    ) -> bool:

        try:

            async with httpx.AsyncClient(
                timeout=300.0
            ) as client:

                response = await client.post(

                    f"{self.config.base_url}/api/pull",

                    json={
                        "name": self.config.model
                    }
                )

                response.raise_for_status()

                return True

        except Exception as e:

            print(
                f"Failed to pull model: {e}"
            )

            return False

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        try:

            self.client.close()

        except Exception:

            pass


# ============================================================
# SINGLETON
# ============================================================

_ollama_extractor_instance = None


def get_ollama_extractor(
    config: OllamaConfig = None
) -> OllamaPrescriptionExtractor:

    global _ollama_extractor_instance

    if _ollama_extractor_instance is None:

        if config is None:

            config = OllamaConfig(
                base_url="http://localhost:11434",
                model="ministral-3:3b",
                temperature=0.1,
                timeout=120.0
            )

        _ollama_extractor_instance = (
            OllamaPrescriptionExtractor(
                config
            )
        )

    return _ollama_extractor_instance