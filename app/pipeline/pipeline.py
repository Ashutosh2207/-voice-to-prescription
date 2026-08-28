import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

import numpy as np

from app.models.schemas import (
    Prescription,
    Medicine,
    ExtractionResult,
    ValidationResult,
    Route,
)

from app.pipeline.stt import get_stt
from app.pipeline.ollama_extractor import (
    get_ollama_extractor,
    OllamaPrescriptionExtractor,
    OllamaConfig,
)

from app.pipeline.validator import get_validator
from app.pipeline.transformer import get_transformer

from app.utils.audio_utils import (
    decode_base64_audio,
    load_audio_file,
    is_audio_valid,
)


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

@dataclass
class PipelineConfig:

    # Local Whisper model
   
    whisper_model: str = "base"

    # Ollama is ALWAYS enabled
    use_ollama_extraction: bool = True

    # Default local Ollama configuration
    ollama_config: Optional[OllamaConfig] = None

    # spaCy is disabled by default.
    # We do NOT want spaCy replacing the LLM.
    use_spacy_fallback: bool = False

    spacy_model: str = "en_core_web_sm"

    language: str = "en"


# ============================================================
# MAIN PIPELINE
# ============================================================

class VoiceToPrescriptionPipeline:
    """
    Voice-to-Prescription pipeline.

    Flow:

        Audio
          ↓
        faster-whisper
          ↓
        Transcript
          ↓
        Ollama / ministral-3:3b
          ↓
        Medicine extraction
          ↓
        Validation
          ↓
        Medhant Lite transformer
          ↓
        Final JSON
    """

    def __init__(
        self,
        config: PipelineConfig = None
    ):

        self.config = config or PipelineConfig()

        # ----------------------------------------------------
        # Speech-to-text
        # ----------------------------------------------------

        self.stt = get_stt(
            self.config.whisper_model
        )

        # ----------------------------------------------------
        # Validator
        # ----------------------------------------------------

        self.validator = get_validator()

        # ----------------------------------------------------
        # Ollama
        # ----------------------------------------------------

        self.ollama_extractor = None

        # Force Ollama
        ollama_cfg = (
            self.config.ollama_config
            or OllamaConfig(
                model="ministral-3:3b",
                base_url="http://localhost:11434"
            )
        )

        self.ollama_extractor = get_ollama_extractor(
            ollama_cfg
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        self.transformer = get_transformer()

    # ========================================================
    # EXTRACT USING OLLAMA
    # ========================================================

    def _extract_with_ollama(
        self,
        text: str
    ):

        if not text or not text.strip():

            return []

        if self.ollama_extractor is None:

            raise RuntimeError(
                "Ollama extractor is not initialized."
            )

        medicines = (
            self.ollama_extractor.extract(
                text
            )
        )

        if not medicines:

            raise RuntimeError(
                "Ollama did not extract any medicines "
                "from the transcript."
            )

        return medicines

    # ========================================================
    # CONVERT EXTRACTED MEDICINES TO SCHEMA
    # ========================================================

    def _build_prescription(
        self,
        extracted_medicines,
        transcript: str
    ) -> Prescription:

        medicines = []

        for ext_med in extracted_medicines:

            med = Medicine(

                medicineName=(
                    ext_med.name
                    if ext_med.name
                    else ""
                ),

                dosage=(
                    ext_med.dosage
                    if ext_med.dosage
                    else ""
                ),

                frequency=(
                    ext_med.frequency
                    if ext_med.frequency
                    else ""
                ),

                duration=(
                    ext_med.duration
                    if ext_med.duration
                    else ""
                ),

                route=(
                    ext_med.route
                    if ext_med.route
                    else Route.UNKNOWN
                ),

                instructions=(
                    ext_med.instructions
                    if ext_med.instructions
                    else ""
                ),

                confidence=(
                    ext_med.confidence
                    if ext_med.confidence is not None
                    else 1.0
                )
            )

            medicines.append(med)

        return Prescription(

            medicines=medicines,

            notes=None,

            rawTranscript=transcript
        )

    # ========================================================
    # PROCESS AUDIO
    # ========================================================

    def process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: str = "en"
    ) -> ExtractionResult:

        start_time = time.time()

        # ----------------------------------------------------
        # Validate audio
        # ----------------------------------------------------

        valid, msg = is_audio_valid(
            audio_data,
            sample_rate
        )

        if not valid:

            return ExtractionResult(

                prescription=Prescription(),

                validation=ValidationResult(
                    isValid=False,
                    issues=[]
                ),

                processingTimeMs=(
                    time.time() - start_time
                ) * 1000,

                sttConfidence=0.0
            )

        # ----------------------------------------------------
        # STEP 1
        # Speech -> Text
        # ----------------------------------------------------

        stt_result = self.stt.transcribe(
            audio_data,
            sample_rate,
            language
        )

        transcript = stt_result.get(
            "text",
            ""
        )

        stt_confidence = stt_result.get(
            "confidence"
        )

        transcript = (
            transcript.strip()
            if transcript
            else ""
        )

        if not transcript:

            return ExtractionResult(

                prescription=Prescription(
                    rawTranscript=""
                ),

                validation=ValidationResult(
                    isValid=False,
                    issues=[]
                ),

                processingTimeMs=(
                    time.time() - start_time
                ) * 1000,

                sttConfidence=stt_confidence
            )

        # ----------------------------------------------------
        # STEP 2
        # Transcript -> Ollama
        # ----------------------------------------------------

        try:

            extracted_medicines = (
                self._extract_with_ollama(
                    transcript
                )
            )

        except Exception as e:

            # IMPORTANT:
            # Do NOT silently switch to spaCy.
            # We want Ollama to be the actual LLM.

            return ExtractionResult(

                prescription=Prescription(
                    medicines=[],
                    notes=f"Ollama extraction failed: {str(e)}",
                    rawTranscript=transcript
                ),

                validation=ValidationResult(
                    isValid=False,
                    issues=[]
                ),

                processingTimeMs=(
                    time.time() - start_time
                ) * 1000,

                sttConfidence=stt_confidence
            )

        # ----------------------------------------------------
        # STEP 3
        # Build internal prescription schema
        # ----------------------------------------------------

        prescription = self._build_prescription(
            extracted_medicines,
            transcript
        )

        # ----------------------------------------------------
        # STEP 4
        # Validate
        # ----------------------------------------------------

        validation = self.validator.validate(
            prescription
        )

        # ----------------------------------------------------
        # STEP 5
        # Return
        # ----------------------------------------------------

        processing_time = (
            time.time() - start_time
        ) * 1000

        return ExtractionResult(

            prescription=prescription,

            validation=validation,

            processingTimeMs=processing_time,

            sttConfidence=stt_confidence
        )

    # ========================================================
    # TRANSFORM PRESCRIPTION
    # ========================================================

    def transform_prescription(
        self,
        prescription: Prescription
    ) -> Dict[str, Any]:

        """
        Convert internal prescription schema
        to exact Medhant Lite JSON structure.
        """

        return self.transformer.transform(
            prescription
        )

    # ========================================================
    # PROCESS AUDIO FILE
    # ========================================================

    def process_audio_file(
        self,
        file_path: str,
        language: str = "en"
    ) -> ExtractionResult:

        audio_data, sample_rate = (
            load_audio_file(file_path)
        )

        return self.process_audio(
            audio_data,
            sample_rate,
            language
        )

    # ========================================================
    # PROCESS BASE64 AUDIO
    # ========================================================

    def process_base64_audio(
        self,
        base64_audio: str,
        language: str = "en"
    ) -> ExtractionResult:

        audio_data, sample_rate = (
            decode_base64_audio(
                base64_audio
            )
        )

        return self.process_audio(
            audio_data,
            sample_rate,
            language
        )

    # ========================================================
    # PROCESS TEXT
    # ========================================================

    def process_text(
        self,
        text: str
    ) -> ExtractionResult:

        start_time = time.time()

        text = (
            text.strip()
            if text
            else ""
        )

        if not text:

            return ExtractionResult(

                prescription=Prescription(
                    rawTranscript=""
                ),

                validation=ValidationResult(
                    isValid=False,
                    issues=[]
                ),

                processingTimeMs=0,

                sttConfidence=None
            )

        # ----------------------------------------------------
        # TEXT -> OLLAMA
        # ----------------------------------------------------

        try:

            extracted_medicines = (
                self._extract_with_ollama(
                    text
                )
            )

        except Exception as e:

            return ExtractionResult(

                prescription=Prescription(

                    medicines=[],

                    notes=(
                        "Ollama extraction failed: "
                        f"{str(e)}"
                    ),

                    rawTranscript=text
                ),

                validation=ValidationResult(
                    isValid=False,
                    issues=[]
                ),

                processingTimeMs=(
                    time.time() - start_time
                ) * 1000,

                sttConfidence=None
            )

        # ----------------------------------------------------
        # BUILD PRESCRIPTION
        # ----------------------------------------------------

        prescription = self._build_prescription(
            extracted_medicines,
            text
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        validation = self.validator.validate(
            prescription
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return ExtractionResult(

            prescription=prescription,

            validation=validation,

            processingTimeMs=(
                time.time() - start_time
            ) * 1000,

            sttConfidence=None
        )


# ============================================================
# GLOBAL PIPELINE
# ============================================================

_pipeline_instance: Optional[
    VoiceToPrescriptionPipeline
] = None


# ============================================================
# GET PIPELINE
# ============================================================

def get_pipeline(
    config: PipelineConfig = None
) -> VoiceToPrescriptionPipeline:

    global _pipeline_instance

    if _pipeline_instance is None:

        # Always create with Ollama enabled
        if config is None:

            config = PipelineConfig(
                use_ollama_extraction=True,
                use_spacy_fallback=False,
                ollama_config=OllamaConfig(
                    model="ministral-3:3b",
                    base_url="http://localhost:11434"
                )
            )

        else:

            # Force local LLM
            config.use_ollama_extraction = True

            # Disable spaCy fallback
            config.use_spacy_fallback = False

            if config.ollama_config is None:

                config.ollama_config = OllamaConfig(
                    model="ministral-3:3b",
                    base_url="http://localhost:11434"
                )

        _pipeline_instance = (
            VoiceToPrescriptionPipeline(
                config
            )
        )

    return _pipeline_instance


# ============================================================
# CREATE NEW PIPELINE
# ============================================================

def create_pipeline(
    config: PipelineConfig
) -> VoiceToPrescriptionPipeline:

    # Force Ollama
    config.use_ollama_extraction = True

    # Disable spaCy fallback
    config.use_spacy_fallback = False

    # Force Ministral
    if config.ollama_config is None:

        config.ollama_config = OllamaConfig(
            model="ministral-3:3b",
            base_url="http://localhost:11434"
        )

    return VoiceToPrescriptionPipeline(
        config
    )