from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import asyncio

from app.models.schemas import (
    PrescriptionResponse,
    AudioProcessRequest,
)

from app.pipeline import (
    get_pipeline,
    create_pipeline,
    PipelineConfig,
    OllamaConfig,
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Voice-to-Prescription API",
    description="AI-powered voice to prescription conversion for Medhant Lite",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODELS
# =========================================================

class TextProcessRequest(BaseModel):
    text: str
    language: str = "en"


class PipelineConfigRequest(BaseModel):
    whisper_model: str = "base"
    use_ollama_extraction: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "ministral-3:3b"
    use_spacy_fallback: bool = True
    spacy_model: str = "en_core_web_sm"
    language: str = "en"


# =========================================================
# GLOBAL PIPELINE
# =========================================================

pipeline = get_pipeline()


# =========================================================
# HELPER
# =========================================================

async def run_in_thread(func, *args):
    """
    Run synchronous pipeline functions outside
    FastAPI's active event loop.
    """

    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(
        None,
        lambda: func(*args)
    )


def get_raw_transcript(result) -> str:
    """
    Safely get Whisper raw transcript
    from pipeline result.
    """

    try:

        if result is None:
            return ""

        prescription = getattr(
            result,
            "prescription",
            None
        )

        if prescription is None:
            return ""

        transcript = getattr(
            prescription,
            "rawTranscript",
            ""
        )

        if transcript is None:
            return ""

        return str(transcript).strip()

    except Exception:
        return ""


def get_validation(result):
    """
    Safely convert validation object to dictionary.
    """

    try:

        validation = getattr(
            result,
            "validation",
            None
        )

        if validation is None:
            return None

        if hasattr(validation, "model_dump"):
            return validation.model_dump()

        if hasattr(validation, "dict"):
            return validation.dict()

        if isinstance(validation, dict):
            return validation

        return None

    except Exception:
        return None


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {
        "service": "Voice-to-Prescription API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/process-audio",
            "/process-base64",
            "/process-text",
            "/process-audio/transformed",
            "/process-base64/transformed",
            "/process-text/transformed",
            "/configure",
            "/schema",
            "/health"
        ]
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "pipeline": "loaded"
    }


# =========================================================
# PROCESS UPLOADED AUDIO
# =========================================================

@app.post(
    "/process-audio",
    response_model=PrescriptionResponse
)
async def process_audio(
    file: UploadFile = File(...),
    language: str = Form("en"),
    return_transcript: bool = Form(True)
):
    """Process uploaded audio file."""

    tmp_path = None

    try:

        suffix = os.path.splitext(
            file.filename or ".wav"
        )[1]

        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            content = await file.read()

            tmp.write(content)

            tmp_path = tmp.name

        result = await run_in_thread(
            pipeline.process_audio_file,
            tmp_path,
            language
        )

        if not return_transcript:

            result.prescription.rawTranscript = None

        return PrescriptionResponse(
            success=True,
            data=result
        )

    except Exception as e:

        return PrescriptionResponse(
            success=False,
            error=str(e)
        )

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            try:
                os.unlink(tmp_path)

            except Exception:
                pass


# =========================================================
# PROCESS BASE64 AUDIO
# =========================================================

@app.post(
    "/process-base64",
    response_model=PrescriptionResponse
)
async def process_base64(
    request: AudioProcessRequest
):
    """Process base64 encoded audio."""

    try:

        if not request.audioBase64:

            raise HTTPException(
                status_code=400,
                detail="audioBase64 is required"
            )

        result = await run_in_thread(
            pipeline.process_base64_audio,
            request.audioBase64,
            request.language
        )

        if not request.returnTranscript:

            result.prescription.rawTranscript = None

        return PrescriptionResponse(
            success=True,
            data=result
        )

    except HTTPException:
        raise

    except Exception as e:

        return PrescriptionResponse(
            success=False,
            error=str(e)
        )


# =========================================================
# PROCESS TEXT
# =========================================================

@app.post(
    "/process-text",
    response_model=PrescriptionResponse
)
async def process_text(
    request: TextProcessRequest
):
    """Process text directly and bypass STT."""

    try:

        result = await run_in_thread(
            pipeline.process_text,
            request.text
        )

        return PrescriptionResponse(
            success=True,
            data=result
        )

    except Exception as e:

        return PrescriptionResponse(
            success=False,
            error=str(e)
        )


# =========================================================
# CONFIGURE PIPELINE
# =========================================================

@app.post(
    "/configure",
    response_model=PrescriptionResponse
)
async def configure_pipeline(
    config: PipelineConfigRequest
):
    """Reconfigure pipeline with new settings."""

    global pipeline

    try:

        ollama_config = None

        if config.use_ollama_extraction:

            ollama_config = OllamaConfig(
                base_url=config.ollama_base_url,
                model=config.ollama_model
            )

        new_pipeline = await run_in_thread(
            create_pipeline,
            PipelineConfig(
                whisper_model=config.whisper_model,
                use_ollama_extraction=config.use_ollama_extraction,
                ollama_config=ollama_config,
                use_spacy_fallback=config.use_spacy_fallback,
                spacy_model=config.spacy_model,
                language=config.language
            )
        )

        pipeline = new_pipeline

        return PrescriptionResponse(
            success=True,
            data=None
        )

    except Exception as e:

        return PrescriptionResponse(
            success=False,
            error=str(e)
        )


# =========================================================
# SCHEMA
# =========================================================

@app.get("/schema")
async def get_schema():
    """Get prescription JSON schema."""

    return {
        "prescription": {
            "medicines": [
                {
                    "medicineName": "string (required)",
                    "dosage": "string (e.g., '500 mg')",
                    "frequency": "string (e.g., '1-0-1', 'SOS')",
                    "duration": "string (e.g., '5 days')",
                    "route": (
                        "enum: oral|topical|injection|"
                        "inhalation|sublingual|rectal|"
                        "ophthalmic|otic|nasal|transdermal|unknown"
                    ),
                    "instructions": "string (e.g., 'After food')",
                    "confidence": "float 0-1"
                }
            ],
            "notes": "string",
            "rawTranscript": "string"
        },
        "validation": {
            "isValid": "boolean",
            "issues": [
                {
                    "field": "string",
                    "medicineIndex": "int",
                    "issue": "string",
                    "severity": "error|warning|info"
                }
            ],
            "missingFields": [
                "string"
            ]
        }
    }


# =========================================================
# TRANSFORMED - UPLOADED AUDIO
# =========================================================

@app.post(
    "/process-audio/transformed"
)
async def process_audio_transformed(
    file: UploadFile = File(...),
    language: str = Form("en"),
):
    """
    Process uploaded audio and return:

    - rawTranscript
    - transformed prescription
    - validation
    - processing time
    - STT confidence
    """

    tmp_path = None

    try:

        suffix = os.path.splitext(
            file.filename or ".wav"
        )[1]

        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            content = await file.read()

            tmp.write(content)

            tmp_path = tmp.name

        # Audio -> Whisper -> Extraction
        result = await run_in_thread(
            pipeline.process_audio_file,
            tmp_path,
            language
        )

        # Raw Whisper transcript
        raw_transcript = get_raw_transcript(
            result
        )

        # Transform to Medhant Lite format
        transformed = pipeline.transform_prescription(
            result.prescription
        )

        return {
            "success": True,

            # Separate API field for UI
            "rawTranscript": raw_transcript,

            "data": transformed,

            "validation": get_validation(
                result
            ),

            "processingTimeMs": getattr(
                result,
                "processingTimeMs",
                None
            ),

            "sttConfidence": getattr(
                result,
                "sttConfidence",
                None
            )
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "rawTranscript": ""
        }

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            try:
                os.unlink(tmp_path)

            except Exception:
                pass


# =========================================================
# TRANSFORMED - BASE64 / MICROPHONE
# =========================================================

@app.post(
    "/process-base64/transformed"
)
async def process_base64_transformed(
    request: AudioProcessRequest
):
    """
    Process microphone/base64 audio and return:

    - rawTranscript
    - transformed prescription
    - validation
    - processing time
    - STT confidence
    """

    try:

        if not request.audioBase64:

            raise HTTPException(
                status_code=400,
                detail="audioBase64 is required"
            )

        # Mic -> Base64 -> Whisper -> Extraction
        result = await run_in_thread(
            pipeline.process_base64_audio,
            request.audioBase64,
            request.language
        )

        # Raw Whisper transcript
        raw_transcript = get_raw_transcript(
            result
        )

        # Transform to Medhant Lite format
        transformed = pipeline.transform_prescription(
            result.prescription
        )

        return {
            "success": True,

            "rawTranscript": raw_transcript,

            "data": transformed,

            "validation": get_validation(
                result
            ),

            "processingTimeMs": getattr(
                result,
                "processingTimeMs",
                None
            ),

            "sttConfidence": getattr(
                result,
                "sttConfidence",
                None
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "rawTranscript": ""
        }


# =========================================================
# TRANSFORMED - TEXT
# =========================================================

@app.post(
    "/process-text/transformed"
)
async def process_text_transformed(
    request: TextProcessRequest
):
    """
    Process text and return transformed prescription.

    For text input, the supplied text is returned as
    rawTranscript so Streamlit can display it in the
    same expandable section.
    """

    try:

        result = await run_in_thread(
            pipeline.process_text,
            request.text
        )

        transformed = pipeline.transform_prescription(
            result.prescription
        )

        return {
            "success": True,

            "rawTranscript": request.text,

            "data": transformed,

            "validation": get_validation(
                result
            ),

            "processingTimeMs": getattr(
                result,
                "processingTimeMs",
                None
            ),

            "sttConfidence": None
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "rawTranscript": ""
        }


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )