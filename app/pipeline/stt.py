from faster_whisper import WhisperModel
import numpy as np
import torch
from typing import Optional, Dict, Any
import time
import os

from app.utils.audio_utils import normalize_audio, resample_audio


class WhisperSTT:
    """Speech-to-Text using faster-whisper."""

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        compute_type: str = "int8"
    ):
        self.model_size = model_size

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.compute_type = (
            compute_type
            if self.device == "cuda"
            else "int8"
        )

        self.model = None

        self._load_model()

    # =========================================================
    # LOAD MODEL
    # =========================================================

    def _load_model(self):

        print(
            f"Loading faster-whisper model: "
            f"{self.model_size} on {self.device} "
            f"({self.compute_type})"
        )

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "whisper"
            )
        )

    # =========================================================
    # AUDIO PREPROCESSING
    # =========================================================

    def _prepare_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:

        audio_data = np.asarray(
            audio_data,
            dtype=np.float32
        )

        # Remove NaN / infinite values
        audio_data = np.nan_to_num(
            audio_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Resample to 16 kHz
        if sample_rate != 16000:

            audio_data = resample_audio(
                audio_data,
                sample_rate,
                16000
            )

        # Normalize
        audio_data = normalize_audio(
            audio_data
        )

        # Remove very small noise
        noise_threshold = 0.005

        audio_data[
            np.abs(audio_data) < noise_threshold
        ] = 0.0

        # Make contiguous float32
        audio_data = np.ascontiguousarray(
            audio_data,
            dtype=np.float32
        )

        return audio_data

    # =========================================================
    # TRANSCRIBE
    # =========================================================

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en"
    ) -> Dict[str, Any]:

        start_time = time.time()

        # -----------------------------------------------------
        # Prepare audio
        # -----------------------------------------------------

        audio_data = self._prepare_audio(
            audio_data,
            sample_rate
        )

        # -----------------------------------------------------
        # Medical context prompt
        # -----------------------------------------------------

        initial_prompt = (
            "Medical prescription. "
            "Medicine names such as aspirin, "
            "paracetamol, acetaminophen, amoxicillin, "
            "azithromycin, ibuprofen, cetirizine, "
            "omeprazole, pantoprazole, metformin. "
            "Medical dosage, tablet, capsule, syrup, "
            "morning, afternoon, evening, night, "
            "before food, after food."
        )

        # -----------------------------------------------------
        # Whisper
        # -----------------------------------------------------

        segments, info = self.model.transcribe(

            audio_data,

            language=(
                language
                if language != "auto"
                else None
            ),

            beam_size=5,

            best_of=5,

            temperature=0.0,

            initial_prompt=initial_prompt,

            condition_on_previous_text=False,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 300
            }
        )

        # -----------------------------------------------------
        # Collect segments
        # -----------------------------------------------------

        segment_list = []

        full_text_parts = []

        all_logprobs = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
                "avg_logprob": segment.avg_logprob
            })

            full_text_parts.append(text)

            all_logprobs.append(
                segment.avg_logprob
            )

        full_text = " ".join(
            full_text_parts
        ).strip()

        # -----------------------------------------------------
        # Processing time
        # -----------------------------------------------------

        processing_time = (
            time.time() - start_time
        ) * 1000

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        confidence = None

        if all_logprobs:

            confidence = float(
                np.exp(
                    np.mean(
                        all_logprobs
                    )
                )
            )

            confidence = min(
                max(confidence, 0.0),
                1.0
            )

        # -----------------------------------------------------
        # Return
        # -----------------------------------------------------

        return {

            "text": full_text,

            "language": info.language,

            "confidence": confidence,

            "processingTimeMs":
                processing_time,

            "segments":
                segment_list
        }

    # =========================================================
    # FILE TRANSCRIPTION
    # =========================================================

    def transcribe_file(
        self,
        file_path: str,
        language: str = "en"
    ) -> Dict[str, Any]:

        start_time = time.time()

        initial_prompt = (
            "Medical prescription. "
            "Aspirin, paracetamol, amoxicillin, "
            "azithromycin, ibuprofen, cetirizine, "
            "omeprazole, pantoprazole, metformin. "
            "Tablet, capsule, syrup, dosage, "
            "morning, afternoon, evening, night, "
            "before food, after food."
        )

        segments, info = self.model.transcribe(

            file_path,

            language=(
                language
                if language != "auto"
                else None
            ),

            beam_size=5,

            best_of=5,

            temperature=0.0,

            initial_prompt=initial_prompt,

            condition_on_previous_text=False,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 300
            }
        )

        segment_list = []

        full_text_parts = []

        all_logprobs = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            segment_list.append({

                "start": segment.start,

                "end": segment.end,

                "text": text,

                "avg_logprob":
                    segment.avg_logprob
            })

            full_text_parts.append(text)

            all_logprobs.append(
                segment.avg_logprob
            )

        full_text = " ".join(
            full_text_parts
        ).strip()

        processing_time = (
            time.time() - start_time
        ) * 1000

        confidence = None

        if all_logprobs:

            confidence = float(
                np.exp(
                    np.mean(
                        all_logprobs
                    )
                )
            )

            confidence = min(
                max(confidence, 0.0),
                1.0
            )

        return {

            "text": full_text,

            "language": info.language,

            "confidence": confidence,

            "processingTimeMs":
                processing_time,

            "segments":
                segment_list
        }


# =============================================================
# SINGLETON
# =============================================================

_stt_instance: Optional[WhisperSTT] = None


def get_stt(
    model_size: str = "small"
) -> WhisperSTT:

    global _stt_instance

    if (
        _stt_instance is None
        or
        _stt_instance.model_size != model_size
    ):

        _stt_instance = WhisperSTT(
            model_size
        )

    return _stt_instance