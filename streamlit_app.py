import streamlit as st
import requests
import json
import base64
from typing import Optional, Dict
from audio_recorder_streamlit import audio_recorder


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Voice-to-Prescription | Medhant Lite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

API_BASE = "http://localhost:8000"

# Audio processing:
# Audio -> Whisper -> Extraction -> Validation
AUDIO_TIMEOUT = 300

# Text processing
TEXT_TIMEOUT = 120


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}

.sub-header {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 2rem;
}

.medicine-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.medicine-name {
    font-weight: 600;
    font-size: 1.1rem;
    color: #333;
}

.field-label {
    font-size: 0.85rem;
    color: #666;
    font-weight: 500;
}

.field-value {
    font-size: 0.95rem;
    color: #333;
}

.warning-badge {
    background: #fff3cd;
    color: #856404;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.error-badge {
    background: #f8d7da;
    color: #721c24;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.success-badge {
    background: #d4edda;
    color: #155724;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.transcript-box {
    background: #f8fbff;
    border: 1px solid #d7e7f7;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0 1rem 0;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# API HEALTH CHECK
# ============================================================

def check_api_health() -> bool:
    """Check if API server is running."""

    try:

        response = requests.get(
            f"{API_BASE}/health",
            timeout=5
        )

        return response.status_code == 200

    except Exception:
        return False


# ============================================================
# PROCESS UPLOADED AUDIO
# ============================================================

def process_audio_file(
    audio_file,
    language: str = "en"
) -> Optional[Dict]:
    """Send uploaded audio to transformed API."""

    try:

        files = {
            "file": (
                audio_file.name,
                audio_file.getvalue(),
                audio_file.type or "application/octet-stream"
            )
        }

        data = {
            "language": language
        }

        response = requests.post(
            f"{API_BASE}/process-audio/transformed",
            files=files,
            data=data,
            timeout=AUDIO_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:

        st.error(
            f"⏱️ Audio processing exceeded "
            f"{AUDIO_TIMEOUT} seconds. "
            "Please try a shorter or clearer recording."
        )

        return None

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the API server. "
            f"Make sure the API is running at {API_BASE}"
        )

        return None

    except requests.exceptions.HTTPError as e:

        st.error(
            f"❌ API HTTP Error: {e}"
        )

        return None

    except Exception as e:

        st.error(
            f"❌ API Error: {e}"
        )

        return None


# ============================================================
# PROCESS RECORDED BASE64 AUDIO
# ============================================================

def process_base64_audio(
    base64_audio: str,
    language: str = "en"
) -> Optional[Dict]:
    """Send recorded microphone audio to transformed API."""

    try:

        payload = {
            "audioBase64": base64_audio,
            "language": language,
            "returnTranscript": True
        }

        response = requests.post(
            f"{API_BASE}/process-base64/transformed",
            json=payload,
            timeout=AUDIO_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:

        st.error(
            f"⏱️ Voice processing exceeded "
            f"{AUDIO_TIMEOUT} seconds. "
            "Please try a shorter or clearer recording."
        )

        return None

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the API server. "
            f"Make sure the API is running at {API_BASE}"
        )

        return None

    except requests.exceptions.HTTPError as e:

        st.error(
            f"❌ API HTTP Error: {e}"
        )

        return None

    except Exception as e:

        st.error(
            f"❌ API Error: {e}"
        )

        return None


# ============================================================
# PROCESS TEXT
# ============================================================

def process_text(
    text: str
) -> Optional[Dict]:
    """Send text to transformed API."""

    try:

        response = requests.post(
            f"{API_BASE}/process-text/transformed",
            json={
                "text": text,
                "language": "en"
            },
            timeout=TEXT_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:

        st.error(
            f"⏱️ Text processing exceeded "
            f"{TEXT_TIMEOUT} seconds."
        )

        return None

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the API server."
        )

        return None

    except requests.exceptions.HTTPError as e:

        st.error(
            f"❌ API HTTP Error: {e}"
        )

        return None

    except Exception as e:

        st.error(
            f"❌ API Error: {e}"
        )

        return None


# ============================================================
# GET RAW TRANSCRIPT
# ============================================================

def get_raw_transcript(
    result: Dict
) -> str:
    """
    Extract raw transcript from API response.
    """

    if not isinstance(result, dict):
        return ""

    # Current API response
    transcript = result.get(
        "rawTranscript"
    )

    if isinstance(transcript, str):
        return transcript.strip()

    # Nested fallback
    data = result.get(
        "data"
    )

    if isinstance(data, dict):

        transcript = data.get(
            "rawTranscript"
        )

        if isinstance(transcript, str):
            return transcript.strip()

    # Older response fallback
    prescription_data = result.get(
        "prescription"
    )

    if isinstance(
        prescription_data,
        dict
    ):

        transcript = prescription_data.get(
            "rawTranscript"
        )

        if isinstance(
            transcript,
            str
        ):
            return transcript.strip()

    return ""


# ============================================================
# GET PRESCRIPTION
# ============================================================

def get_prescription(
    result: Dict
) -> list:
    """
    Extract prescription list from API response.
    """

    if not isinstance(
        result,
        dict
    ):
        return []

    # Direct format
    prescription = result.get(
        "prescription"
    )

    if isinstance(
        prescription,
        list
    ):
        return prescription

    # Wrapped format
    data = result.get(
        "data"
    )

    if isinstance(
        data,
        dict
    ):

        prescription = data.get(
            "prescription"
        )

        if isinstance(
            prescription,
            list
        ):
            return prescription

    return []


# ============================================================
# RENDER MEDICINE
# ============================================================

def render_medicine_card(
    medicine: Dict,
    index: int
):
    """Render one medicine."""

    st.markdown(
        f"### 💊 Medicine {index}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Name**")

        st.write(
            medicine.get(
                "name",
                "Not specified"
            )
        )

        st.markdown("**Form**")

        st.write(
            medicine.get(
                "form",
                "Not specified"
            )
        )

        st.markdown("**Dosage**")

        st.write(
            medicine.get(
                "dosage",
                "Not specified"
            )
        )

        st.markdown("**Unit Per Time**")

        st.write(
            medicine.get(
                "unitPerTime",
                "Not specified"
            )
        )

    with col2:

        st.markdown("**Number Of Time**")

        st.write(
            medicine.get(
                "numberOfTime",
                "Not specified"
            )
        )

        st.markdown("**Remarks**")

        st.write(
            medicine.get(
                "remarks",
                "Not specified"
            )
        )

        st.markdown("**Duration**")

        st.write(
            medicine.get(
                "timesPerDay",
                "Not specified"
            )
        )

    st.divider()


# ============================================================
# RENDER RESULTS
# ============================================================

def render_results(
    result: Dict
):
    """
    Render prescription results.

    Raw transcript is shown only inside
    a separate expandable section.
    """

    if not result:

        st.error(
            "No response received from API."
        )

        return

    # ========================================================
    # RAW TRANSCRIPT
    # ========================================================

    raw_transcript = get_raw_transcript(
        result
    )

    with st.expander(
        "🎙️ Raw Transcript — Click to View",
        expanded=False
    ):

        if raw_transcript:

            st.markdown(
                "### 📝 Speech-to-Text"
            )

            st.caption(
                "This is the text generated by Whisper "
                "from your uploaded or microphone voice."
            )

            st.markdown(
                '<div class="transcript-box">',
                unsafe_allow_html=True
            )

            st.text_area(
                "What the system heard",
                value=raw_transcript,
                height=180,
                disabled=True,
                label_visibility="collapsed"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "No raw transcript was returned by the API."
            )

    # ========================================================
    # STT INFORMATION
    # ========================================================

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.markdown(
            "### 🎯 STT Confidence"
        )

        confidence = result.get(
            "sttConfidence"
        )

        if confidence is not None:

            try:

                st.write(
                    f"{float(confidence):.2%}"
                )

            except Exception:

                st.write(
                    str(confidence)
                )

        else:

            st.write(
                "Not available"
            )

    with info_col2:

        st.markdown(
            "### ⏱ Processing Time"
        )

        processing_time = result.get(
            "processingTimeMs"
        )

        if processing_time is not None:

            try:

                processing_seconds = (
                    float(processing_time) / 1000
                )

                st.write(
                    f"{processing_seconds:.2f} seconds"
                )

            except Exception:

                st.write(
                    str(processing_time)
                )

        else:

            st.write(
                "Not available"
            )

    st.divider()

    # ========================================================
    # VALIDATION
    # ========================================================

    validation = result.get(
        "validation"
    )

    if isinstance(
        validation,
        dict
    ):

        is_valid = validation.get(
            "isValid"
        )

        if is_valid is True:

            st.success(
                "✅ Prescription validation passed."
            )

        elif is_valid is False:

            st.warning(
                "⚠️ Prescription validation "
                "reported issues."
            )

        issues = validation.get(
            "issues",
            []
        )

        if issues:

            with st.expander(
                "⚠️ View Validation Issues",
                expanded=False
            ):

                for issue in issues:

                    issue_text = issue.get(
                        "issue",
                        "Unknown issue"
                    )

                    severity = issue.get(
                        "severity",
                        "warning"
                    )

                    st.write(
                        f"**{severity.upper()}** — "
                        f"{issue_text}"
                    )

    # ========================================================
    # PRESCRIPTION
    # ========================================================

    prescription = get_prescription(
        result
    )

    st.success(
        f"✅ {len(prescription)} medicine(s) extracted"
    )

    if prescription:

        st.markdown(
            "## 💊 Prescription"
        )

        for index, medicine in enumerate(
            prescription,
            start=1
        ):

            render_medicine_card(
                medicine,
                index
            )

    else:

        st.warning(
            "No medicines extracted. "
            "Please try again with a clearer "
            "voice command."
        )

    # ========================================================
    # JSON OUTPUT
    # ========================================================

    final_output = {
        "prescription": prescription
    }

    st.markdown(
        "## 📋 JSON Output"
    )

    # Raw transcript intentionally excluded
    st.json(
        final_output
    )

    # ========================================================
    # DOWNLOAD JSON
    # ========================================================

    json_string = json.dumps(
        final_output,
        indent=2
    )

    st.download_button(
        label="📥 Download JSON",
        data=json_string,
        file_name="prescription.json",
        mime="application/json"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-header">'
        '🩺 Voice-to-Prescription'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-header">'
        'Medhant Lite — Dictate prescriptions, get structured JSON'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # API HEALTH
    # ========================================================

    api_healthy = check_api_health()

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.markdown(
            "## ⚙️ Settings"
        )

        # ----------------------------------------------------
        # API STATUS
        # ----------------------------------------------------

        if api_healthy:

            st.success(
                "✅ API Connected"
            )

        else:

            st.error(
                "❌ API Disconnected"
            )

            st.caption(
                f"Expected at {API_BASE}"
            )

        st.divider()

        # ----------------------------------------------------
        # INPUT MODE
        # ----------------------------------------------------

        st.markdown(
            "### 🎤 Input Mode"
        )

        input_mode = st.radio(
            "Choose input type",
            [
                "🎤 Upload Audio",
                "🎙️ Record Audio",
                "⌨️ Text Input"
            ],
            label_visibility="collapsed",
            help="Choose how to provide prescription input."
        )

        st.divider()

        # ----------------------------------------------------
        # HOW TO USE
        # ----------------------------------------------------

        st.markdown(
            "### ℹ️ How to Use"
        )

        st.markdown(
            """
**1. Choose an input method**

🎤 Upload an audio file  
🎙️ Record using the microphone  
⌨️ Enter prescription text

**2. Process the input**

The system converts the voice into text and extracts the prescription.

**3. Review the result**

Open **🎙️ Raw Transcript** to see exactly what Whisper understood.

**4. Check the prescription**

Review medicine name, dosage, frequency, remarks, and duration.

**5. Download JSON**

Download the structured prescription when required.

### Tips

- Speak clearly at a normal pace.
- Say the medicine name first.
- Mention dosage.
- Mention frequency.
- Mention before/after food.
- Mention duration.

### Example

*"Aspirin 100 mg tablet, one tablet morning and evening after food for 2 days."*
"""
        )

    # ========================================================
    # API NOT RUNNING
    # ========================================================

    if not api_healthy:

        st.error(
            "⚠️ API server not running. "
            "Start it with: "
            "`python run.py api`"
        )

        st.stop()

    # ========================================================
    # INPUT SECTION
    # ========================================================

    st.markdown(
        "## 📥 Input Prescription"
    )

    # ========================================================
    # UPLOAD AUDIO
    # ========================================================

    if input_mode == "🎤 Upload Audio":

        audio_file = st.file_uploader(
            "Upload audio file (WAV, MP3, M4A, OGG, FLAC)",
            type=[
                "wav",
                "mp3",
                "m4a",
                "ogg",
                "flac"
            ],
            help=(
                "Shorter and clearer recordings "
                "generally provide better results."
            )
        )

        if audio_file:

            st.audio(
                audio_file,
                format=audio_file.type
            )

            if st.button(
                "🔄 Process Audio",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Processing audio... "
                    "Whisper + prescription extraction "
                    "may take some time."
                ):

                    result = process_audio_file(
                        audio_file
                    )

                if result:

                    st.session_state[
                        "last_result"
                    ] = result

                    st.rerun()

    # ========================================================
    # RECORD AUDIO
    # ========================================================

    elif input_mode == "🎙️ Record Audio":

        st.markdown(
            "### 🎙️ Record Prescription"
        )

        st.caption(
            "Click the microphone button, "
            "speak your prescription clearly, "
            "then stop the recording."
        )

        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            key="voice_recorder"
        )

        if audio_bytes:

            st.audio(
                audio_bytes,
                format="audio/wav"
            )

            st.success(
                f"✅ Recording ready "
                f"({len(audio_bytes)} bytes)"
            )

            if st.button(
                "🔄 Process Recording",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Processing recording... "
                    "Whisper + prescription extraction "
                    "may take some time."
                ):

                    encoded_audio = (
                        base64
                        .b64encode(audio_bytes)
                        .decode("utf-8")
                    )

                    result = process_base64_audio(
                        encoded_audio
                    )

                if result:

                    st.session_state[
                        "last_result"
                    ] = result

                    st.rerun()

    # ========================================================
    # TEXT INPUT
    # ========================================================

    elif input_mode == "⌨️ Text Input":

        text_input = st.text_area(
            "Type or paste prescription text",

            placeholder=(
                "Example: "
                "Aspirin 100 mg tablet, "
                "morning and evening after food "
                "for 2 days."
            ),

            height=150
        )

        if st.button(
            "🔄 Process Text",
            type="primary",
            use_container_width=True
        ):

            if text_input.strip():

                with st.spinner(
                    "Extracting prescription..."
                ):

                    result = process_text(
                        text_input
                    )

                if result:

                    # For text input, keep the entered
                    # text available in the same Raw
                    # Transcript section.
                    if not result.get(
                        "rawTranscript"
                    ):

                        result[
                            "rawTranscript"
                        ] = text_input.strip()

                    st.session_state[
                        "last_result"
                    ] = result

                    st.rerun()

            else:

                st.warning(
                    "Please enter some text."
                )

    # ========================================================
    # RESULTS
    # ========================================================

    if "last_result" in st.session_state:

        st.divider()

        st.markdown(
            "## 📤 Results"
        )

        render_results(
            st.session_state[
                "last_result"
            ]
        )

        # ====================================================
        # ACTION BUTTONS
        # ====================================================

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # PROCESS ANOTHER
        # ----------------------------------------------------

        with col1:

            if st.button(
                "🔄 Process Another",
                use_container_width=True
            ):

                st.session_state.pop(
                    "last_result",
                    None
                )

                st.rerun()

        # ----------------------------------------------------
        # COPY JSON
        # ----------------------------------------------------

        with col2:

            if st.button(
                "📋 Copy JSON",
                use_container_width=True
            ):

                result = st.session_state[
                    "last_result"
                ]

                prescription = get_prescription(
                    result
                )

                final_output = {
                    "prescription":
                        prescription
                }

                json_string = json.dumps(
                    final_output,
                    indent=2
                )

                st.code(
                    json_string,
                    language="json"
                )

                st.toast(
                    "JSON displayed above."
                )

        # ----------------------------------------------------
        # SUBMIT
        # ----------------------------------------------------

        with col3:

            if st.button(
                "✅ Submit to Medhant Lite",
                use_container_width=True
            ):

                st.success(
                    "Prescription submitted! "
                    "(Demo mode - integrate with your API)"
                )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()