import pytest
import json
from app.pipeline import get_pipeline, create_pipeline, PipelineConfig
from app.models.schemas import Prescription, Medicine, ValidationResult


class TestPipeline:
    """Test the voice-to-prescription pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline with spaCy only (no Ollama)."""
        config = PipelineConfig(
            whisper_model="tiny",
            use_ollama_extraction=False,
            use_spacy_fallback=True,
            spacy_model="en_core_web_sm"
        )
        return create_pipeline(config)
    
    def test_simple_prescription(self, pipeline):
        """Test basic single medicine extraction."""
        text = "Paracetamol 500 mg 1-0-1 for 5 days after food"
        result = pipeline.process_text(text)
        
        assert result.success is True
        assert len(result.prescription.medicines) == 1
        
        med = result.prescription.medicines[0]
        assert med.medicineName == "Paracetamol"
        assert med.dosage == "500 mg"
        assert med.frequency == "1-0-1"
        assert med.duration == "5 days"
        assert med.instructions == "After food"
        assert med.route.value == "oral"
    
    def test_multiple_medicines(self, pipeline):
        """Test multiple medicines in one dictation."""
        text = "Metformin 500 mg twice daily for 30 days. Atorvastatin 10 mg once daily at night"
        result = pipeline.process_text(text)
        
        assert len(result.prescription.medicines) == 2
        
        med1 = result.prescription.medicines[0]
        assert med1.medicineName == "Metformin"
        assert med1.frequency == "1-0-1"
        
        med2 = result.prescription.medicines[1]
        assert med2.medicineName == "Atorvastatin"
        assert med2.frequency == "1-0-0"
        assert med2.instructions == "At bedtime"
    
    def test_sos_frequency(self, pipeline):
        """Test SOS/as-needed frequency."""
        text = "Ibuprofen 400 mg SOS for pain after food"
        result = pipeline.process_text(text)
        
        med = result.prescription.medicines[0]
        assert med.frequency == "SOS"
        assert med.instructions == "After food"
    
    def test_inhalation_route(self, pipeline):
        """Test inhalation route detection."""
        text = "Salbutamol inhaler 2 puffs four times daily as needed"
        result = pipeline.process_text(text)
        
        med = result.prescription.medicines[0]
        assert med.route.value == "inhalation"
        assert med.dosage == "2 puffs"
    
    def test_injection_route(self, pipeline):
        """Test injection route detection."""
        text = "Insulin regular 10 units before breakfast"
        result = pipeline.process_text(text)
        
        med = result.prescription.medicines[0]
        assert med.route.value == "injection"
    
    def test_validation_missing_fields(self, pipeline):
        """Test validation catches missing fields."""
        text = "Paracetamol"
        result = pipeline.process_text(text)
        
        assert result.validation.isValid is False or len(result.validation.missingFields) > 0
        assert "dosage" in result.validation.missingFields
        assert "frequency" in result.validation.missingFields
    
    def test_unknown_medicine_warning(self, pipeline):
        """Test unknown medicine generates warning."""
        text = "Unknownmedicine 100 mg once daily"
        result = pipeline.process_text(text)
        
        # Should still extract but with warning
        assert len(result.prescription.medicines) == 1
        warnings = [i for i in result.validation.issues if i.severity == "warning"]
        assert any("unknown" in i.issue.lower() for i in warnings)


class TestNormalizer:
    """Test text normalization."""
    
    def test_frequency_normalization(self):
        from app.pipeline.normalizer import get_normalizer
        normalizer = get_normalizer()
        
        test_cases = [
            ("once daily", "1-0-0"),
            ("twice daily", "1-0-1"),
            ("thrice daily", "1-1-1"),
            ("morning and evening", "1-0-1"),
            ("SOS", "SOS"),
            ("as needed", "SOS"),
            ("BD", "1-0-1"),
            ("TID", "1-1-1"),
        ]
        
        for input_text, expected in test_cases:
            result = normalizer.normalize(input_text)
            assert expected in result["normalized_text"], f"Failed for: {input_text}"
    
    def test_duration_normalization(self):
        from app.pipeline.normalizer import get_normalizer
        normalizer = get_normalizer()
        
        result = normalizer.normalize("for 5 days")
        assert "5 days" in result["normalized_text"]
        
        result = normalizer.normalize("2 weeks")
        assert "2 weeks" in result["normalized_text"]
    
    def test_medicine_name_correction(self):
        from app.pipeline.normalizer import get_normalizer
        normalizer = get_normalizer()
        
        result = normalizer.normalize("parasetamol 500 mg")
        assert "paracetamol" in result["normalized_text"]


class TestValidator:
    """Test prescription validation."""
    
    def test_valid_prescription(self):
        from app.pipeline.validator import get_validator
        from app.models.schemas import Prescription, Medicine, Route
        
        validator = get_validator()
        
        prescription = Prescription(medicines=[
            Medicine(
                medicineName="Paracetamol",
                dosage="500 mg",
                frequency="1-0-1",
                duration="5 days",
                route=Route.ORAL,
                instructions="After food"
            )
        ])
        
        result = validator.validate(prescription)
        assert result.isValid is True
        assert len(result.issues) == 0
    
    def test_missing_required_field(self):
        from app.pipeline.validator import get_validator
        from app.models.schemas import Prescription, Medicine
        
        validator = get_validator()
        
        prescription = Prescription(medicines=[
            Medicine(medicineName="", dosage="500 mg")  # Empty name
        ])
        
        result = validator.validate(prescription)
        assert result.isValid is False
        assert any(i.field == "medicineName" and i.severity == "error" for i in result.issues)
    
    def test_duplicate_medicine_warning(self):
        from app.pipeline.validator import get_validator
        from app.models.schemas import Prescription, Medicine, Route
        
        validator = get_validator()
        
        prescription = Prescription(medicines=[
            Medicine(medicineName="Paracetamol", dosage="500 mg", route=Route.ORAL),
            Medicine(medicineName="Paracetamol", dosage="500 mg", route=Route.ORAL),
        ])
        
        result = validator.validate(prescription)
        assert any(i.severity == "warning" and "duplicate" in i.issue.lower() for i in result.issues)


class TestSchemas:
    """Test Pydantic schemas."""
    
    def test_medicine_schema(self):
        from app.models.schemas import Medicine, Route
        
        med = Medicine(
            medicineName="Paracetamol",
            dosage="500 mg",
            frequency="1-0-1",
            duration="5 days",
            route=Route.ORAL,
            instructions="After food",
            confidence=0.95
        )
        
        assert med.medicineName == "Paracetamol"
        assert med.route == Route.ORAL
        assert med.confidence == 0.95
    
    def test_frequency_normalization_in_schema(self):
        from app.models.schemas import Medicine
        
        # Test that frequency gets normalized
        med = Medicine(medicineName="Test", frequency="BD")
        assert med.frequency == "1-0-1"
        
        med = Medicine(medicineName="Test", frequency="tid")
        assert med.frequency == "1-1-1"
    
    def test_prescription_response(self):
        from app.models.schemas import PrescriptionResponse, ExtractionResult, Prescription, ValidationResult
        
        response = PrescriptionResponse(success=True, data=ExtractionResult(
            prescription=Prescription(),
            validation=ValidationResult(isValid=True),
            processingTimeMs=100
        ))
        
        assert response.success is True
        assert response.data.processingTimeMs == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])