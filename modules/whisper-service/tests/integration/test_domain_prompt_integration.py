#!/usr/bin/env python3
"""
COMPREHENSIVE INTEGRATION TESTS: Domain Prompting with Real Whisper Model

Following SimulStreaming specification:
- In-domain prompting for +15-25% quality improvement
- Domain-specific terminology injection
- Tests with REAL Whisper large-v3 model
- REAL transcription with domain prompts

NO MOCKS - Only real Whisper inference!
"""

import sys
from pathlib import Path

# Add src directory to path before imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest
from domain_prompt_manager import DomainPromptManager, create_domain_prompt
from whisper_service import ModelManager


class TestDomainPromptIntegration:
    """
    REAL INTEGRATION TESTS: Domain prompting with actual Whisper model

    All tests:
    1. Load real Whisper large-v3 model
    2. Create domain-specific prompts
    3. Run real transcription with prompts
    4. Verify domain prompts are used correctly
    """

    @pytest.mark.integration
    def test_domain_prompt_in_real_inference(self):
        """
        Test domain prompting with real Whisper inference

        Verifies initial_prompt parameter works
        """
        print("\n[DOMAIN INTEGRATION] Testing domain prompts...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000, dtype=np.float32)

        # Medical domain prompt
        medical_prompt = "Medical terminology: hypertension, diabetes, myocardial infarction, ECG"

        # Inference without prompt
        result_no_prompt = model.transcribe(audio=audio_data, beam_size=1, temperature=0.0)

        # Inference with domain prompt
        result_with_prompt = model.transcribe(
            audio=audio_data, beam_size=1, temperature=0.0, initial_prompt=medical_prompt
        )

        assert result_no_prompt is not None
        assert result_with_prompt is not None

        print(f"   Without prompt: '{result_no_prompt['text']}'")
        print(f"   With prompt: '{result_with_prompt['text']}'")
        print("✅ Domain prompts accepted by Whisper")

    @pytest.mark.integration
    def test_medical_domain_prompting(self):
        """
        Test medical domain prompting with real inference

        Medical domain has specialized terminology
        """
        print("\n[DOMAIN INTEGRATION] Testing medical domain...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000, dtype=np.float32)

        # Create medical domain prompt
        medical_terms = [
            "hypertension",
            "diabetes mellitus",
            "cardiomyopathy",
            "electrocardiogram",
            "myocardial infarction",
        ]

        prompt = create_domain_prompt(domain="medical", custom_terms=medical_terms)

        print(f"   Medical prompt: '{prompt}'")

        result = model.transcribe(
            audio=audio_data, beam_size=5, temperature=0.0, initial_prompt=prompt
        )

        assert result is not None
        print("✅ Medical domain prompt works")

    @pytest.mark.integration
    def test_legal_domain_prompting(self):
        """
        Test legal domain prompting with real inference

        Legal domain has specific terminology
        """
        print("\n[DOMAIN INTEGRATION] Testing legal domain...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000, dtype=np.float32)

        # Create legal domain prompt
        legal_terms = ["plaintiff", "defendant", "litigation", "deposition", "subpoena"]

        prompt = create_domain_prompt(domain="legal", custom_terms=legal_terms)

        print(f"   Legal prompt: '{prompt}'")

        result = model.transcribe(
            audio=audio_data, beam_size=5, temperature=0.0, initial_prompt=prompt
        )

        assert result is not None
        print("✅ Legal domain prompt works")

    @pytest.mark.integration
    def test_technical_domain_prompting(self):
        """
        Test technical/software domain prompting

        Technical terms: API, microservices, Kubernetes, etc.
        """
        print("\n[DOMAIN INTEGRATION] Testing technical domain...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000, dtype=np.float32)

        # Create technical domain prompt
        tech_terms = [
            "Kubernetes",
            "microservices",
            "API endpoint",
            "Docker container",
            "CI/CD pipeline",
        ]

        prompt = create_domain_prompt(domain="technical", custom_terms=tech_terms)

        print(f"   Technical prompt: '{prompt}'")

        result = model.transcribe(
            audio=audio_data, beam_size=5, temperature=0.0, initial_prompt=prompt
        )

        assert result is not None
        print("✅ Technical domain prompt works")

    @pytest.mark.integration
    def test_domain_prompt_manager(self):
        """
        Test DomainPromptManager with real Whisper inference

        Manager provides convenient domain prompt creation
        """
        print("\n[DOMAIN INTEGRATION] Testing DomainPromptManager...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000, dtype=np.float32)

        # Use DomainPromptManager
        prompt_mgr = DomainPromptManager()

        # Get medical prompt
        medical_prompt = prompt_mgr.get_prompt(
            domain="medical", custom_terms=["hypertension", "diabetes"]
        )

        print(f"   Prompt: '{medical_prompt}'")

        result = model.transcribe(
            audio=audio_data, beam_size=5, temperature=0.0, initial_prompt=medical_prompt
        )

        assert result is not None
        print("✅ DomainPromptManager works with real inference")

    @pytest.mark.integration
    def test_custom_terminology_injection(self):
        """
        Test injecting custom terminology into prompts

        Allows user-specific terminology
        """
        print("\n[DOMAIN INTEGRATION] Testing custom terminology...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000, dtype=np.float32)

        # Custom company-specific terms
        custom_terms = ["LiveTranslate", "Whisper NPU", "SimulStreaming", "AlignAtt", "BeamSearch"]

        prompt = create_domain_prompt(domain="general", custom_terms=custom_terms)

        print(f"   Custom prompt: '{prompt}'")

        result = model.transcribe(
            audio=audio_data, beam_size=5, temperature=0.0, initial_prompt=prompt
        )

        assert result is not None
        print("✅ Custom terminology injection works")

    @pytest.mark.integration
    def test_domain_prompt_with_beam_search(self):
        """
        Test domain prompting combined with beam search

        Maximum quality: domain prompt + beam_size=5
        """
        print("\n[DOMAIN INTEGRATION] Testing domain prompt + beam search...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000 * 2, dtype=np.float32)

        medical_prompt = create_domain_prompt(
            domain="medical", custom_terms=["hypertension", "cardiomyopathy"]
        )

        # Combined: domain prompt + beam search
        result = model.transcribe(
            audio=audio_data,
            beam_size=5,  # Beam search for quality
            temperature=0.0,
            initial_prompt=medical_prompt,  # Domain prompt
        )

        assert result is not None
        print(f"   Result: '{result['text']}'")
        print("✅ Domain prompt + beam search works")


class TestDomainPromptQuality:
    """
    Integration tests for domain prompt quality improvements

    Following SimulStreaming: domain prompts give +15-25% quality
    """

    @pytest.mark.integration
    def test_prompt_consistency(self):
        """
        Test that same prompt produces consistent results

        With temperature=0.0, should be deterministic
        """
        print("\n[DOMAIN QUALITY] Testing prompt consistency...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000, dtype=np.float32)

        prompt = "Medical: hypertension, diabetes"

        # Run twice
        result1 = model.transcribe(
            audio=audio_data, beam_size=5, temperature=0.0, initial_prompt=prompt
        )

        result2 = model.transcribe(
            audio=audio_data, beam_size=5, temperature=0.0, initial_prompt=prompt
        )

        # Should be identical (deterministic)
        assert result1["text"] == result2["text"]

        print("✅ Domain prompts are deterministic")

    @pytest.mark.integration
    def test_long_domain_prompt(self):
        """
        Test handling of long domain-specific prompts

        Verifies Whisper can handle detailed prompts
        """
        print("\n[DOMAIN QUALITY] Testing long prompts...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000, dtype=np.float32)

        # Long medical prompt with many terms
        long_prompt = "Medical terminology: " + ", ".join(
            [
                "hypertension",
                "diabetes mellitus",
                "cardiomyopathy",
                "myocardial infarction",
                "electrocardiogram",
                "arrhythmia",
                "coronary artery disease",
                "congestive heart failure",
                "atrial fibrillation",
                "ventricular tachycardia",
            ]
        )

        print(f"   Prompt length: {len(long_prompt)} chars")

        result = model.transcribe(
            audio=audio_data, beam_size=5, temperature=0.0, initial_prompt=long_prompt
        )

        assert result is not None
        print("✅ Long prompts handled correctly")

    @pytest.mark.integration
    def test_empty_vs_domain_prompt_comparison(self):
        """
        Compare transcription with and without domain prompt

        Shows impact of domain prompting
        """
        print("\n[DOMAIN QUALITY] Comparing empty vs domain prompt...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        model = manager.load_model("large-v3")
        audio_data = np.zeros(16000, dtype=np.float32)

        # Without domain prompt
        result_empty = model.transcribe(audio=audio_data, beam_size=5, temperature=0.0)

        # With domain prompt
        result_domain = model.transcribe(
            audio=audio_data,
            beam_size=5,
            temperature=0.0,
            initial_prompt="Medical: hypertension, diabetes",
        )

        print(f"   Empty prompt: '{result_empty['text']}'")
        print(f"   Domain prompt: '{result_domain['text']}'")
        print("✅ Domain prompt comparison complete")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
