"""
System-Wide Constants

Centralized configuration constants used across all services.
This is the SINGLE SOURCE OF TRUTH for:
- Supported languages
- Glossary domains
- Translation backends
- Prompt template metadata

DO NOT duplicate these values elsewhere - always import from here.
"""

from typing import Any

# =============================================================================
# SUPPORTED LANGUAGES
# =============================================================================
# Complete list of languages supported by the translation system.
# Each entry includes:
#   - code: ISO 639-1 language code
#   - name: English name
#   - native: Native script name
#   - rtl: Right-to-left (for Arabic, Hebrew, etc.)

SUPPORTED_LANGUAGES: list[dict[str, Any]] = [
    {"code": "en", "name": "English", "native": "English", "rtl": False},
    {"code": "es", "name": "Spanish", "native": "Español", "rtl": False},
    {"code": "fr", "name": "French", "native": "Français", "rtl": False},
    {"code": "de", "name": "German", "native": "Deutsch", "rtl": False},
    {"code": "pt", "name": "Portuguese", "native": "Português", "rtl": False},
    {"code": "it", "name": "Italian", "native": "Italiano", "rtl": False},
    {"code": "zh", "name": "Chinese", "native": "中文", "rtl": False},
    {"code": "ja", "name": "Japanese", "native": "日本語", "rtl": False},
    {"code": "ko", "name": "Korean", "native": "한국어", "rtl": False},
    {"code": "ru", "name": "Russian", "native": "Русский", "rtl": False},
    {"code": "ar", "name": "Arabic", "native": "العربية", "rtl": True},
    {"code": "hi", "name": "Hindi", "native": "हिन्दी", "rtl": False},
    {"code": "tr", "name": "Turkish", "native": "Türkçe", "rtl": False},
    {"code": "pl", "name": "Polish", "native": "Polski", "rtl": False},
    {"code": "nl", "name": "Dutch", "native": "Nederlands", "rtl": False},
    {"code": "sv", "name": "Swedish", "native": "Svenska", "rtl": False},
    {"code": "da", "name": "Danish", "native": "Dansk", "rtl": False},
    {"code": "no", "name": "Norwegian", "native": "Norsk", "rtl": False},
    {"code": "fi", "name": "Finnish", "native": "Suomi", "rtl": False},
    {"code": "el", "name": "Greek", "native": "Ελληνικά", "rtl": False},
    {"code": "he", "name": "Hebrew", "native": "עברית", "rtl": True},
    {"code": "th", "name": "Thai", "native": "ไทย", "rtl": False},
    {"code": "vi", "name": "Vietnamese", "native": "Tiếng Việt", "rtl": False},
    {"code": "id", "name": "Indonesian", "native": "Bahasa Indonesia", "rtl": False},
    {"code": "ms", "name": "Malay", "native": "Bahasa Melayu", "rtl": False},
    {"code": "uk", "name": "Ukrainian", "native": "Українська", "rtl": False},
    {"code": "cs", "name": "Czech", "native": "Čeština", "rtl": False},
    {"code": "ro", "name": "Romanian", "native": "Română", "rtl": False},
    {"code": "hu", "name": "Hungarian", "native": "Magyar", "rtl": False},
    {"code": "bg", "name": "Bulgarian", "native": "Български", "rtl": False},
    {"code": "hr", "name": "Croatian", "native": "Hrvatski", "rtl": False},
    {"code": "sk", "name": "Slovak", "native": "Slovenčina", "rtl": False},
    {"code": "sl", "name": "Slovenian", "native": "Slovenščina", "rtl": False},
    {"code": "et", "name": "Estonian", "native": "Eesti", "rtl": False},
    {"code": "lv", "name": "Latvian", "native": "Latviešu", "rtl": False},
    {"code": "lt", "name": "Lithuanian", "native": "Lietuvių", "rtl": False},
    {"code": "fa", "name": "Persian", "native": "فارسی", "rtl": True},
    {"code": "ur", "name": "Urdu", "native": "اردو", "rtl": True},
    {"code": "bn", "name": "Bengali", "native": "বাংলা", "rtl": False},
    {"code": "ta", "name": "Tamil", "native": "தமிழ்", "rtl": False},
    {"code": "te", "name": "Telugu", "native": "తెలుగు", "rtl": False},
    {"code": "mr", "name": "Marathi", "native": "मराठी", "rtl": False},
    {"code": "gu", "name": "Gujarati", "native": "ગુજરાતી", "rtl": False},
    {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ", "rtl": False},
    {"code": "ml", "name": "Malayalam", "native": "മലയാളം", "rtl": False},
    {"code": "pa", "name": "Punjabi", "native": "ਪੰਜਾਬੀ", "rtl": False},
    {"code": "sw", "name": "Swahili", "native": "Kiswahili", "rtl": False},
    {"code": "tl", "name": "Filipino", "native": "Filipino", "rtl": False},
]

# Quick lookup by code
LANGUAGE_CODE_MAP: dict[str, dict[str, Any]] = {lang["code"]: lang for lang in SUPPORTED_LANGUAGES}

# List of just codes for validation
VALID_LANGUAGE_CODES: list[str] = [lang["code"] for lang in SUPPORTED_LANGUAGES]


# =============================================================================
# GLOSSARY DOMAINS
# =============================================================================
# Domain categories for glossary organization.
# Empty value ("") represents "General" / no specific domain.

GLOSSARY_DOMAINS: list[dict[str, str]] = [
    {"value": "", "label": "General", "description": "General purpose terminology"},
    {"value": "medical", "label": "Medical", "description": "Healthcare and medical terminology"},
    {"value": "legal", "label": "Legal", "description": "Legal and regulatory terminology"},
    {"value": "technology", "label": "Technology", "description": "IT and software terminology"},
    {"value": "business", "label": "Business", "description": "Business and commerce terminology"},
    {"value": "finance", "label": "Finance", "description": "Financial and banking terminology"},
    {"value": "academic", "label": "Academic", "description": "Academic and research terminology"},
    {
        "value": "marketing",
        "label": "Marketing",
        "description": "Marketing and advertising terminology",
    },
    {
        "value": "engineering",
        "label": "Engineering",
        "description": "Engineering and technical terminology",
    },
    {
        "value": "scientific",
        "label": "Scientific",
        "description": "Scientific research terminology",
    },
    {
        "value": "pharmaceutical",
        "label": "Pharmaceutical",
        "description": "Drug and pharmaceutical terminology",
    },
    {
        "value": "manufacturing",
        "label": "Manufacturing",
        "description": "Manufacturing and production terminology",
    },
]

# List of just domain values
VALID_DOMAINS: list[str] = [d["value"] for d in GLOSSARY_DOMAINS]


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
# System defaults - can be overridden by environment or user settings

DEFAULT_CONFIG: dict[str, Any] = {
    "default_source_language": "en",
    "default_target_languages": ["en"],
    "default_translation_model": "default",
    "auto_detect_language": True,
    "confidence_threshold": 0.8,
    "context_window_size": 3,
    "max_buffer_words": 50,
    "pause_threshold_ms": 500,
}


# =============================================================================
# PROMPT TEMPLATE VARIABLES
# =============================================================================
# Available variables for prompt templates

PROMPT_TEMPLATE_VARIABLES: list[dict[str, str]] = [
    {
        "name": "target_language",
        "description": "Target language name (e.g., 'Spanish')",
        "example": "Spanish",
    },
    {
        "name": "current_sentence",
        "description": "The text to be translated",
        "example": "Hello, how are you?",
    },
    {
        "name": "glossary_section",
        "description": "Formatted glossary terms for the domain",
        "example": "Term definitions:\n- API: Interfaz de programación",
    },
    {
        "name": "context_window",
        "description": "Previous sentences for context",
        "example": "Previous: 'Welcome to the meeting.'",
    },
    {
        "name": "source_language",
        "description": "Detected source language",
        "example": "English",
    },
    {
        "name": "speaker_name",
        "description": "Name of the current speaker",
        "example": "John Doe",
    },
    {
        "name": "domain",
        "description": "Domain/industry context",
        "example": "medical",
    },
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_language_by_code(code: str) -> dict[str, Any] | None:
    """Get language details by ISO code."""
    return LANGUAGE_CODE_MAP.get(code.lower())


def is_valid_language_code(code: str) -> bool:
    """Check if a language code is supported."""
    return code.lower() in VALID_LANGUAGE_CODES


def is_rtl_language(code: str) -> bool:
    """Check if a language is right-to-left."""
    lang = get_language_by_code(code)
    return lang.get("rtl", False) if lang else False


def get_language_display_name(code: str, include_native: bool = True) -> str:
    """Get display name for a language code."""
    lang = get_language_by_code(code)
    if not lang:
        return code.upper()
    name: str = lang["name"]
    if include_native and lang.get("native"):
        native: str = lang["native"]
        return f"{name} ({native})"
    return name
