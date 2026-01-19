/**
 * Centralized Translation Configuration
 *
 * This file contains all default translation settings used across the application.
 * Modify these values to change defaults globally.
 */

export const DEFAULT_TARGET_LANGUAGES = ["en", "zh"] as const;

export const AVAILABLE_LANGUAGES = [
  { code: "en", name: "English", flag: "🇺🇸" },
  { code: "zh", name: "Chinese", flag: "🇨🇳" },
  { code: "es", name: "Spanish", flag: "🇪🇸" },
  { code: "fr", name: "French", flag: "🇫🇷" },
  { code: "de", name: "German", flag: "🇩🇪" },
  { code: "ja", name: "Japanese", flag: "🇯🇵" },
  { code: "ko", name: "Korean", flag: "🇰🇷" },
  { code: "pt", name: "Portuguese", flag: "🇵🇹" },
  { code: "ru", name: "Russian", flag: "🇷🇺" },
  { code: "it", name: "Italian", flag: "🇮🇹" },
  { code: "ar", name: "Arabic", flag: "🇸🇦" },
  { code: "hi", name: "Hindi", flag: "🇮🇳" },
] as const;

export const DEFAULT_SOURCE_LANGUAGE = "auto";
export const DEFAULT_CONFIDENCE_THRESHOLD = 0.8;
export const DEFAULT_TRANSLATION_QUALITY = "balanced" as const;

export type TargetLanguage = (typeof DEFAULT_TARGET_LANGUAGES)[number];
export type AvailableLanguage = (typeof AVAILABLE_LANGUAGES)[number]["code"];
export type TranslationQuality = "fast" | "balanced" | "quality";
