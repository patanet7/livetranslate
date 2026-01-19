/**
 * Language Constants
 *
 * Centralized language definitions and utilities.
 * Consolidates 5 duplicate definitions across:
 * - CreateBotModal.tsx
 * - BotSpawner.tsx
 * - BotSettings.tsx
 * - TranslationSettings.tsx
 */

export const SUPPORTED_LANGUAGES = [
  { code: "en", name: "English" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "zh", name: "Chinese" },
  { code: "ru", name: "Russian" },
  { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" },
  { code: "tr", name: "Turkish" },
  { code: "pl", name: "Polish" },
  { code: "nl", name: "Dutch" },
  { code: "sv", name: "Swedish" },
  { code: "da", name: "Danish" },
  { code: "no", name: "Norwegian" },
] as const;

export type LanguageCode = (typeof SUPPORTED_LANGUAGES)[number]["code"];
export type Language = (typeof SUPPORTED_LANGUAGES)[number];

/**
 * Get language name from language code
 * @param code - Language code (e.g., 'en', 'es')
 * @returns Language name or the code itself if not found
 */
export function getLanguageName(code: string): string {
  return SUPPORTED_LANGUAGES.find((lang) => lang.code === code)?.name || code;
}

/**
 * Validate if a language code is supported
 * @param code - Language code to validate
 * @returns true if supported, false otherwise
 */
export function isLanguageSupported(code: string): boolean {
  return SUPPORTED_LANGUAGES.some((lang) => lang.code === code);
}

/**
 * Get all supported language codes
 * @returns Array of language codes
 */
export function getLanguageCodes(): string[] {
  return SUPPORTED_LANGUAGES.map((lang) => lang.code);
}
