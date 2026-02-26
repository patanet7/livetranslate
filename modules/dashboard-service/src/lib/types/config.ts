export interface UserSettings {
	user_id: string;
	theme: 'dark' | 'light';
	language: string;
	notifications: boolean;
	audio_auto_start: boolean;
	default_translation_language: string;
	transcription_model: string;
	custom_settings: Record<string, unknown>;
	updated_at: string;
}

export interface TranslationConfig {
	backend: 'ollama' | 'vllm' | 'openai' | 'groq';
	model: string;
	base_url: string;
	target_language: string;
	temperature: number;
	max_tokens: number;
}

export interface TranslationSettings {
	enabled: boolean;
	default_model: string;
	default_target_language: string;
}

export interface UiConfig {
	languages: Array<{ code: string; name: string; nativeName: string }>;
	language_codes: string[];
	domains: string[];
	defaults: Record<string, unknown>;
	translation_models: Array<{
		name: string;
		backend: string;
		languages: string[];
		default: boolean;
	}>;
	translation_service_available: boolean;
	config_version: string;
}

export interface Glossary {
	glossary_id: string;
	name: string;
	description: string;
	domain: string;
	source_language: string;
	target_languages: string[];
	is_active: boolean;
	is_default: boolean;
	entry_count: number;
	created_at: string;
	updated_at: string;
}

export interface GlossaryEntry {
	entry_id: string;
	glossary_id: string;
	source_term: string;
	translations: Record<string, string>;
	context: string;
	notes: string;
	case_sensitive: boolean;
	match_whole_word: boolean;
	priority: number;
	created_at: string;
	updated_at: string;
}
