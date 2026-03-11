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
	languages: Array<{ code: string; name: string; native: string; rtl?: boolean }>;
	language_codes: string[];
	domains: Array<{ value: string; label: string; description?: string }>;
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

export interface DomainItem {
	value: string;
	label: string;
	description?: string;
}

export interface SystemConfigUpdate {
	enabled_languages?: string[];
	custom_domains?: DomainItem[];
	disabled_domains?: string[];
	defaults?: Record<string, unknown>;
}

export interface TranslationModel {
	name: string;
	model: string;
	display_name: string;
	backend: string;
}

export interface TranslationModelsResponse {
	models: TranslationModel[];
}

export interface TranslationHealth {
	backend: string;
	device: string;
	available_backends: string[];
	model: string;
	status: string;
}

// --- Translation Connections (Open WebUI-style multi-backend) ---

export interface TranslationConnection {
	id: string;
	name: string;
	engine: 'ollama' | 'vllm' | 'triton' | 'openai_compatible';
	url: string;
	prefix: string;
	api_key: string;
	enabled: boolean;
	timeout_ms: number;
	max_retries: number;
}

export interface VerifyConnectionRequest {
	url: string;
	engine: string;
	api_key?: string;
}

export interface VerifyConnectionResponse {
	status: 'connected' | 'error';
	message: string;
	version?: string;
	models?: string[];
	latency_ms?: number;
}

export interface AggregatedModel {
	id: string;
	name: string;
	connection_id: string;
	connection_name: string;
	prefix: string;
	engine: string;
}

export interface AggregateModelsResponse {
	models: AggregatedModel[];
	errors: Array<{ connection_id: string; connection_name: string; message: string }>;
}

export interface FullTranslationConfig {
	connections: TranslationConnection[];
	active_model: string;
	fallback_model: string;
	service: Record<string, unknown>;
	languages: Record<string, unknown>;
	quality: Record<string, unknown>;
	model: Record<string, unknown>;
	realtime?: Record<string, unknown>;
	caching?: Record<string, unknown>;
}

// --- Glossary ---

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
