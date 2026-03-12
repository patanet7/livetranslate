// Connections API client for dashboard server-side loads.
//
// NOTE: This uses createApi() which imports $env/static/private, so it can ONLY
// be used in +page.server.ts files, NOT in client-side +page.svelte.
// The +page.svelte uses direct fetch() calls to avoid this boundary.
//
import { createApi } from './orchestration';

export interface AIConnection {
	id: string;
	name: string;
	engine: 'ollama' | 'openai' | 'anthropic' | 'openai_compatible';
	url: string;
	has_api_key: boolean;
	prefix: string;
	enabled: boolean;
	context_length: number | null;
	timeout_ms: number;
	max_retries: number;
	priority: number;
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

export interface FeaturePreference {
	active_model: string;
	fallback_model: string;
	temperature: number;
	max_tokens: number;
}

export function connectionsApi(fetch: typeof globalThis.fetch) {
	const api = createApi(fetch);
	return {
		list: (enabledOnly = false) =>
			api.get<AIConnection[]>(`/api/connections?enabled_only=${enabledOnly}`),
		aggregateModels: () =>
			api.post<AggregateModelsResponse>('/api/connections/aggregate-models'),
		getPreferences: () =>
			api.get<Record<string, FeaturePreference>>('/api/connections/preferences/all')
	};
}
