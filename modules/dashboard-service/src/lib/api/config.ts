import type {
	UserSettings,
	TranslationSettings,
	UiConfig,
	SystemConfigUpdate,
	TranslationModelsResponse,
	TranslationHealth
} from '$lib/types';
import { createApi } from './orchestration';

export function configApi(fetch: typeof globalThis.fetch) {
	const api = createApi(fetch);
	return {
		getUserSettings: () => api.get<UserSettings>('/api/settings/user'),

		updateUserSettings: (settings: Partial<UserSettings>) =>
			api.put<UserSettings>('/api/settings/user', settings),

		getTranslationSettings: () =>
			api.get<TranslationSettings>('/api/settings/translation'),

		saveTranslationSettings: (settings: TranslationSettings) =>
			api.post<{ message: string; config: TranslationSettings }>(
				'/api/settings/translation',
				settings
			),

		testTranslation: (text: string, targetLanguage: string) =>
			api.post('/api/settings/translation/test', { text, target_language: targetLanguage }),

		getUiConfig: () => api.get<UiConfig>('/api/system/ui-config'),

		updateUiConfig: (config: SystemConfigUpdate) =>
			api.put<{ status: string; message: string }>('/api/system/ui-config', config),

		resetUiConfig: () =>
			api.post<{ status: string; message: string }>('/api/system/ui-config/reset'),

		getHealth: () => api.get('/api/system/health'),

		getServices: () => api.get('/api/system/services'),

		getTranslationModels: () =>
			api.get<TranslationModelsResponse>('/api/translation/models'),

		getTranslationHealth: () =>
			api.get<TranslationHealth>('/api/translation/health'),

		switchTranslationModel: (model: string) =>
			api.post<{ message: string }>('/api/translation/model', { model }),

		getAudioProcessing: () =>
			api.get<Record<string, unknown>>('/api/settings/audio-processing'),

		saveAudioProcessing: (config: Record<string, unknown>) =>
			api.post<{ message: string; config: Record<string, unknown> }>(
				'/api/settings/audio-processing',
				config
			),

		getChunking: () =>
			api.get<Record<string, unknown>>('/api/settings/chunking'),

		saveChunking: (config: Record<string, unknown>) =>
			api.post<{ message: string; config: Record<string, unknown> }>(
				'/api/settings/chunking',
				config
			),

		getPrompts: () =>
			api.get<{ success: boolean; prompts: unknown[] }>('/api/settings/prompts'),

		savePrompt: (prompt: Record<string, unknown>) =>
			api.post<{ success: boolean; prompt_id: string }>('/api/settings/prompts', prompt),

		updatePrompt: (id: string, updates: Record<string, unknown>) =>
			api.put<{ success: boolean }>(`/api/settings/prompts/${id}`, updates)
	};
}
