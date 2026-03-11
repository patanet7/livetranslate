import type {
	UserSettings,
	TranslationSettings,
	UiConfig,
	SystemConfigUpdate,
	TranslationModelsResponse,
	TranslationHealth,
	VerifyConnectionRequest,
	VerifyConnectionResponse,
	AggregateModelsResponse,
	FullTranslationConfig
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

		// --- Translation Connections ---

		verifyConnection: (req: VerifyConnectionRequest) =>
			api.post<VerifyConnectionResponse>('/api/settings/translation/verify-connection', req),

		aggregateModels: () =>
			api.post<AggregateModelsResponse>('/api/settings/translation/aggregate-models'),

		getFullTranslationConfig: () =>
			api.get<FullTranslationConfig>('/api/settings/translation'),

		saveFullTranslationConfig: (config: FullTranslationConfig) =>
			api.post<{ message: string; config: FullTranslationConfig }>(
				'/api/settings/translation',
				config
			)
	};
}
