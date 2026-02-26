import type { UserSettings, TranslationSettings, UiConfig } from '$lib/types';
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

		getHealth: () => api.get('/api/system/health'),

		getServices: () => api.get('/api/system/services')
	};
}
