import type { TranslateRequest, TranslateResponse } from '$lib/types';
import { createApi } from './orchestration';

export function translationApi(fetch: typeof globalThis.fetch) {
	const api = createApi(fetch);
	return {
		translate: (req: TranslateRequest) =>
			api.post<TranslateResponse>('/api/translation/translate', req),

		batchTranslate: (requests: TranslateRequest[]) =>
			api.post<TranslateResponse[]>('/api/translation/batch', { requests }),

		detectLanguage: (text: string) =>
			api.post<{ detected_language: string; confidence: number }>(
				'/api/translation/detect',
				{ text }
			),

		getModels: () =>
			api.get<{
				models: Array<{
					name: string;
					backend: string;
					languages: string[];
					default: boolean;
				}>;
			}>('/api/translation/models')
	};
}
