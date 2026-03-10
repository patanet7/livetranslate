import type { TranslateRequest, TranslateResponse } from '$lib/types';
import { createApi } from './orchestration';

export function translationApi(fetch: typeof globalThis.fetch) {
	const api = createApi(fetch);
	return {
		translate: (req: TranslateRequest) =>
			api.post<TranslateResponse>('/api/translation/translate', req),

		getModels: () =>
			api.get<{
				models: Array<{
					name: string;
					backend: string;
					backend_name?: string;
					languages: string[];
					default: boolean;
				}>;
			}>('/api/translation/models')
	};
}
