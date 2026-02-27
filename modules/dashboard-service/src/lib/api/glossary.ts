import type { Glossary, GlossaryEntry } from '$lib/types';
import { createApi } from './orchestration';

export function glossaryApi(fetch: typeof globalThis.fetch) {
	const api = createApi(fetch);
	return {
		list: (params?: { domain?: string; source_language?: string; active_only?: boolean }) => {
			const query = new URLSearchParams();
			if (params?.domain) query.set('domain', params.domain);
			if (params?.source_language) query.set('source_language', params.source_language);
			if (params?.active_only !== undefined)
				query.set('active_only', String(params.active_only));
			const qs = query.toString();
			return api.get<Glossary[]>(`/api/glossaries${qs ? '?' + qs : ''}`);
		},

		get: (glossaryId: string) => api.get<Glossary>(`/api/glossaries/${glossaryId}`),

		create: (
			glossary: Omit<Glossary, 'glossary_id' | 'entry_count' | 'created_at' | 'updated_at' | 'is_active'>
		) => api.post<Glossary>('/api/glossaries', glossary),

		update: (glossaryId: string, patch: Partial<Glossary>) =>
			api.patch<Glossary>(`/api/glossaries/${glossaryId}`, patch),

		delete: (glossaryId: string) => api.del(`/api/glossaries/${glossaryId}`),

		listEntries: (glossaryId: string, targetLanguage?: string) => {
			const qs = targetLanguage ? `?target_language=${targetLanguage}` : '';
			return api.get<GlossaryEntry[]>(`/api/glossaries/${glossaryId}/entries${qs}`);
		},

		createEntry: (
			glossaryId: string,
			entry: Omit<GlossaryEntry, 'entry_id' | 'glossary_id' | 'created_at' | 'updated_at'>
		) => api.post<GlossaryEntry>(`/api/glossaries/${glossaryId}/entries`, entry),

		updateEntry: (glossaryId: string, entryId: string, patch: Partial<GlossaryEntry>) =>
			api.patch<GlossaryEntry>(`/api/glossaries/${glossaryId}/entries/${entryId}`, patch),

		deleteEntry: (glossaryId: string, entryId: string) =>
			api.del(`/api/glossaries/${glossaryId}/entries/${entryId}`)
	};
}
