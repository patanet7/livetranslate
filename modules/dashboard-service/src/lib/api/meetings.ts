// modules/dashboard-service/src/lib/api/meetings.ts

import type {
	MeetingListResponse,
	MeetingSearchResponse,
	Meeting,
	MeetingTranscriptResponse,
	MeetingInsightsResponse,
	MeetingSpeakersResponse,
	InsightGenerateResponse
} from '$lib/types';
import { createApi } from './orchestration';

export function meetingsApi(fetch: typeof globalThis.fetch) {
	const api = createApi(fetch);

	return {
		list: (limit = 50, offset = 0) =>
			api.get<MeetingListResponse>(`/api/meetings/?limit=${limit}&offset=${offset}`),

		search: (q: string, limit = 20) =>
			api.get<MeetingSearchResponse>(
				`/api/meetings/search?q=${encodeURIComponent(q)}&limit=${limit}`
			),

		get: (id: string) => api.get<{ meeting: Meeting }>(`/api/meetings/${id}`),

		getTranscript: (id: string) =>
			api.get<MeetingTranscriptResponse>(`/api/meetings/${id}/transcript`),

		getInsights: (id: string) =>
			api.get<MeetingInsightsResponse>(`/api/meetings/${id}/insights`),

		getSpeakers: (id: string) =>
			api.get<MeetingSpeakersResponse>(`/api/meetings/${id}/speakers`),

		generateInsights: (id: string, types?: string[]) =>
			api.post<InsightGenerateResponse>(`/api/meetings/${id}/insights/generate`, {
				insight_types: types ?? ['summary', 'action_items', 'keywords']
			}),

		syncNow: (id: string) =>
			api.post<{ success: boolean }>(`/api/meetings/${id}/sync`)
	};
}
