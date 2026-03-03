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
			api.get<MeetingListResponse>(`/meetings/?limit=${limit}&offset=${offset}`),

		search: (q: string, limit = 20) =>
			api.get<MeetingSearchResponse>(
				`/meetings/search?q=${encodeURIComponent(q)}&limit=${limit}`
			),

		get: (id: string) => api.get<{ meeting: Meeting }>(`/meetings/${id}`),

		getTranscript: (id: string) =>
			api.get<MeetingTranscriptResponse>(`/meetings/${id}/transcript`),

		getInsights: (id: string) =>
			api.get<MeetingInsightsResponse>(`/meetings/${id}/insights`),

		getSpeakers: (id: string) =>
			api.get<MeetingSpeakersResponse>(`/meetings/${id}/speakers`),

		generateInsights: (id: string, types?: string[]) =>
			api.post<InsightGenerateResponse>(`/meetings/${id}/insights/generate`, {
				insight_types: types ?? ['summary', 'action_items', 'keywords']
			}),

		syncNow: (id: string) =>
			api.post<{ success: boolean }>(`/meetings/${id}/sync`)
	};
}
