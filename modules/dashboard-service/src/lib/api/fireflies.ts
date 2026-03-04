import type {
	ConnectRequest,
	ConnectResponse,
	FirefliesSession,
	TranslationConfig
} from '$lib/types';
import { createApi } from './orchestration';

export function firefliesApi(fetch: typeof globalThis.fetch) {
	const api = createApi(fetch);
	return {
		connect: (req: ConnectRequest) =>
			api.post<ConnectResponse>('/fireflies/connect', req),

		disconnect: (sessionId: string) =>
			api.post<{ success: boolean; message: string }>('/fireflies/disconnect', {
				session_id: sessionId
			}),

		listSessions: () => api.get<FirefliesSession[]>('/fireflies/sessions'),

		getSession: (sessionId: string) =>
			api.get<FirefliesSession>(`/fireflies/sessions/${sessionId}`),

		setDisplayMode: (sessionId: string, mode: string) =>
			api.put(`/fireflies/sessions/${sessionId}/display-mode`, { mode }),

		pause: (sessionId: string) =>
			api.post(`/fireflies/sessions/${sessionId}/pause`),

		resume: (sessionId: string) =>
			api.post(`/fireflies/sessions/${sessionId}/resume`),

		getTranslationConfig: () =>
			api.get<TranslationConfig>('/fireflies/translation-config'),

		updateTranslationConfig: (config: Record<string, unknown>) =>
			api.put('/fireflies/translation-config', config),

		syncAll: (apiKey?: string) =>
			api.post<{
				synced: number;
				skipped: number;
				errors: number;
				total: number;
				api_calls_used: number;
			}>('/fireflies/sync-all', apiKey ? { api_key: apiKey } : {}),

		getSyncStatus: () =>
			api.get<{
				last_sync_at: string | null;
				updated_at?: string | null;
				message?: string;
			}>('/fireflies/sync-status'),

		inviteBot: (meetingLink: string, title?: string, duration?: number) =>
			api.post<{ success: boolean; message: string }>('/fireflies/invite-bot', {
				meeting_link: meetingLink,
				title,
				duration: duration ?? 60
			}),

		listTranscripts: (limit = 20, skip = 0) =>
			api.post<{ success: boolean; transcripts: Record<string, unknown>[]; count: number }>(
				'/fireflies/transcripts',
				{ limit, skip }
			)
	};
}
