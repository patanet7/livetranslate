import type { ConnectRequest, ConnectResponse, FirefliesSession } from '$lib/types';
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

		getTranslationConfig: () => api.get('/fireflies/translation-config'),

		updateTranslationConfig: (config: Record<string, unknown>) =>
			api.put('/fireflies/translation-config', config)
	};
}
