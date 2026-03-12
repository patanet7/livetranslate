import { createApi } from './orchestration';

// ── Types ──────────────────────────────────────────────────────────

export interface Provider {
	name: string;
	configured: boolean;
	healthy: boolean | null;
}

export interface ModelInfo {
	id: string;
	name: string;
	provider: string;
	context_window: number | null;
}

export interface ChatSettings {
	active_model: string;
	temperature: number;
	max_tokens: number;
}

export interface Conversation {
	id: string;
	title: string | null;
	provider: string | null;
	model: string | null;
	message_count: number;
	created_at: string;
	updated_at: string;
}

export interface ToolCallInfo {
	tool_name: string;
	arguments: Record<string, unknown>;
	result: string | null;
}

export interface ChatMessage {
	id: string;
	conversation_id: string;
	role: 'user' | 'assistant' | 'tool' | 'system';
	content: string | null;
	tool_calls: ToolCallInfo[] | null;
	model: string | null;
	provider: string | null;
	tokens_used: number | null;
	created_at: string;
}

// ── API Client ─────────────────────────────────────────────────────

export function chatApi(fetch: typeof globalThis.fetch) {
	const api = createApi(fetch);

	return {
		getProviders: () => api.get<Provider[]>('/api/chat/providers'),

		getModels: (provider: string) =>
			api.get<ModelInfo[]>(`/api/chat/providers/${provider}/models`),

		getSettings: () => api.get<ChatSettings>('/api/chat/settings'),

		updateSettings: (settings: Partial<ChatSettings> & { api_key?: string }) =>
			api.put<ChatSettings>('/api/chat/settings', settings),

		getConversations: () => api.get<Conversation[]>('/api/chat/conversations'),

		createConversation: (title?: string) =>
			api.post<Conversation>('/api/chat/conversations', { title }),

		getConversation: (id: string) =>
			api.get<{ conversation: Conversation; messages: ChatMessage[] }>(
				`/api/chat/conversations/${id}`
			),

		deleteConversation: (id: string) =>
			api.del<void>(`/api/chat/conversations/${id}`),

		sendMessage: (conversationId: string, content: string, provider?: string, model?: string) =>
			api.post<ChatMessage>(`/api/chat/conversations/${conversationId}/messages`, {
				content,
				provider,
				model
			}),

		getSuggestions: (conversationId: string) =>
			api.get<{ suggestions: string[] }>(
				`/api/chat/conversations/${conversationId}/suggestions`
			)
	};
}
