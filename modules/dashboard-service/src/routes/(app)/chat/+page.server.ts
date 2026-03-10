import type { PageServerLoad } from './$types';
import { chatApi } from '$lib/api/chat';

export const load: PageServerLoad = async ({ fetch }) => {
	const api = chatApi(fetch);

	const [conversations, settings] = await Promise.all([
		api.getConversations().catch(() => []),
		api.getSettings().catch(() => ({
			provider: 'ollama',
			model: null,
			temperature: 0.7,
			max_tokens: 4096,
			has_api_key: false,
			base_url: null
		}))
	]);

	return { conversations, settings };
};
