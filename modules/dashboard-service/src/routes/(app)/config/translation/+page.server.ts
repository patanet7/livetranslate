import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
	const ff = firefliesApi(fetch);
	const cfg = configApi(fetch);
	const [translationConfig, uiConfig, translationModels, translationHealth, activePromptRes] =
		await Promise.all([
			ff.getTranslationConfig().catch(() => null),
			cfg.getUiConfig().catch(() => null),
			cfg.getTranslationModels().catch(() => null),
			cfg.getTranslationHealth().catch(() => null),
			// Try to load the active prompt from server
			(async () => {
				try {
					const api = configApi(fetch);
					const res = await api.getPrompts();
					const prompts = res?.prompts ?? [];
					const active = (prompts as any[]).find(
						(p: any) => p.id === 'active_translation' && p.is_active
					);
					if (active) {
						return { template: active.template, style: active.metadata?.style ?? 'simple' };
					}
				} catch { /* server unavailable */ }
				return null;
			})(),
		]);
	return { translationConfig, uiConfig, translationModels, translationHealth, activePrompt: activePromptRes };
};

export const actions: Actions = {
	update: async ({ request, fetch }) => {
		const data = await request.formData();
		const target_language = data.get('target_language')?.toString() ?? 'es';
		const temperature = parseFloat(data.get('temperature')?.toString() ?? '0.3');
		const max_tokens = parseInt(data.get('max_tokens')?.toString() ?? '512', 10);

		const ff = firefliesApi(fetch);
		try {
			await ff.updateTranslationConfig({ target_language, temperature, max_tokens });
			return { success: true };
		} catch (err) {
			return fail(500, { errors: { form: `Update failed: ${err}` } });
		}
	}
};
