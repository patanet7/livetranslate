import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
	const ff = firefliesApi(fetch);
	const cfg = configApi(fetch);
	const [translationConfig, fullConfig, uiConfig, translationModels, translationHealth] =
		await Promise.all([
			ff.getTranslationConfig().catch(() => null),
			cfg.getFullTranslationConfig().catch(() => null),
			cfg.getUiConfig().catch(() => null),
			cfg.getTranslationModels().catch(() => null),
			cfg.getTranslationHealth().catch(() => null)
		]);
	return { translationConfig, fullConfig, uiConfig, translationModels, translationHealth };
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
