import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
	const cfg = configApi(fetch);
	const uiConfig = await cfg.getUiConfig().catch(() => null);
	return { uiConfig };
};

export const actions: Actions = {
	updateLanguages: async ({ request, fetch }) => {
		const data = await request.formData();
		const enabled = data.getAll('languages') as string[];

		const cfg = configApi(fetch);
		try {
			await cfg.updateUiConfig({ enabled_languages: enabled.length > 0 ? enabled : undefined });
			return { success: true, section: 'languages' };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to save languages: ${err}` } });
		}
	},

	updateDomains: async ({ request, fetch }) => {
		const data = await request.formData();
		const customDomainsJson = data.get('custom_domains') as string;
		const disabledDomainsJson = data.get('disabled_domains') as string;

		const cfg = configApi(fetch);
		try {
			await cfg.updateUiConfig({
				custom_domains: customDomainsJson ? JSON.parse(customDomainsJson) : [],
				disabled_domains: disabledDomainsJson ? JSON.parse(disabledDomainsJson) : []
			});
			return { success: true, section: 'domains' };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to save domains: ${err}` } });
		}
	},

	updateDefaults: async ({ request, fetch }) => {
		const data = await request.formData();
		const defaults: Record<string, unknown> = {};

		const source = data.get('default_source_language') as string;
		if (source) defaults.default_source_language = source;

		const targets = data.getAll('default_target_languages') as string[];
		if (targets.length > 0) defaults.default_target_languages = targets;

		defaults.auto_detect_language = data.get('auto_detect_language') === 'on';

		const threshold = data.get('confidence_threshold');
		if (threshold) defaults.confidence_threshold = parseFloat(threshold as string);

		const contextWindow = data.get('context_window_size');
		if (contextWindow) defaults.context_window_size = parseInt(contextWindow as string);

		const maxBuffer = data.get('max_buffer_words');
		if (maxBuffer) defaults.max_buffer_words = parseInt(maxBuffer as string);

		const pauseThreshold = data.get('pause_threshold_ms');
		if (pauseThreshold) defaults.pause_threshold_ms = parseInt(pauseThreshold as string);

		const cfg = configApi(fetch);
		try {
			await cfg.updateUiConfig({ defaults });
			return { success: true, section: 'defaults' };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to save defaults: ${err}` } });
		}
	},

	reset: async ({ fetch }) => {
		const cfg = configApi(fetch);
		try {
			await cfg.resetUiConfig();
			return { success: true, section: 'reset' };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to reset: ${err}` } });
		}
	}
};
