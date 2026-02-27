import type { PageServerLoad } from './$types';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
	const cfg = configApi(fetch);
	const uiConfig = await cfg.getUiConfig().catch(() => ({
		languages: [],
		language_codes: [],
		domains: [],
		defaults: {},
		translation_models: [],
		translation_service_available: false,
		config_version: ''
	}));
	return { uiConfig };
};
