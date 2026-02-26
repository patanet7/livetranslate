import type { PageServerLoad } from './$types';
import { glossaryApi } from '$lib/api/glossary';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
	const gApi = glossaryApi(fetch);
	const cApi = configApi(fetch);

	const [glossaries, uiConfig] = await Promise.all([
		gApi.list().catch(() => []),
		cApi.getUiConfig().catch(() => ({
			languages: [],
			language_codes: [],
			domains: [],
			defaults: {},
			translation_models: [],
			translation_service_available: false,
			config_version: ''
		}))
	]);

	const defaultGlossary = glossaries.find((g) => g.is_default) ?? glossaries[0] ?? null;
	let entries: Awaited<ReturnType<typeof gApi.listEntries>> = [];

	if (defaultGlossary) {
		entries = await gApi.listEntries(defaultGlossary.glossary_id).catch(() => []);
	}

	return {
		glossaries,
		entries,
		activeGlossaryId: defaultGlossary?.glossary_id ?? null,
		uiConfig
	};
};
