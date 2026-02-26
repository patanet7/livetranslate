import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { glossaryApi } from '$lib/api/glossary';

export const load: PageServerLoad = async ({ fetch }) => {
	const gApi = glossaryApi(fetch);
	const glossaries = await gApi.list().catch(() => []);

	let entries: Awaited<ReturnType<typeof gApi.listEntries>> = [];
	const defaultGlossary = glossaries.find((g) => g.is_default) ?? glossaries[0];
	if (defaultGlossary) {
		entries = await gApi.listEntries(defaultGlossary.glossary_id).catch(() => []);
	}

	return { glossaries, entries, activeGlossaryId: defaultGlossary?.glossary_id ?? null };
};

export const actions: Actions = {
	addEntry: async ({ request, fetch }) => {
		const data = await request.formData();
		const glossary_id = data.get('glossary_id')?.toString();
		const source_term = data.get('source_term')?.toString()?.trim();
		const translation = data.get('translation')?.toString()?.trim();
		const target_language = data.get('target_language')?.toString() ?? 'es';

		if (!glossary_id || !source_term || !translation) {
			return fail(400, { errors: { form: 'All fields are required' } });
		}

		const gApi = glossaryApi(fetch);
		try {
			await gApi.createEntry(glossary_id, {
				source_term,
				translations: { [target_language]: translation },
				context: '',
				notes: '',
				case_sensitive: false,
				match_whole_word: true,
				priority: 5
			});
			return { success: true };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to add entry: ${err}` } });
		}
	},

	deleteEntry: async ({ request, fetch }) => {
		const data = await request.formData();
		const glossary_id = data.get('glossary_id')?.toString();
		const entry_id = data.get('entry_id')?.toString();

		if (!glossary_id || !entry_id) return fail(400, { errors: { form: 'Missing IDs' } });

		const gApi = glossaryApi(fetch);
		try {
			await gApi.deleteEntry(glossary_id, entry_id);
			return { success: true };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to delete: ${err}` } });
		}
	}
};
