import { fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
	const ff = firefliesApi(fetch);
	const cfg = configApi(fetch);

	const [sessions, uiConfig] = await Promise.all([
		ff.listSessions().catch(() => []),
		cfg.getUiConfig().catch(() => null)
	]);

	return { sessions, uiConfig };
};

export const actions: Actions = {
	connect: async ({ request, fetch }) => {
		const data = await request.formData();
		const transcript_id = data.get('transcript_id')?.toString()?.trim();
		const api_key = data.get('api_key')?.toString()?.trim() || null;
		const target_languages =
			data
				.get('target_languages')
				?.toString()
				?.split(',')
				.filter(Boolean) ?? [];
		const domain = data.get('domain')?.toString() || null;

		if (!transcript_id) {
			return fail(400, {
				transcript_id: '' as string,
				errors: { transcript_id: 'Transcript ID is required', form: '' }
			});
		}

		const ff = firefliesApi(fetch);
		try {
			const result = await ff.connect({
				transcript_id,
				api_key,
				target_languages: target_languages.length > 0 ? target_languages : null,
				domain
			});

			redirect(303, `/fireflies/connect?session=${result.session_id}`);
		} catch (err) {
			return fail(500, {
				transcript_id,
				errors: { transcript_id: '', form: `Connection failed: ${err}` }
			});
		}
	}
};
