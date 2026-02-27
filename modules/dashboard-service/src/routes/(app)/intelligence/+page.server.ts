import type { PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { createApi } from '$lib/api/orchestration';

export const load: PageServerLoad = async ({ fetch }) => {
	const ff = firefliesApi(fetch);
	const api = createApi(fetch);

	const [sessions, templates] = await Promise.all([
		ff.listSessions().catch(() => []),
		api
			.get<{ id: string; name: string; description: string; type: string }[]>(
				'/api/intelligence/templates'
			)
			.catch(() => [])
	]);

	return { sessions, templates };
};
