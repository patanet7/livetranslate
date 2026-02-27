import type { PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { configApi } from '$lib/api/config';
import { createApi } from '$lib/api/orchestration';

export const load: PageServerLoad = async ({ fetch }) => {
	const ff = firefliesApi(fetch);
	const cfg = configApi(fetch);
	const api = createApi(fetch);

	const [sessions, systemHealth, translationHealth] = await Promise.all([
		ff.listSessions().catch(() => null),
		cfg.getHealth().catch(() => null),
		api.get<{ status: string }>('/api/translation/health').catch(() => null)
	]);

	return { sessions, systemHealth, translationHealth };
};
