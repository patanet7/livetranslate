import type { PageServerLoad } from './$types';
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
