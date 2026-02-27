import type { PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';

export const load: PageServerLoad = async ({ fetch, url }) => {
	const ff = firefliesApi(fetch);
	const sessions = await ff.listSessions().catch(() => []);
	const preSelectedSession = url.searchParams.get('session') ?? '';
	return { sessions, preSelectedSession };
};
