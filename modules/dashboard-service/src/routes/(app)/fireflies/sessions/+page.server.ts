import type { PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';

export const load: PageServerLoad = async ({ fetch }) => {
	const ff = firefliesApi(fetch);
	const sessions = await ff.listSessions().catch(() => []);
	return { sessions };
};
