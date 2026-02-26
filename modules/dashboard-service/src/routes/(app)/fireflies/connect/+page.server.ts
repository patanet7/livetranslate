import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';

export const load: PageServerLoad = async ({ url, fetch }) => {
	const sessionId = url.searchParams.get('session');
	if (!sessionId) {
		throw error(400, 'Missing session parameter');
	}

	const ff = firefliesApi(fetch);
	try {
		const session = await ff.getSession(sessionId);
		return { session };
	} catch {
		throw error(404, `Session ${sessionId} not found`);
	}
};
