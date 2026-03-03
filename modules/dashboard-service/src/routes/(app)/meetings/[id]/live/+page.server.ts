import { firefliesApi } from '$lib/api/fireflies';

export async function load({ fetch, url }) {
	const ffApi = firefliesApi(fetch);

	const sessionId = url.searchParams.get('session');
	let session = null;

	if (sessionId) {
		session = await ffApi.getSession(sessionId).catch(() => null);
	}

	return {
		session,
		sessionId
	};
}
