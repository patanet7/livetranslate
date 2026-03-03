import { error } from '@sveltejs/kit';
import { meetingsApi } from '$lib/api/meetings';
import { firefliesApi } from '$lib/api/fireflies';

export async function load({ params, fetch, url }) {
	const meetingApi = meetingsApi(fetch);
	const ffApi = firefliesApi(fetch);

	const meetingResult = await meetingApi.get(params.id).catch(() => null);
	if (!meetingResult?.meeting) {
		error(404, 'Meeting not found');
	}

	const sessionId = url.searchParams.get('session');
	let session = null;

	if (sessionId) {
		session = await ffApi.getSession(sessionId).catch(() => null);
	}

	return {
		meeting: meetingResult.meeting,
		session,
		sessionId
	};
}
