import { error } from '@sveltejs/kit';
import { meetingsApi } from '$lib/api/meetings';

export async function load({ params, fetch }) {
	const api = meetingsApi(fetch);

	const meetingResult = await api.get(params.id).catch(() => null);
	if (!meetingResult?.meeting) {
		error(404, 'Meeting not found');
	}

	return {
		meeting: meetingResult.meeting
	};
}
