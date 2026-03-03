import { error } from '@sveltejs/kit';
import { meetingsApi } from '$lib/api/meetings';

export async function load({ params, fetch }) {
	const api = meetingsApi(fetch);

	const meetingResult = await api.get(params.id).catch(() => null);
	if (!meetingResult?.meeting) {
		error(404, 'Meeting not found');
	}

	const meeting = meetingResult.meeting;
	// PostgreSQL JSONB columns may arrive as JSON strings
	if (typeof meeting.participants === 'string') {
		try { meeting.participants = JSON.parse(meeting.participants); } catch { meeting.participants = []; }
	}

	return { meeting };
}
