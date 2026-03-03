import { error } from '@sveltejs/kit';
import { meetingsApi } from '$lib/api/meetings';

export async function load({ params, fetch }) {
	const api = meetingsApi(fetch);

	const meetingResult = await api.get(params.id).catch(() => null);
	if (!meetingResult?.meeting) {
		error(404, 'Meeting not found');
	}

	// Load transcript and insights in parallel (non-critical — don't block on failure)
	const [transcriptResult, insightsResult] = await Promise.all([
		api.getTranscript(params.id).catch(() => ({ meeting_id: params.id, sentences: [], count: 0 })),
		api.getInsights(params.id).catch(() => ({ meeting_id: params.id, insights: [], count: 0 }))
	]);

	return {
		meeting: meetingResult.meeting,
		transcript: transcriptResult,
		insights: insightsResult
	};
}
