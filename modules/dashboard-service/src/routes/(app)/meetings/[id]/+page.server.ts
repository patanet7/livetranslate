import { meetingsApi } from '$lib/api/meetings';
import type { MeetingSentence } from '$lib/types';

/** PostgreSQL JSONB columns sometimes arrive as JSON strings — parse them into real arrays. */
function parseSentenceArrays(sentence: MeetingSentence): MeetingSentence {
	// eslint-disable-next-line @typescript-eslint/no-explicit-any -- backend may return JSON strings for JSONB columns
	const s = sentence as any;
	if (typeof s.translations === 'string') {
		try { s.translations = JSON.parse(s.translations); } catch { s.translations = []; }
	}
	if (typeof s.chunk_ids === 'string') {
		try { s.chunk_ids = JSON.parse(s.chunk_ids); } catch { s.chunk_ids = []; }
	}
	return s as MeetingSentence;
}

export async function load({ params, fetch }) {
	const api = meetingsApi(fetch);

	// Load transcript and insights in parallel (non-critical — don't block on failure)
	const [transcriptResult, insightsResult] = await Promise.all([
		api.getTranscript(params.id).catch(() => ({ meeting_id: params.id, sentences: [], count: 0 })),
		api.getInsights(params.id).catch(() => ({ meeting_id: params.id, insights: [], count: 0 }))
	]);

	// Normalize JSONB string columns to real arrays
	if (transcriptResult.sentences) {
		transcriptResult.sentences = transcriptResult.sentences.map(parseSentenceArrays);
	}

	return {
		transcript: transcriptResult,
		insights: insightsResult
	};
}
