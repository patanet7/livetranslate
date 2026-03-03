import { meetingsApi } from '$lib/api/meetings';
import type { Meeting } from '$lib/types';

/** PostgreSQL JSONB columns may arrive as JSON strings — parse them in place. */
function parseMeetingArrays(m: Meeting): Meeting {
	if (typeof m.participants === 'string') {
		try { m.participants = JSON.parse(m.participants); } catch { m.participants = []; }
	}
	return m;
}

export async function load({ fetch, url }) {
	const api = meetingsApi(fetch);
	const q = url.searchParams.get('q');
	const limit = Number(url.searchParams.get('limit')) || 50;
	const offset = Number(url.searchParams.get('offset')) || 0;

	try {
		if (q) {
			const result = await api.search(q, limit);
			return { meetings: result.results.map(parseMeetingArrays), query: q, total: result.count, limit, offset };
		}
		const result = await api.list(limit, offset);
		return { meetings: result.meetings.map(parseMeetingArrays), query: null, total: result.count ?? result.meetings.length, limit, offset };
	} catch {
		return { meetings: [], query: q, total: 0, limit, offset };
	}
}
