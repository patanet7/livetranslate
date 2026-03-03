import { meetingsApi } from '$lib/api/meetings';

export async function load({ fetch, url }) {
	const api = meetingsApi(fetch);
	const q = url.searchParams.get('q');
	const limit = Number(url.searchParams.get('limit')) || 50;
	const offset = Number(url.searchParams.get('offset')) || 0;

	try {
		if (q) {
			const result = await api.search(q, limit);
			return { meetings: result.results, query: q, total: result.count, limit, offset };
		}
		const result = await api.list(limit, offset);
		return { meetings: result.meetings, query: null, total: result.count ?? result.meetings.length, limit, offset };
	} catch {
		return { meetings: [], query: q, total: 0, limit, offset };
	}
}
