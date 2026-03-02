import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch, url }) => {
	try {
		const params = new URLSearchParams();
		const limit = url.searchParams.get('limit');
		const offset = url.searchParams.get('offset');
		if (limit) params.set('limit', limit);
		if (offset) params.set('offset', offset);

		const res = await fetch(
			`${ORCHESTRATION_URL}/api/meetings/?${params.toString()}`
		);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ meetings: [] }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
