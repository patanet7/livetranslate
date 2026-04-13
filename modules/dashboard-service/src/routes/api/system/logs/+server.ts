import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch, url }) => {
	try {
		// Forward query params (limit, level, service)
		const params = new URLSearchParams();
		const limit = url.searchParams.get('limit');
		const level = url.searchParams.get('level');
		const service = url.searchParams.get('service');
		if (limit) params.set('limit', limit);
		if (level) params.set('level', level);
		if (service) params.set('service', service);

		const queryString = params.toString();
		const endpoint = `${ORCHESTRATION_URL}/api/system/logs${queryString ? `?${queryString}` : ''}`;

		const res = await fetch(endpoint);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch (e) {
		console.error('Logs fetch failed:', e);
		return new Response(JSON.stringify({ entries: [], total_buffered: 0 }), {
			status: 200,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
