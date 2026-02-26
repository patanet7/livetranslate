import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch }) => {
	try {
		const res = await fetch(`${ORCHESTRATION_URL}/api/translation/models`);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ models: [] }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
