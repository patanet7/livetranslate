import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch }) => {
	try {
		const res = await fetch(`${ORCHESTRATION_URL}/api/system/health`);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ status: 'down', services: {} }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
