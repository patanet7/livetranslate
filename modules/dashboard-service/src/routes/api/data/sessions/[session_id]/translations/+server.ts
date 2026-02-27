import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch, params }) => {
	try {
		const res = await fetch(
			`${ORCHESTRATION_URL}/api/data/sessions/${params.session_id}/translations`
		);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify([]), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
