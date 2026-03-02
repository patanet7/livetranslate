import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch, params }) => {
	try {
		const res = await fetch(
			`${ORCHESTRATION_URL}/api/meetings/${params.meeting_id}/transcript`
		);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ sentences: [] }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
