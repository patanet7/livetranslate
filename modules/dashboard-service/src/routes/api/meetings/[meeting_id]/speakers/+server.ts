import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, fetch }) => {
	try {
		const res = await fetch(
			`${ORCHESTRATION_URL}/api/meetings/${params.meeting_id}/speakers`
		);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ speakers: [] }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
