import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ params, fetch }) => {
	try {
		const res = await fetch(
			`${ORCHESTRATION_URL}/api/meetings/${params.meeting_id}/insights/generate`,
			{ method: 'POST' }
		);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ error: 'Failed to reach orchestration service' }), {
			status: 502,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
