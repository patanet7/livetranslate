import { ORCHESTRATION_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ params }) => {
	try {
		const res = await fetch(
			`${ORCHESTRATION_URL}/api/meetings/${params.meeting_id}/sync`,
			{
				method: 'POST',
				headers: { 'Content-Type': 'application/json' }
			}
		);
		const data = await res.json();
		return json(data, { status: res.status });
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Unknown error';
		return json({ error: `Proxy failed: ${message}` }, { status: 502 });
	}
};
