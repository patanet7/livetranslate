import { ORCHESTRATION_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
	try {
		const res = await fetch(`${ORCHESTRATION_URL}/api/diarization/speakers`);
		const data = await res.json();
		return json(data, { status: res.status });
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Unknown error';
		return json({ error: `Proxy failed: ${message}` }, { status: 502 });
	}
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();
		const res = await fetch(`${ORCHESTRATION_URL}/api/diarization/speakers`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
		const data = await res.json();
		return json(data, { status: res.status });
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Unknown error';
		return json({ error: `Proxy failed: ${message}` }, { status: 502 });
	}
};
