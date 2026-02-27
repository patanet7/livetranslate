import { ORCHESTRATION_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
	let body: { api_key?: string };
	try {
		body = await request.json();
	} catch {
		return json({ valid: false, error: 'Invalid request body' }, { status: 400 });
	}

	const apiKey = body.api_key;
	if (!apiKey || typeof apiKey !== 'string' || apiKey.trim().length === 0) {
		return json({ valid: false, error: 'API key is required' }, { status: 400 });
	}

	try {
		const res = await fetch(`${ORCHESTRATION_URL}/fireflies/meetings`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ api_key: apiKey })
		});

		if (!res.ok) {
			const text = await res.text().catch(() => 'Unknown error');
			return json(
				{ valid: false, error: `Validation failed: ${res.status} - ${text}` },
				{ status: 200 }
			);
		}

		const data = await res.json();
		const meetingCount =
			Array.isArray(data) ? data.length : (data.meeting_count ?? data.count ?? 0);

		return json({ valid: true, meeting_count: meetingCount });
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Unknown error';
		return json(
			{ valid: false, error: `Connection failed: ${message}` },
			{ status: 200 }
		);
	}
};
