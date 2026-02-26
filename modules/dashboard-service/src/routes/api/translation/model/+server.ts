import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, fetch }) => {
	try {
		const body = await request.json();
		const res = await fetch(`${ORCHESTRATION_URL}/api/translation/model`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ message: 'Failed to switch model' }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
