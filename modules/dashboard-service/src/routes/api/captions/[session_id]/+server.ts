import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ params, request, fetch }) => {
	const { session_id } = params;

	try {
		let body: string | undefined;
		const contentType = request.headers.get('content-type');

		if (contentType?.includes('application/json')) {
			body = await request.text();
		}

		const res = await fetch(`${ORCHESTRATION_URL}/api/captions/${session_id}`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: body ?? '{}'
		});

		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(
			JSON.stringify({ error: 'Failed to reach orchestration service' }),
			{
				status: 502,
				headers: { 'Content-Type': 'application/json' }
			}
		);
	}
};

export const DELETE: RequestHandler = async ({ params, fetch }) => {
	const { session_id } = params;

	try {
		const res = await fetch(`${ORCHESTRATION_URL}/api/captions/${session_id}`, {
			method: 'DELETE'
		});

		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(
			JSON.stringify({ error: 'Failed to reach orchestration service' }),
			{
				status: 502,
				headers: { 'Content-Type': 'application/json' }
			}
		);
	}
};
