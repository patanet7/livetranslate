import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ params, request, fetch }) => {
	const body = await request.json();
	const res = await fetch(
		`${ORCHESTRATION_URL}/api/intelligence/agent/conversations/${params.id}/messages/stream`,
		{
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				Accept: 'text/event-stream'
			},
			body: JSON.stringify(body)
		}
	);
	return new Response(res.body, {
		status: res.status,
		headers: {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			Connection: 'keep-alive'
		}
	});
};
