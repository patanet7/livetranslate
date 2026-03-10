import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch }) => {
	const res = await fetch(`${ORCHESTRATION_URL}/api/chat/settings`);
	return new Response(res.body, {
		status: res.status,
		headers: { 'Content-Type': 'application/json' }
	});
};

export const PUT: RequestHandler = async ({ request, fetch }) => {
	const body = await request.json();
	const res = await fetch(`${ORCHESTRATION_URL}/api/chat/settings`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	return new Response(res.body, {
		status: res.status,
		headers: { 'Content-Type': 'application/json' }
	});
};
