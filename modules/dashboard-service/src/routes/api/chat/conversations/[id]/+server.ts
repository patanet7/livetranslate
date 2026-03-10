import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(
		`${ORCHESTRATION_URL}/api/chat/conversations/${params.id}`
	);
	return new Response(res.body, {
		status: res.status,
		headers: { 'Content-Type': 'application/json' }
	});
};

export const DELETE: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(
		`${ORCHESTRATION_URL}/api/chat/conversations/${params.id}`,
		{ method: 'DELETE' }
	);
	return new Response(res.body, {
		status: res.status,
		headers: { 'Content-Type': 'application/json' }
	});
};
