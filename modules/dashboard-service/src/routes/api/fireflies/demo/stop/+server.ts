import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async () => {
	const res = await fetch(`${ORCHESTRATION_URL}/fireflies/demo/stop`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' }
	});
	return new Response(res.body, {
		status: res.status,
		headers: { 'Content-Type': 'application/json' }
	});
};
