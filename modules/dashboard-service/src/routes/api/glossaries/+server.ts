import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch, url }) => {
	try {
		const qs = url.searchParams.toString();
		const path = `/api/glossaries${qs ? '?' + qs : ''}`;
		const res = await fetch(`${ORCHESTRATION_URL}${path}`);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify([]), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};

export const POST: RequestHandler = async ({ fetch, request }) => {
	try {
		const body = await request.text();
		const res = await fetch(`${ORCHESTRATION_URL}/api/glossaries`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body
		});
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ error: 'Failed to create glossary' }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
