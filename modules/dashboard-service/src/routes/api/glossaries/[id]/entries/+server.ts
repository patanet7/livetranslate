import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch, params, url }) => {
	try {
		const qs = url.searchParams.toString();
		const path = `/api/glossaries/${params.id}/entries${qs ? '?' + qs : ''}`;
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

export const POST: RequestHandler = async ({ fetch, request, params }) => {
	try {
		const body = await request.text();
		const res = await fetch(`${ORCHESTRATION_URL}/api/glossaries/${params.id}/entries`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body
		});
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ error: 'Failed to create entry' }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
