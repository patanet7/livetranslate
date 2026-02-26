import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const PATCH: RequestHandler = async ({ fetch, request, params }) => {
	try {
		const body = await request.text();
		const res = await fetch(`${ORCHESTRATION_URL}/api/glossaries/${params.id}`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body
		});
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ error: 'Failed to update glossary' }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};

export const DELETE: RequestHandler = async ({ fetch, params }) => {
	try {
		const res = await fetch(`${ORCHESTRATION_URL}/api/glossaries/${params.id}`, {
			method: 'DELETE'
		});
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ error: 'Failed to delete glossary' }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
