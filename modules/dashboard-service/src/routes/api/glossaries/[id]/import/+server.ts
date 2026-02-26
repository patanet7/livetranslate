import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ fetch, request, params }) => {
	try {
		const formData = await request.formData();
		const res = await fetch(`${ORCHESTRATION_URL}/api/glossaries/${params.id}/import`, {
			method: 'POST',
			body: formData
		});
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch {
		return new Response(JSON.stringify({ error: 'Failed to import CSV' }), {
			status: 503,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
