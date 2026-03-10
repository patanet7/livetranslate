import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(
		`${ORCHESTRATION_URL}/api/intelligence/sessions/${params.id}/notes`
	);
	if (!res.ok) {
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	}
	// Unwrap envelope and map field names to match frontend interface
	const data = await res.json();
	const raw = data.notes ?? data ?? [];
	const notes = Array.isArray(raw)
		? raw.map((n: Record<string, unknown>) => ({
				id: n.note_id ?? n.id,
				type: n.note_type ?? n.type ?? 'auto',
				speaker: n.speaker_name ?? n.speaker,
				content: n.content,
				timestamp: n.created_at ?? n.timestamp,
				processing_time_ms: n.processing_time_ms
			}))
		: [];
	return new Response(JSON.stringify(notes), {
		status: 200,
		headers: { 'Content-Type': 'application/json' }
	});
};

export const POST: RequestHandler = async ({ params, request, fetch }) => {
	const body = await request.json();
	const res = await fetch(
		`${ORCHESTRATION_URL}/api/intelligence/sessions/${params.id}/notes`,
		{
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		}
	);
	return new Response(res.body, {
		status: res.status,
		headers: { 'Content-Type': 'application/json' }
	});
};
