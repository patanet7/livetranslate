import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(
		`${ORCHESTRATION_URL}/api/intelligence/sessions/${params.id}/insights`
	);
	if (!res.ok) {
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	}
	// Unwrap envelope and map field names to match frontend interface
	const data = await res.json();
	const raw = data.insights ?? data ?? [];
	const insights = Array.isArray(raw)
		? raw.map((i: Record<string, unknown>) => ({
				id: i.insight_id ?? i.id,
				title: i.title ?? String(i.insight_type ?? '').replace(/_/g, ' '),
				type: i.insight_type ?? i.type,
				content: i.content,
				processing_time_ms: i.processing_time_ms ?? 0,
				llm_model: i.llm_model ?? i.llm_backend ?? 'unknown',
				created_at: i.created_at
			}))
		: [];
	return new Response(JSON.stringify(insights), {
		status: 200,
		headers: { 'Content-Type': 'application/json' }
	});
};
