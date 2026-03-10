import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch, params }) => {
	try {
		const res = await fetch(
			`${ORCHESTRATION_URL}/api/export/meetings/${params.meeting_id}/archive`
		);
		return new Response(res.body, {
			status: res.status,
			headers: {
				'Content-Type': res.headers.get('Content-Type') || 'application/zip',
				'Content-Disposition':
					res.headers.get('Content-Disposition') || `attachment; filename="meeting_export.zip"`
			}
		});
	} catch {
		return new Response('Export service unavailable', { status: 503 });
	}
};
