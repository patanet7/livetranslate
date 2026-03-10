import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch, params, url }) => {
	const format = url.searchParams.get('format') || 'srt';
	try {
		const res = await fetch(
			`${ORCHESTRATION_URL}/api/export/meetings/${params.meeting_id}/transcript?format=${format}`
		);
		return new Response(res.body, {
			status: res.status,
			headers: {
				'Content-Type': res.headers.get('Content-Type') || 'application/octet-stream',
				'Content-Disposition':
					res.headers.get('Content-Disposition') || `attachment; filename="transcript.${format}"`
			}
		});
	} catch {
		return new Response('Export service unavailable', { status: 503 });
	}
};
