import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch }) => {
	try {
		const res = await fetch(`${ORCHESTRATION_URL}/api/system/dashboard/stats`);
		return new Response(res.body, {
			status: res.status,
			headers: { 'Content-Type': 'application/json' }
		});
	} catch (e) {
		console.error('Dashboard stats fetch failed:', e);
		// Return minimal stats on connection failure
		return new Response(JSON.stringify({
			total_meetings: 0,
			active_meetings: 0,
			total_chunks: 0,
			total_translations: 0,
			total_audio_minutes: 0,
			by_source: { fireflies: 0, loopback: 0, gmeet: 0, other: 0 },
			by_status: { ephemeral: 0, active: 0, completed: 0, interrupted: 0 },
			active_meeting_list: [],
			daily_activity: [],
			services: [],
			generated_at: new Date().toISOString(),
			database_connected: false
		}), {
			status: 200,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};
