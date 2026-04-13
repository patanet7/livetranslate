import type { PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { createApi } from '$lib/api/orchestration';

export const load: PageServerLoad = async ({ fetch, url }) => {
	const ff = firefliesApi(fetch);
	const api = createApi(fetch);

	const [sessions, meetingsRes] = await Promise.all([
		ff.listSessions().catch(() => []),
		// Request up to 250 meetings (max allowed by API)
		api.get<{ meetings: Array<{
			id: string;
			title: string | null;
			status: string;
			source: string;
			created_at: string;
			sentence_count: number;
			chunk_count: number;
			duration: number;
		}>, total: number }>('/api/meetings/?limit=250').catch(() => ({ meetings: [], total: 0 }))
	]);

	const preSelectedSession = url.searchParams.get('session') ?? '';
	const preSelectedMeeting = url.searchParams.get('meeting') ?? '';

	return {
		sessions,
		meetings: meetingsRes.meetings ?? [],
		meetingsTotal: meetingsRes.total ?? 0,
		preSelectedSession,
		preSelectedMeeting
	};
};
