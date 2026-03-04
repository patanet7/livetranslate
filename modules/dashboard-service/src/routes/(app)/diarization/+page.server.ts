import { diarizationApi } from '$lib/api/diarization';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ fetch }) => {
	const api = diarizationApi(fetch);

	const [jobs, speakers, rules] = await Promise.all([
		api.listJobs().catch(() => []),
		api.listSpeakers().catch(() => []),
		api.getRules().catch(() => ({
			enabled: false,
			participant_patterns: [],
			title_patterns: [],
			min_duration_minutes: 5,
			exclude_empty: true
		}))
	]);

	return { jobs, speakers, rules };
};
