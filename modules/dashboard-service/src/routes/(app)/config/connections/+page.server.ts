import type { PageServerLoad } from './$types';
import { ORCHESTRATION_URL } from '$env/static/private';

export const load: PageServerLoad = async ({ fetch }) => {
	const base = ORCHESTRATION_URL || 'http://localhost:3000';

	const [connectionsRes, preferencesRes] = await Promise.all([
		fetch(`${base}/api/connections`).catch(() => null),
		fetch(`${base}/api/connections/preferences/all`).catch(() => null)
	]);

	const connections = connectionsRes?.ok ? await connectionsRes.json() : [];
	const preferences = preferencesRes?.ok ? await preferencesRes.json() : {};

	return { connections, preferences };
};
