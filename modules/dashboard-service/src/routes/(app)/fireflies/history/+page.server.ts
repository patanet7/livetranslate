import type { PageServerLoad } from './$types';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
	const cfg = configApi(fetch);
	const uiConfig = await cfg.getUiConfig().catch(() => null);
	return { uiConfig };
};
