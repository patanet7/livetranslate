import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
  const cfg = configApi(fetch);
  const settings = await cfg.getUserSettings().catch(() => null);
  return { settings };
};

export const actions: Actions = {
  update: async ({ request, fetch }) => {
    const data = await request.formData();
    const theme = data.get('theme')?.toString() as 'dark' | 'light' ?? 'dark';
    const language = data.get('language')?.toString() ?? 'en';
    const notifications = data.get('notifications') === 'on';

    const cfg = configApi(fetch);
    try {
      await cfg.updateUserSettings({ theme, language, notifications });
      return { success: true };
    } catch (err) {
      return fail(500, { errors: { form: `Update failed: ${err}` } });
    }
  }
};
