import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
  const ff = firefliesApi(fetch);
  const cfg = configApi(fetch);
  const [translationConfig, uiConfig] = await Promise.all([
    ff.getTranslationConfig().catch(() => null),
    cfg.getUiConfig().catch(() => null)
  ]);
  return { translationConfig, uiConfig };
};

export const actions: Actions = {
  update: async ({ request, fetch }) => {
    const data = await request.formData();
    const backend = data.get('backend')?.toString() ?? 'ollama';
    const model = data.get('model')?.toString() ?? '';
    const target_language = data.get('target_language')?.toString() ?? 'es';
    const temperature = parseFloat(data.get('temperature')?.toString() ?? '0.3');

    const ff = firefliesApi(fetch);
    try {
      await ff.updateTranslationConfig({ backend, model, target_language, temperature });
      return { success: true };
    } catch (err) {
      return fail(500, { errors: { form: `Update failed: ${err}` } });
    }
  }
};
