import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { translationApi } from '$lib/api/translation';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
  const cfg = configApi(fetch);
  const tApi = translationApi(fetch);

  const [uiConfig, models] = await Promise.all([
    cfg.getUiConfig().catch(() => null),
    tApi.getModels().catch(() => ({ models: [] }))
  ]);

  return { uiConfig, models: models.models };
};

export const actions: Actions = {
  translate: async ({ request, fetch }) => {
    const data = await request.formData();
    const text = data.get('text')?.toString()?.trim();
    const target_language = data.get('target_language')?.toString() ?? 'es';
    const service = data.get('service')?.toString() ?? 'ollama';

    if (!text) {
      return fail(400, { text: '', errors: { text: 'Text is required', form: '' } });
    }

    const tApi = translationApi(fetch);
    try {
      const result = await tApi.translate({
        text,
        target_language,
        service,
        quality: 'balanced'
      });
      return { success: true, result, text };
    } catch (err) {
      return fail(500, { text, errors: { text: '', form: `Translation failed: ${err}` } });
    }
  }
};
