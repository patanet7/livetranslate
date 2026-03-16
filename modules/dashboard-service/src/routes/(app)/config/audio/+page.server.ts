import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
	const cfg = configApi(fetch);
	const [settings, audioProcessing, chunking] = await Promise.all([
		cfg.getUserSettings().catch(() => null),
		cfg.getAudioProcessing().catch(() => null),
		cfg.getChunking().catch(() => null),
	]);
	return { settings, audioProcessing, chunking };
};

export const actions: Actions = {
	update: async ({ request, fetch }) => {
		const data = await request.formData();
		const cfg = configApi(fetch);

		const audio_auto_start = data.get('audio_auto_start') === 'on';
		const vad_silence_ms = parseInt(data.get('vad_silence_ms')?.toString() ?? '300', 10);
		const noise_suppression = data.get('noise_suppression') === 'on';

		// VAC chunking
		const preset = data.get('vac_preset')?.toString() ?? 'best';
		const stride = parseFloat(data.get('stride')?.toString() ?? '6.0');
		const overlap = parseFloat(data.get('overlap')?.toString() ?? '1.5');
		const prebuffer = parseFloat(data.get('prebuffer')?.toString() ?? '0.5');
		const draft_enabled = data.get('draft_enabled') === 'on';

		try {
			await Promise.all([
				cfg.updateUserSettings({ audio_auto_start }),
				cfg.saveAudioProcessing({
					vad: {
						enabled: true,
						mode: 'webrtc',
						aggressiveness: 2,
						energy_threshold: 0.01,
						voice_freq_min: 85,
						voice_freq_max: 300,
					},
					noise_reduction: {
						enabled: noise_suppression,
						mode: 'moderate',
						strength: 0.7,
						voice_protection: true,
					},
					voice_filter: { enabled: true, fundamental_min: 85, fundamental_max: 300, formant1_min: 200, formant1_max: 1000, preserve_formants: true },
					voice_enhancement: { enabled: true, normalize: false, compressor: { threshold: -20, ratio: 3, knee: 2.0 } },
					limiting: { enabled: true, threshold: -3, release_time: 10 },
					quality_control: { min_snr_db: 10, max_clipping_percent: 1.0, silence_threshold: vad_silence_ms / 1000, enable_quality_gates: true },
				}),
				cfg.saveChunking({
					chunking: {
						chunk_duration: stride,
						overlap_duration: overlap,
						overlap_mode: 'adaptive',
						min_chunk_duration: prebuffer,
						max_chunk_duration: 30.0,
						voice_activity_chunking: true,
					},
					storage: { audio_storage_path: '/data/audio', file_format: 'wav', compression: false, cleanup_old_chunks: true, retention_hours: 24 },
					coordination: { coordinate_with_services: true, sync_chunk_boundaries: true, chunk_metadata_storage: true, enable_chunk_correlation: true },
					database: { store_chunk_metadata: true, store_audio_hashes: true, correlation_tracking: true, performance_metrics: true },
				}),
			]);
			return { success: true, preset, draft_enabled };
		} catch (err) {
			return fail(500, { errors: { form: `Update failed: ${err}` } });
		}
	},
};
