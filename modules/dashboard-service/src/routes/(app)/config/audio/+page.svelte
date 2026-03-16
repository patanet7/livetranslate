<script lang="ts">
	import { enhance } from '$app/forms';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Label } from '$lib/components/ui/label';
	import { Badge } from '$lib/components/ui/badge';
	import { toastStore } from '$lib/stores/toast.svelte';

	let { data, form } = $props();

	let submitting = $state(false);

	// Benchmark-optimal presets from Plan 6
	const PRESETS = {
		best:     { stride: 6.0, overlap_zh: 1.5, overlap_en: 0.5, label: 'Best Accuracy', desc: 'CER 9.5% / WER 19.1% — caption every 6s' },
		balanced: { stride: 4.5, overlap_zh: 0.5, overlap_en: 1.0, label: 'Balanced', desc: 'CER 15.6% / WER 21.7% — caption every 4.5s' },
		fastest:  { stride: 1.5, overlap_zh: 0.5, overlap_en: 0.5, label: 'Real-time Subtitles', desc: 'CER 20.4% / WER 32.2% — caption every 1.5s' },
		custom:   { stride: 0, overlap_zh: 0, overlap_en: 0, label: 'Custom', desc: 'Set your own values' },
	} as const;

	type PresetKey = keyof typeof PRESETS;

	// Extract current values from server data
	const chunkingData = data.chunking as Record<string, any> | null;
	const audioData = data.audioProcessing as Record<string, any> | null;

	const currentStride = chunkingData?.chunking?.chunk_duration ?? 6.0;
	const currentOverlap = chunkingData?.chunking?.overlap_duration ?? 1.5;
	const currentPrebuffer = chunkingData?.chunking?.min_chunk_duration ?? 0.5;

	// Determine initial preset from current values
	function detectPreset(stride: number, overlap: number): PresetKey {
		for (const [key, p] of Object.entries(PRESETS) as [PresetKey, typeof PRESETS[PresetKey]][]) {
			if (key === 'custom') continue;
			if (Math.abs(p.stride - stride) < 0.1 && (Math.abs(p.overlap_zh - overlap) < 0.1 || Math.abs(p.overlap_en - overlap) < 0.1)) {
				return key;
			}
		}
		return 'custom';
	}

	let selectedPreset = $state<PresetKey>(detectPreset(currentStride, currentOverlap));
	let customStride = $state(currentStride);
	let customOverlap = $state(currentOverlap);
	let prebuffer = $state(currentPrebuffer);
	let vadSilenceMs = $state(Math.round((audioData?.quality_control?.silence_threshold ?? 0.3) * 1000));
	let noiseSuppression = $state(audioData?.noise_reduction?.enabled ?? true);
	let draftEnabled = $state(true);

	let activeStride = $derived(selectedPreset === 'custom' ? customStride : PRESETS[selectedPreset].stride);
	let activeOverlap = $derived(selectedPreset === 'custom' ? customOverlap : PRESETS[selectedPreset].overlap_zh);
	let draftStride = $derived(activeStride / 2);
</script>

<PageHeader title="Audio Configuration" description="Audio processing, VAC chunking, and two-pass settings" />

<div class="max-w-2xl space-y-6">
	<!-- Section A: Audio Processing -->
	<Card.Root>
		<Card.Header>
			<Card.Title>Audio Processing</Card.Title>
			<Card.Description>Sample rate conversion and voice activity detection</Card.Description>
		</Card.Header>
		<Card.Content>
			<form method="POST" action="?/update" use:enhance={() => {
				submitting = true;
				return async ({ result, update }) => {
					await update();
					submitting = false;
					if (result.type === 'success') {
						toastStore.success('Audio settings saved');
					} else if (result.type === 'failure') {
						toastStore.error('Failed to save audio settings');
					}
				};
			}} class="space-y-6">
				<div class="grid grid-cols-2 gap-4">
					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Browser Sample Rate</p>
						<p class="text-sm font-medium">48,000 Hz</p>
					</div>
					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Backend Sample Rate</p>
						<p class="text-sm font-medium">16,000 Hz</p>
					</div>
				</div>

				<div class="space-y-2">
					<Label for="vad_silence_ms">VAD Silence Threshold</Label>
					<div class="flex items-center gap-3">
						<input
							id="vad_silence_ms"
							name="vad_silence_ms"
							type="range"
							min="100"
							max="1000"
							step="50"
							bind:value={vadSilenceMs}
							class="flex-1"
						/>
						<span class="w-16 text-right text-sm tabular-nums">{vadSilenceMs}ms</span>
					</div>
					<p class="text-xs text-muted-foreground">Lower = more responsive, higher = fewer false triggers (benchmark optimal: 300ms)</p>
				</div>

				<div class="flex items-center justify-between">
					<Label for="noise_suppression">Noise Suppression</Label>
					<input
						type="checkbox"
						id="noise_suppression"
						name="noise_suppression"
						bind:checked={noiseSuppression}
						class="h-4 w-4"
					/>
				</div>

				<div class="flex items-center justify-between">
					<Label for="audio_auto_start">Auto-start audio capture</Label>
					<input
						type="checkbox"
						id="audio_auto_start"
						name="audio_auto_start"
						checked={data.settings?.audio_auto_start ?? false}
						class="h-4 w-4"
					/>
				</div>

				<!-- Section B: VAC Chunking -->
				<div class="border-t pt-4">
					<h3 class="text-sm font-semibold mb-1">VAC Chunking</h3>
					<p class="text-xs text-muted-foreground mb-4">Voice Activity Chunking parameters from Plan 6 benchmark sweep</p>

					<div class="space-y-3">
						<div class="space-y-2">
							<Label>Preset</Label>
							<div class="grid grid-cols-2 gap-2">
								{#each Object.entries(PRESETS) as [key, preset] (key)}
									<button
										type="button"
										class="rounded-md border p-3 text-left text-sm transition-colors
											{selectedPreset === key ? 'border-primary bg-primary/10' : 'hover:bg-accent/50'}"
										onclick={() => { selectedPreset = key as PresetKey; }}
									>
										<span class="font-medium">{preset.label}</span>
										<span class="block text-xs text-muted-foreground mt-0.5">{preset.desc}</span>
									</button>
								{/each}
							</div>
						</div>

						<input type="hidden" name="vac_preset" value={selectedPreset} />
						<input type="hidden" name="stride" value={activeStride} />
						<input type="hidden" name="overlap" value={activeOverlap} />
						<input type="hidden" name="prebuffer" value={prebuffer} />

						{#if selectedPreset === 'custom'}
							<div class="grid grid-cols-3 gap-3 rounded-md border bg-muted/50 p-3">
								<div class="space-y-1">
									<Label for="custom_stride" class="text-xs">Stride (s)</Label>
									<input
										id="custom_stride"
										type="number"
										min="0.5"
										max="30"
										step="0.5"
										bind:value={customStride}
										class="w-full rounded-md border bg-background px-2 py-1 text-sm"
									/>
								</div>
								<div class="space-y-1">
									<Label for="custom_overlap" class="text-xs">Overlap (s)</Label>
									<input
										id="custom_overlap"
										type="number"
										min="0"
										max="5"
										step="0.1"
										bind:value={customOverlap}
										class="w-full rounded-md border bg-background px-2 py-1 text-sm"
									/>
								</div>
								<div class="space-y-1">
									<Label for="custom_prebuffer" class="text-xs">Prebuffer (s)</Label>
									<input
										id="custom_prebuffer"
										type="number"
										min="0"
										max="2"
										step="0.1"
										bind:value={prebuffer}
										class="w-full rounded-md border bg-background px-2 py-1 text-sm"
									/>
								</div>
							</div>
						{:else}
							<div class="rounded-md border bg-muted/50 p-3">
								<div class="grid grid-cols-3 gap-3 text-sm">
									<div>
										<span class="text-xs text-muted-foreground">Stride</span>
										<p class="font-medium">{activeStride}s</p>
									</div>
									<div>
										<span class="text-xs text-muted-foreground">Overlap</span>
										<p class="font-medium">{activeOverlap}s</p>
									</div>
									<div>
										<span class="text-xs text-muted-foreground">Prebuffer</span>
										<p class="font-medium">{prebuffer}s</p>
									</div>
								</div>
							</div>
						{/if}

						<div class="rounded-md bg-blue-950/30 border border-blue-800/30 p-3 text-xs text-blue-200">
							English needs overlap=0.5s (word boundaries dedup well).
							Chinese needs overlap=1.5s (CJK characters need more context).
						</div>
					</div>
				</div>

				<!-- Section C: Two-Pass Draft/Final -->
				<div class="border-t pt-4">
					<h3 class="text-sm font-semibold mb-1">Two-Pass Draft/Final</h3>
					<p class="text-xs text-muted-foreground mb-3">Fast first-pass captions refined by a second full pass</p>

					<div class="flex items-center justify-between mb-3">
						<Label for="draft_enabled">Enable draft pass</Label>
						<input
							type="checkbox"
							id="draft_enabled"
							name="draft_enabled"
							bind:checked={draftEnabled}
							class="h-4 w-4"
						/>
					</div>

					{#if draftEnabled}
						<div class="rounded-md border bg-muted/50 p-3 text-sm">
							<span class="text-xs text-muted-foreground">Draft stride (auto)</span>
							<p class="font-medium">{draftStride.toFixed(1)}s <span class="text-xs text-muted-foreground">(stride / 2)</span></p>
						</div>
					{/if}
				</div>

				{#if form?.errors?.form}
					<p class="text-sm text-destructive">{form.errors.form}</p>
				{/if}
				{#if form?.success}
					<p class="text-sm text-green-600">Settings saved</p>
				{/if}

				<Button type="submit" disabled={submitting}>
					{#if submitting}Saving...{:else}Save{/if}
				</Button>
			</form>
		</Card.Content>
	</Card.Root>
</div>
