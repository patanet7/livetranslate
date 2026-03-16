<script lang="ts">
	import { untrack } from 'svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Select from '$lib/components/ui/select';
	import * as Dialog from '$lib/components/ui/dialog';
	import { loopbackStore, type DisplayMode } from '$lib/stores/loopback.svelte';

	interface Props {
		devices: MediaDeviceInfo[];
		selectedDeviceId?: string;
		onDeviceChange?: (deviceId: string) => void;
		onStartCapture?: () => void;
		onStopCapture?: () => void;
		onStartMeeting?: () => void;
		onEndMeeting?: () => void;
		/** I1: Callback to send config changes to the server via WebSocket */
		onConfigChange?: (config: { model?: string; language?: string; target_language?: string }) => void;
		onStartDemo?: () => void;
		onStopDemo?: () => void;
		isDemoRunning?: boolean;
		/** RMS audio level from capture (0-1) for VU meter */
		audioLevel?: number;
	}

	let {
		devices,
		selectedDeviceId = $bindable(''),
		onDeviceChange,
		onStartCapture,
		onStopCapture,
		onStartMeeting,
		onEndMeeting,
		onConfigChange,
		onStartDemo,
		onStopDemo,
		isDemoRunning = false,
		audioLevel = 0,
	}: Props = $props();

	// Scale peak (0-1) to percentage using dB scale.
	// Speech peaks at 0.05-0.5; log scale makes quiet speech visible.
	// Maps -60dB..0dB → 0..100%
	let levelDb = $derived(audioLevel > 0 ? 20 * Math.log10(audioLevel) : -60);
	let levelPercent = $derived(Math.max(0, Math.min(100, Math.round((levelDb + 60) * (100 / 60)))));
	let levelColor = $derived(
		levelPercent > 90 ? 'var(--status-clip, #ef4444)' :
		levelPercent > 5 ? 'var(--status-up, #22c55e)' :
		'var(--text-muted, #94a3b8)'
	);

	const SOURCE_LANGUAGES = [
		{ value: 'auto', label: 'Auto Detect' },
		{ value: 'en', label: 'English' },
		{ value: 'zh', label: 'Chinese' },
		{ value: 'ja', label: 'Japanese' },
		{ value: 'es', label: 'Spanish' },
		{ value: 'fr', label: 'French' },
	];

	const TARGET_LANGUAGES = [
		{ value: 'en', label: 'English' },
		{ value: 'zh', label: 'Chinese' },
		{ value: 'ja', label: 'Japanese' },
		{ value: 'es', label: 'Spanish' },
		{ value: 'fr', label: 'French' },
	];

	const MODELS = [
		{ value: 'auto', label: 'Auto' },
		{ value: 'large-v3-turbo', label: 'large-v3-turbo' },
		{ value: 'SenseVoiceSmall', label: 'SenseVoiceSmall' },
	];

	const DISPLAY_MODES: { value: DisplayMode; label: string }[] = [
		{ value: 'split', label: 'Split' },
		{ value: 'subtitle', label: 'Subtitle' },
		{ value: 'transcript', label: 'Transcript' },
	];

	let showEndMeetingDialog = $state(false);

	// Local select values bound to store
	let sourceLanguageValue = $state(loopbackStore.sourceLanguage ?? 'auto');
	let targetLanguageValue = $state(loopbackStore.targetLanguage);
	let modelOverride = $state('auto');

	// Track previous values to avoid writing to the store on initial render.
	// Effects run immediately during component initialization and writing to
	// the shared store can cause reactive cascades that break hydration.
	let prevSourceLang = $state(untrack(() => sourceLanguageValue));
	let prevTargetLang = $state(untrack(() => targetLanguageValue));
	let prevModel = $state(untrack(() => modelOverride));

	// Sync source language changes to store and notify server
	$effect(() => {
		const val = sourceLanguageValue;
		if (val === prevSourceLang) return;
		prevSourceLang = val;
		const lang = val === 'auto' ? null : val;
		loopbackStore.sourceLanguage = lang;
		// I2: Send config change to server if capturing
		if (loopbackStore.isCapturing && lang) {
			onConfigChange?.({ language: lang });
		}
	});

	// Reverse sync: when server sends language_detected, the store updates
	// but the local select value must follow so the dropdown reflects reality.
	$effect(() => {
		const storeLang = loopbackStore.sourceLanguage;
		const selectVal = storeLang ?? 'auto';
		if (selectVal !== sourceLanguageValue) {
			sourceLanguageValue = selectVal;
			prevSourceLang = selectVal;
		}
	});

	// Sync target language changes to store and notify server
	$effect(() => {
		const val = targetLanguageValue;
		if (val === prevTargetLang) return;
		prevTargetLang = val;
		loopbackStore.targetLanguage = val;
		if (loopbackStore.isCapturing) {
			onConfigChange?.({ target_language: val });
		}
	});

	// I1: Send model override to server when changed during active session
	$effect(() => {
		const val = modelOverride;
		if (val === prevModel) return;
		prevModel = val;
		if (loopbackStore.isCapturing && val !== 'auto') {
			onConfigChange?.({ model: val });
		}
	});

	function statusColor(status: 'up' | 'down'): string {
		return status === 'up' ? 'var(--status-up, #22c55e)' : 'var(--status-down, #ef4444)';
	}

	function handleEndMeetingConfirm() {
		showEndMeetingDialog = false;
		onEndMeeting?.();
	}
</script>

<div class="toolbar">
	<!-- Audio Source -->
	<div class="toolbar-group">
		<span class="toolbar-label">Audio Source</span>
		<Select.Root type="single" value={selectedDeviceId} onValueChange={(v) => { if (v) { selectedDeviceId = v; onDeviceChange?.(v); } }}>
			<Select.Trigger class="toolbar-select">
				{devices.find((d) => d.deviceId === selectedDeviceId)?.label || 'Select device'}
			</Select.Trigger>
			<Select.Content>
				{#each devices as device (device.deviceId)}
					<Select.Item value={device.deviceId} label={device.label || device.deviceId}>
						{device.label || device.deviceId}
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>

	<!-- Source Language -->
	<div class="toolbar-group">
		<span class="toolbar-label">Source</span>
		<Select.Root type="single" bind:value={sourceLanguageValue}>
			<Select.Trigger class="toolbar-select">
				{SOURCE_LANGUAGES.find((l) => l.value === sourceLanguageValue)?.label ?? 'Auto Detect'}
			</Select.Trigger>
			<Select.Content>
				{#each SOURCE_LANGUAGES as lang (lang.value)}
					<Select.Item value={lang.value} label={lang.label}>{lang.label}</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>

	<!-- Target Language -->
	<div class="toolbar-group">
		<span class="toolbar-label">Target</span>
		<Select.Root type="single" bind:value={targetLanguageValue}>
			<Select.Trigger class="toolbar-select">
				{TARGET_LANGUAGES.find((l) => l.value === targetLanguageValue)?.label ?? 'English'}
			</Select.Trigger>
			<Select.Content>
				{#each TARGET_LANGUAGES as lang (lang.value)}
					<Select.Item value={lang.value} label={lang.label}>{lang.label}</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>

	<!-- Model Override -->
	<div class="toolbar-group">
		<span class="toolbar-label">Model</span>
		<Select.Root type="single" bind:value={modelOverride}>
			<Select.Trigger class="toolbar-select">
				{MODELS.find((m) => m.value === modelOverride)?.label ?? 'Auto'}
			</Select.Trigger>
			<Select.Content>
				{#each MODELS as model (model.value)}
					<Select.Item value={model.value} label={model.label}>{model.label}</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>

	<!-- Display Mode -->
	<div class="toolbar-group">
		<span class="toolbar-label">Display</span>
		<!-- I5: Use radiogroup semantics for display mode switcher -->
		<div class="display-mode-switcher" role="radiogroup" aria-label="Display mode">
			<button
				class="display-mode-btn"
				class:active={loopbackStore.displayMode === 'split'}
				role="radio"
				aria-checked={loopbackStore.displayMode === 'split'}
				onclick={() => { loopbackStore.displayMode = 'split'; }}
			>Split</button>
			<button
				class="display-mode-btn"
				class:active={loopbackStore.displayMode === 'subtitle'}
				role="radio"
				aria-checked={loopbackStore.displayMode === 'subtitle'}
				onclick={() => { loopbackStore.displayMode = 'subtitle'; }}
			>Subtitle</button>
			<button
				class="display-mode-btn"
				class:active={loopbackStore.displayMode === 'transcript'}
				role="radio"
				aria-checked={loopbackStore.displayMode === 'transcript'}
				onclick={() => { loopbackStore.displayMode = 'transcript'; }}
			>Transcript</button>
		</div>
	</div>

	<!-- Connection Status + Level -->
	<div class="toolbar-group">
		<span class="toolbar-label">Status</span>
		<div class="status-dots">
			<span class="status-dot" role="status" aria-label="Speech-to-text: {loopbackStore.transcriptionStatus}" style="background-color: {statusColor(loopbackStore.transcriptionStatus)};"></span>
			<span class="status-label">STT</span>
			<span class="status-dot" role="status" aria-label="Machine translation: {loopbackStore.translationStatus}" style="background-color: {statusColor(loopbackStore.translationStatus)};"></span>
			<span class="status-label">MT</span>
		</div>
	</div>

	<!-- Audio Level + Pipeline Counters -->
	{#if loopbackStore.isCapturing}
		<div class="toolbar-group">
			<span class="toolbar-label">Level</span>
			<div class="vu-meter" role="meter" aria-label="Audio input level" aria-valuenow={levelPercent} aria-valuemin={0} aria-valuemax={100}>
				<div class="vu-bar" style="width: {levelPercent}%; background-color: {levelColor};"></div>
			</div>
		</div>
		<div class="toolbar-group">
			<span class="toolbar-label">Pipeline</span>
			<div class="pipeline-counters">
				<span class="counter" title="Audio chunks sent to server">{loopbackStore.chunksSent} <span class="counter-label">chunks</span></span>
				<span class="counter-sep">&rarr;</span>
				<span class="counter" title="Transcription segments received">{loopbackStore.segmentsReceived} <span class="counter-label">segs</span></span>
				<span class="counter-sep">&rarr;</span>
				<span class="counter" title="Translations received">{loopbackStore.translationsReceived} <span class="counter-label">xlat</span></span>
			</div>
		</div>
	{/if}

	<!-- Capture Controls -->
	<div class="toolbar-group toolbar-actions">
		{#if isDemoRunning}
			<Button variant="destructive" size="sm" onclick={onStopDemo}>
				Stop Demo
			</Button>
		{:else if loopbackStore.isCapturing}
			<Button variant="destructive" size="sm" onclick={onStopCapture}>
				Stop Capture
			</Button>
		{:else}
			<Button variant="outline" size="sm" onclick={onStartDemo}>
				Demo
			</Button>
			<Button variant="default" size="sm" onclick={onStartCapture}>
				Start Capture
			</Button>
		{/if}

		{#if !isDemoRunning}
			{#if loopbackStore.isMeetingActive}
				<Button
					variant="destructive"
					size="sm"
					onclick={(e: MouseEvent) => { e.stopPropagation(); showEndMeetingDialog = true; }}
				>
					End Meeting
				</Button>
			{:else}
				<Button variant="secondary" size="sm" onclick={onStartMeeting}>
					Start Meeting
				</Button>
			{/if}
		{/if}
	</div>
</div>

<!-- End Meeting Confirmation Dialog -->
<Dialog.Root bind:open={showEndMeetingDialog}>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>End Meeting</Dialog.Title>
			<Dialog.Description>
				Are you sure you want to end this meeting? Recording and transcription will stop,
				and the session will be finalized.
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer>
			<Button variant="ghost" onclick={() => { showEndMeetingDialog = false; }}>
				Cancel
			</Button>
			<Button variant="destructive" onclick={handleEndMeetingConfirm}>
				End Meeting
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<style>
	.toolbar {
		display: flex;
		flex-wrap: wrap;
		align-items: end;
		gap: 1rem;
		padding: 0.75rem 1rem;
		border-bottom: 1px solid var(--border, #333);
		background: var(--bg-secondary, #1e293b);
	}

	.toolbar-group {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.toolbar-label {
		font-size: 0.675rem;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-muted, #94a3b8);
	}

	.toolbar-actions {
		flex-direction: row;
		align-items: center;
		gap: 0.5rem;
		margin-left: auto;
	}

	:global(.toolbar-select) {
		min-width: 8rem;
	}


	.display-mode-switcher {
		display: flex;
		border: 1px solid var(--border, #333);
		border-radius: 0.375rem;
		overflow: hidden;
	}

	.display-mode-btn {
		padding: 0.25rem 0.625rem;
		font-size: 0.8125rem;
		background: transparent;
		color: var(--text-muted, #94a3b8);
		border: none;
		cursor: pointer;
		transition: background-color 0.15s, color 0.15s;
	}

	.display-mode-btn:not(:last-child) {
		border-right: 1px solid var(--border, #333);
	}

	.display-mode-btn:hover {
		background: var(--bg-hover, rgba(255, 255, 255, 0.05));
	}

	.display-mode-btn.active {
		background: var(--primary, #3b82f6);
		color: white;
	}

	.status-dots {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.25rem 0;
	}

	.status-dot {
		display: inline-block;
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 50%;
	}

	.status-label {
		font-size: 0.6875rem;
		color: var(--text-muted, #94a3b8);
		margin-right: 0.25rem;
	}

	.vu-meter {
		width: 5rem;
		height: 0.5rem;
		background: var(--bg-hover, rgba(255, 255, 255, 0.05));
		border-radius: 0.25rem;
		overflow: hidden;
		margin-top: 0.125rem;
	}

	.vu-bar {
		height: 100%;
		border-radius: 0.25rem;
		transition: width 80ms linear;
	}

	.pipeline-counters {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.6875rem;
		font-variant-numeric: tabular-nums;
		padding: 0.125rem 0;
	}

	.counter {
		color: var(--text, #e2e8f0);
	}

	.counter-label {
		color: var(--text-muted, #94a3b8);
	}

	.counter-sep {
		color: var(--text-muted, #94a3b8);
		font-size: 0.5625rem;
	}
</style>
