<script lang="ts">
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
		onStart?: () => void;
		onStop?: () => void;
		onStartMeeting?: () => void;
		onEndMeeting?: () => void;
	}

	let {
		devices,
		selectedDeviceId = $bindable(''),
		onDeviceChange,
		onStartCapture,
		onStopCapture,
		onStart,
		onStop,
		onStartMeeting,
		onEndMeeting,
	}: Props = $props();

	function handleStartCapture() {
		(onStartCapture ?? onStart)?.();
	}

	function handleStopCapture() {
		(onStopCapture ?? onStop)?.();
	}

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

	// Sync source language changes to store
	$effect(() => {
		loopbackStore.sourceLanguage = sourceLanguageValue === 'auto' ? null : sourceLanguageValue;
	});

	// Sync target language changes to store
	$effect(() => {
		loopbackStore.targetLanguage = targetLanguageValue;
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
		<label class="toolbar-label">Audio Source</label>
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
		<label class="toolbar-label">Source</label>
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
		<label class="toolbar-label">Target</label>
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
		<label class="toolbar-label">Model</label>
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
		<label class="toolbar-label">Display</label>
		<div class="display-mode-switcher">
			{#each DISPLAY_MODES as mode (mode.value)}
				<button
					class="display-mode-btn"
					class:active={loopbackStore.displayMode === mode.value}
					onclick={() => { loopbackStore.displayMode = mode.value; }}
				>
					{mode.label}
				</button>
			{/each}
		</div>
	</div>

	<!-- Connection Status -->
	<div class="toolbar-group">
		<label class="toolbar-label">Status</label>
		<div class="status-dots">
			<span class="status-dot" style="background-color: {statusColor(loopbackStore.transcriptionStatus)};" title="STT: {loopbackStore.transcriptionStatus}"></span>
			<span class="status-label">STT</span>
			<span class="status-dot" style="background-color: {statusColor(loopbackStore.translationStatus)};" title="MT: {loopbackStore.translationStatus}"></span>
			<span class="status-label">MT</span>
		</div>
	</div>

	<!-- Capture Controls -->
	<div class="toolbar-group toolbar-actions">
		{#if loopbackStore.isCapturing}
			<Button variant="destructive" size="sm" onclick={handleStopCapture}>
				Stop Capture
			</Button>
		{:else}
			<Button variant="default" size="sm" onclick={handleStartCapture}>
				Start Capture
			</Button>
		{/if}

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
</style>
