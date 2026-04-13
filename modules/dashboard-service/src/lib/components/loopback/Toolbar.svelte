<script lang="ts">
	import { untrack } from 'svelte';
	import { onMount } from 'svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Select from '$lib/components/ui/select';
	import * as Dialog from '$lib/components/ui/dialog';
	import { captionStore, type DisplayMode } from '$lib/stores/caption.svelte';
	import ChevronDownIcon from '@lucide/svelte/icons/chevron-down';
	import ChevronUpIcon from '@lucide/svelte/icons/chevron-up';

	let toolbarExpanded = $state(true);
	let screenCaptureAvailable = $state(false);

	onMount(async () => {
		try {
			const res = await fetch('/api/system/screencapture-available');
			if (res.ok) {
				const data = await res.json() as { available?: boolean };
				screenCaptureAvailable = data.available ?? false;
			}
		} catch {
			// Endpoint not yet available — treat as not installed
			screenCaptureAvailable = false;
		}
	});

	interface Props {
		devices: MediaDeviceInfo[];
		selectedDeviceId?: string;
		onDeviceChange?: (deviceId: string) => void;
		onStartCapture?: () => void;
		onStopCapture?: () => void;
		onStartMeeting?: () => void;
		onEndMeeting?: () => void;
		/** I1: Callback to send config changes to the server via WebSocket */
		onConfigChange?: (config: { model?: string; language?: string | null; target_language?: string; interpreter_languages?: [string, string] | null }) => void;
		onStartDemo?: () => void;
		onStopDemo?: () => void;
		isDemoRunning?: boolean;
		/** True while draining pending translations after stop */
		isDraining?: boolean;
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
		isDraining = false,
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
		{ value: 'interpreter', label: 'Interpreter' },
	];

	let showEndMeetingDialog = $state(false);

	// Fireflies meeting picker
	interface FirefliesMeeting {
		id: string;
		title: string;
		started_at: string | null;
		organizer: string;
		state: string;
	}

	let meetings = $state<FirefliesMeeting[]>([]);
	let loadingMeetings = $state(false);
	let selectedMeetingId = $state<string | null>(null);

	function formatTimeAgo(dateStr: string | null): string {
		if (!dateStr) return '';
		const diff = Date.now() - new Date(dateStr).getTime();
		const mins = Math.floor(diff / 60000);
		if (mins < 1) return 'just now';
		if (mins < 60) return `${mins}m ago`;
		const hours = Math.floor(mins / 60);
		if (hours < 24) return `${hours}h ago`;
		return `${Math.floor(hours / 24)}d ago`;
	}

	$effect(() => {
		if (captionStore.captionSource !== 'fireflies') return;
		loadingMeetings = true;
		fetch('/api/fireflies/sessions/active')
			.then(r => r.ok ? r.json() as Promise<{ meetings?: FirefliesMeeting[]; auto_select_id?: string }> : Promise.reject(r.status))
			.then(data => {
				meetings = data.meetings ?? [];
				if (meetings.length === 1 && data.auto_select_id) {
					selectedMeetingId = data.auto_select_id;
					captionStore.firefliesSessionId = data.auto_select_id;
				}
			})
			.catch(() => { meetings = []; })
			.finally(() => { loadingMeetings = false; });
	});

	// Local select values bound to store
	let sourceLanguageValue = $state(captionStore.sourceLanguage ?? 'auto');
	let targetLanguageValue = $state(captionStore.targetLanguage);
	let modelOverride = $state('auto');
	let interpreterLangAValue = $state(captionStore.interpreterLangA);
	let interpreterLangBValue = $state(captionStore.interpreterLangB);

	// Whether interpreter mode is active (derived for template conditionals)
	let isInterpreterMode = $derived(captionStore.displayMode === 'interpreter');

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
		captionStore.sourceLanguage = lang;
		// I2: Send config change to server if capturing.
		// Send null too — resets server to auto-detect when switching back from explicit language.
		if (captionStore.isCapturing) {
			onConfigChange?.({ language: lang });
		}
	});

	// Sync target language changes to store and notify server
	$effect(() => {
		const val = targetLanguageValue;
		if (val === prevTargetLang) return;
		prevTargetLang = val;
		captionStore.targetLanguage = val;
		if (captionStore.isCapturing) {
			onConfigChange?.({ target_language: val });
		}
	});

	// I1: Send model override to server when changed during active session
	$effect(() => {
		const val = modelOverride;
		if (val === prevModel) return;
		prevModel = val;
		if (captionStore.isCapturing && val !== 'auto') {
			onConfigChange?.({ model: val });
		}
	});

	// Sync interpreter language A/B to store and send config when changed
	let prevInterpA = $state(untrack(() => interpreterLangAValue));
	let prevInterpB = $state(untrack(() => interpreterLangBValue));

	$effect(() => {
		const a = interpreterLangAValue;
		const b = interpreterLangBValue;
		if (a === prevInterpA && b === prevInterpB) return;
		// Swap guard: if user picks same language for both, swap them
		if (a === b) {
			if (a !== prevInterpA) {
				// A was changed to match B — swap B to old A
				interpreterLangBValue = prevInterpA;
			} else {
				// B was changed to match A — swap A to old B
				interpreterLangAValue = prevInterpB;
			}
			return; // the swap will re-trigger this effect
		}
		prevInterpA = a;
		prevInterpB = b;
		captionStore.interpreterLangA = a;
		captionStore.interpreterLangB = b;
		if (isInterpreterMode && captionStore.isCapturing) {
			onConfigChange?.({ language: undefined, interpreter_languages: [a, b] });
		}
	});

	// When entering/leaving interpreter mode, send appropriate config
	let prevDisplayMode = $state(untrack(() => captionStore.displayMode));
	$effect(() => {
		const mode = captionStore.displayMode;
		if (mode === prevDisplayMode) return;
		const wasInterpreter = prevDisplayMode === 'interpreter';
		prevDisplayMode = mode;
		if (!captionStore.isCapturing) return;
		if (mode === 'interpreter') {
			// Entering interpreter mode — send interpreter_languages, force auto-detect
			onConfigChange?.({ language: undefined, interpreter_languages: [interpreterLangAValue, interpreterLangBValue] });
		} else if (wasInterpreter) {
			// Leaving interpreter mode — clear interpreter state, restore target language
			onConfigChange?.({ interpreter_languages: null, target_language: captionStore.targetLanguage });
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

<div class="toolbar" class:toolbar-collapsed={!toolbarExpanded}>
	<!-- Toggle button -->
	<button
		class="toolbar-toggle"
		title={toolbarExpanded ? 'Collapse settings' : 'Expand settings'}
		onclick={() => { toolbarExpanded = !toolbarExpanded; }}
	>
		{#if toolbarExpanded}
			<ChevronUpIcon class="size-3" />
		{:else}
			<ChevronDownIcon class="size-3" />
		{/if}
	</button>

	{#if toolbarExpanded}
	<!-- Caption Source Selector -->
	<div class="toolbar-group">
		<span class="toolbar-label">Source</span>
		<div class="source-options" role="radiogroup" aria-label="Caption source">
			<label class="source-option">
				<input
					type="radio"
					name="caption-source"
					value="local"
					checked={captionStore.captionSource === 'local'}
					onchange={() => { captionStore.captionSource = 'local'; }}
					disabled={captionStore.isCapturing}
				/>
				<span>Mic</span>
			</label>
			<label class="source-option" class:source-option-disabled={!screenCaptureAvailable}>
				<input
					type="radio"
					name="caption-source"
					value="screencapture"
					checked={captionStore.captionSource === 'screencapture'}
					onchange={() => { captionStore.captionSource = 'screencapture'; }}
					disabled={captionStore.isCapturing || !screenCaptureAvailable}
				/>
				<span>System Audio</span>
				{#if !screenCaptureAvailable}<span class="source-badge">Install</span>{/if}
			</label>
			<label class="source-option">
				<input
					type="radio"
					name="caption-source"
					value="fireflies"
					checked={captionStore.captionSource === 'fireflies'}
					onchange={() => { captionStore.captionSource = 'fireflies'; }}
					disabled={captionStore.isCapturing}
				/>
				<span>Fireflies</span>
			</label>
		</div>
	</div>

	<!-- Fireflies Meeting Picker -->
	{#if captionStore.captionSource === 'fireflies'}
	<div class="toolbar-group">
		<span class="toolbar-label">Meeting</span>
		{#if loadingMeetings}
			<span class="fireflies-status">Loading...</span>
		{:else if meetings.length === 0}
			<span class="fireflies-status fireflies-status-empty">No active meetings</span>
		{:else if meetings.length === 1}
			<span class="fireflies-auto-selected" title={meetings[0].title}>
				{meetings[0].title}
				{#if meetings[0].started_at}<span class="fireflies-time">{formatTimeAgo(meetings[0].started_at)}</span>{/if}
			</span>
		{:else}
			<Select.Root
				type="single"
				value={selectedMeetingId ?? ''}
				onValueChange={(v) => {
					if (v) {
						selectedMeetingId = v;
						captionStore.firefliesSessionId = v;
					}
				}}
			>
				<Select.Trigger class="toolbar-select">
					{meetings.find(m => m.id === selectedMeetingId)?.title ?? 'Select meeting'}
				</Select.Trigger>
				<Select.Content>
					{#each meetings as meeting (meeting.id)}
						<Select.Item value={meeting.id} label={meeting.title}>
							<span class="meeting-option-title">{meeting.title}</span>
							{#if meeting.started_at}
								<span class="meeting-option-time">{formatTimeAgo(meeting.started_at)}</span>
							{/if}
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		{/if}
	</div>
	{/if}

	<!-- Audio Source (only for local mic mode) -->
	{#if captionStore.captionSource === 'local'}
	<div class="toolbar-group">
		<span class="toolbar-label">Audio Device</span>
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
	{/if}

	{#if isInterpreterMode}
		<!-- Interpreter: Language A -->
		<div class="toolbar-group">
			<span class="toolbar-label">Language A</span>
			<Select.Root type="single" bind:value={interpreterLangAValue}>
				<Select.Trigger class="toolbar-select">
					{TARGET_LANGUAGES.find((l) => l.value === interpreterLangAValue)?.label ?? interpreterLangAValue}
				</Select.Trigger>
				<Select.Content>
					{#each TARGET_LANGUAGES as lang (lang.value)}
						<Select.Item value={lang.value} label={lang.label}>{lang.label}</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		</div>

		<!-- Interpreter: Language B -->
		<div class="toolbar-group">
			<span class="toolbar-label">Language B</span>
			<Select.Root type="single" bind:value={interpreterLangBValue}>
				<Select.Trigger class="toolbar-select">
					{TARGET_LANGUAGES.find((l) => l.value === interpreterLangBValue)?.label ?? interpreterLangBValue}
				</Select.Trigger>
				<Select.Content>
					{#each TARGET_LANGUAGES as lang (lang.value)}
						<Select.Item value={lang.value} label={lang.label}>{lang.label}</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		</div>
	{:else}
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
	{/if}

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

	{/if}

	<!-- Display Mode (always visible) -->
	<div class="toolbar-group">
		<span class="toolbar-label">Display</span>
		<!-- I5: Use radiogroup semantics for display mode switcher -->
		<div class="display-mode-switcher" role="radiogroup" aria-label="Display mode">
			{#each DISPLAY_MODES as mode (mode.value)}
				<button
					class="display-mode-btn"
					class:active={captionStore.displayMode === mode.value}
					role="radio"
					aria-checked={captionStore.displayMode === mode.value}
					onclick={() => { captionStore.displayMode = mode.value; }}
				>{mode.label}</button>
			{/each}
		</div>
	</div>

	<!-- Connection Status + Level -->
	<div class="toolbar-group">
		<span class="toolbar-label">Status</span>
		<div class="status-dots">
			<span class="status-dot" role="status" aria-label="Speech-to-text: {captionStore.transcriptionStatus}" style="background-color: {statusColor(captionStore.transcriptionStatus)};"></span>
			<span class="status-label">STT</span>
			<span class="status-dot" role="status" aria-label="Machine translation: {captionStore.translationStatus}" style="background-color: {statusColor(captionStore.translationStatus)};"></span>
			<span class="status-label">MT</span>
		</div>
	</div>

	<!-- Audio Level + Pipeline Counters -->
	{#if captionStore.isCapturing}
		<div class="toolbar-group">
			<span class="toolbar-label">Level</span>
			<div class="vu-meter" role="meter" aria-label="Audio input level" aria-valuenow={levelPercent} aria-valuemin={0} aria-valuemax={100}>
				<div class="vu-bar" style="width: {levelPercent}%; background-color: {levelColor};"></div>
			</div>
		</div>
		<div class="toolbar-group">
			<span class="toolbar-label">Pipeline</span>
			<div class="pipeline-counters">
				<span class="counter" title="Audio chunks sent to server">{captionStore.chunksSent} <span class="counter-label">chunks</span></span>
				<span class="counter-sep">&rarr;</span>
				<span class="counter" title="Transcription segments received">{captionStore.segmentsReceived} <span class="counter-label">segs</span></span>
				<span class="counter-sep">&rarr;</span>
				<span class="counter" title="Translations received">{captionStore.translationsReceived} <span class="counter-label">xlat</span></span>
			</div>
		</div>
	{/if}

	<!-- Capture Controls -->
	<div class="toolbar-group toolbar-actions">
		{#if isDemoRunning}
			<Button variant="destructive" size="sm" onclick={onStopDemo}>
				Stop Demo
			</Button>
		{:else if isDraining}
			<Button variant="ghost" size="sm" disabled>
				<span class="drain-spinner"></span> Finishing...
			</Button>
		{:else if captionStore.isCapturing}
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
			{#if captionStore.isMeetingActive}
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
		position: relative;
	}

	.toolbar-collapsed {
		padding: 0.5rem 1rem;
	}

	.toolbar-toggle {
		position: absolute;
		top: 0.25rem;
		right: 0.5rem;
		padding: 0.125rem;
		border: none;
		background: transparent;
		color: var(--text-muted, #94a3b8);
		cursor: pointer;
		opacity: 0.5;
		transition: opacity 0.15s;
		z-index: 1;
	}

	.toolbar-toggle:hover {
		opacity: 1;
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

	.drain-spinner {
		display: inline-block;
		width: 12px;
		height: 12px;
		border: 2px solid var(--text-muted, #94a3b8);
		border-top-color: transparent;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
		margin-right: 4px;
		vertical-align: middle;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.source-options {
		display: flex;
		gap: 0.25rem;
		flex-wrap: wrap;
	}

	.source-option {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.25rem 0.5rem;
		font-size: 0.8125rem;
		border: 1px solid var(--border, #333);
		border-radius: 0.375rem;
		cursor: pointer;
		color: var(--text, #e2e8f0);
		background: transparent;
		transition: background-color 0.15s, border-color 0.15s;
		user-select: none;
		white-space: nowrap;
	}

	.source-option:hover:not(.source-option-disabled) {
		background: var(--bg-hover, rgba(255, 255, 255, 0.05));
	}

	.source-option input[type="radio"] {
		accent-color: var(--primary, #3b82f6);
		width: 0.75rem;
		height: 0.75rem;
		flex-shrink: 0;
	}

	.source-option-disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}

	.source-badge {
		font-size: 0.625rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 0.0625rem 0.3125rem;
		border-radius: 0.25rem;
		background: var(--warning, #f59e0b);
		color: #000;
		margin-left: 0.125rem;
	}

	.fireflies-status {
		font-size: 0.8125rem;
		color: var(--text-muted, #94a3b8);
		padding: 0.25rem 0;
	}

	.fireflies-status-empty {
		font-style: italic;
	}

	.fireflies-auto-selected {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		font-size: 0.8125rem;
		color: var(--text, #e2e8f0);
		padding: 0.25rem 0.5rem;
		border: 1px solid var(--primary, #3b82f6);
		border-radius: 0.375rem;
		max-width: 14rem;
		overflow: hidden;
		white-space: nowrap;
		text-overflow: ellipsis;
	}

	.fireflies-time {
		font-size: 0.6875rem;
		color: var(--text-muted, #94a3b8);
		flex-shrink: 0;
	}

	.meeting-option-title {
		font-size: 0.8125rem;
	}

	.meeting-option-time {
		font-size: 0.6875rem;
		color: var(--text-muted, #94a3b8);
		margin-left: 0.375rem;
	}
</style>
