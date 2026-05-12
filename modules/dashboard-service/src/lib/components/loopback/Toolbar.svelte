<script lang="ts">
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
		onConfigChange?: (config: {
			model?: string;
			language?: string | null;
			target_language?: string;
			interpreter_languages?: [string, string] | null;
			/** Per-session LLM tunables — temperature, max_tokens, etc.
			 *  See `LLMOverridesMessage` in $lib/types/ws-messages. */
			llm?: {
				connection_id?: string | null;
				model?: string | null;
				temperature?: number | null;
				max_tokens?: number | null;
				top_p?: number | null;
				top_k?: number | null;
				repetition_penalty?: number | null;
				presence_penalty?: number | null;
			};
			/** Per-session Whisper decoding tunables — beam_size, language_hint, etc.
			 *  See `WhisperOverridesMessage` in $lib/types/ws-messages. */
			whisper?: {
				connection_id?: string | null;
				model?: string | null;
				temperature?: number | null;
				beam_size?: number | null;
				no_speech_threshold?: number | null;
				compression_ratio_threshold?: number | null;
				language_hint?: string | null;
				initial_prompt?: string | null;
			};
		}) => void;
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
		levelPercent > 90 ? 'var(--oxblood)' :
		levelPercent > 5 ? 'var(--peach-deep)' :
		'var(--ink-faint)'
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
		{ value: 'wire', label: 'Wire' },
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

	// Local select values mirror the store. Kept separate so dropdown UI
	// settles independently of the store, but writes are event-driven (no
	// $effect-watch pattern) — see handle* functions below.
	let sourceLanguageValue = $state(captionStore.sourceLanguage ?? 'auto');
	let targetLanguageValue = $state(captionStore.targetLanguage);
	let modelOverride = $state('auto');
	let interpreterLangAValue = $state(captionStore.interpreterLangA);
	let interpreterLangBValue = $state(captionStore.interpreterLangB);

	let isInterpreterMode = $derived(captionStore.displayMode === 'interpreter');

	// Event-driven side effects. Replaces a previous set of $effect blocks
	// that watched these values via prev* state variables — the autofixer
	// flagged that pattern as state-writes-inside-$effect, and the cost
	// compounded with the captionStore reactivity surface (Toolbar reads
	// chunksSent every audio frame).
	//
	// All handlers run synchronously from a user interaction (radio change /
	// select change / button click), so the side effect happens once per
	// user action, with no reactive cascade.

	function loadFirefliesMeetings(): void {
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
	}

	function handleCaptionSourceChange(source: 'local' | 'screencapture' | 'fireflies'): void {
		captionStore.captionSource = source;
		if (source === 'fireflies') {
			loadFirefliesMeetings();
		}
	}

	function handleSourceLanguageChange(val: string): void {
		sourceLanguageValue = val;
		const lang = val === 'auto' ? null : val;
		captionStore.sourceLanguage = lang;
		// I2: send null too so server clears any forced language and returns to auto-detect.
		if (captionStore.isCapturing) {
			onConfigChange?.({ language: lang });
		}
	}

	function handleTargetLanguageChange(val: string): void {
		targetLanguageValue = val;
		captionStore.targetLanguage = val;
		if (captionStore.isCapturing) {
			onConfigChange?.({ target_language: val });
		}
	}

	function handleModelChange(val: string): void {
		modelOverride = val;
		if (captionStore.isCapturing && val !== 'auto') {
			onConfigChange?.({ model: val });
		}
	}

	function handleInterpreterAChange(val: string): void {
		// Swap guard: if user picks same as B, swap B to the old A.
		if (val === interpreterLangBValue) {
			interpreterLangBValue = interpreterLangAValue;
			captionStore.interpreterLangB = interpreterLangAValue;
		}
		interpreterLangAValue = val;
		captionStore.interpreterLangA = val;
		if (isInterpreterMode && captionStore.isCapturing) {
			onConfigChange?.({ language: undefined, interpreter_languages: [interpreterLangAValue, interpreterLangBValue] });
		}
	}

	function handleInterpreterBChange(val: string): void {
		if (val === interpreterLangAValue) {
			interpreterLangAValue = interpreterLangBValue;
			captionStore.interpreterLangA = interpreterLangBValue;
		}
		interpreterLangBValue = val;
		captionStore.interpreterLangB = val;
		if (isInterpreterMode && captionStore.isCapturing) {
			onConfigChange?.({ language: undefined, interpreter_languages: [interpreterLangAValue, interpreterLangBValue] });
		}
	}

	/** LLM tunables — bound to advanced disclosure section. Persisted via
	 *  captionStore.updateLLMOverrides which writes to localStorage and the
	 *  reactive `llm` slice. When capturing, every change pushes the patch
	 *  through ConfigMessage.llm so the orchestration session-config picks
	 *  up the override for subsequent translations. */
	function handleLLMOverride(patch: {
		temperature?: number | null;
		maxTokens?: number | null;
		topP?: number | null;
		topK?: number | null;
		repetitionPenalty?: number | null;
		presencePenalty?: number | null;
		connectionId?: string | null;
		model?: string | null;
	}): void {
		captionStore.updateLLMOverrides(patch);
		if (!captionStore.isCapturing) return;
		const wsPatch: NonNullable<Parameters<NonNullable<typeof onConfigChange>>[0]['llm']> = {};
		if (patch.temperature !== undefined) wsPatch.temperature = patch.temperature;
		if (patch.maxTokens !== undefined) wsPatch.max_tokens = patch.maxTokens;
		if (patch.topP !== undefined) wsPatch.top_p = patch.topP;
		if (patch.topK !== undefined) wsPatch.top_k = patch.topK;
		if (patch.repetitionPenalty !== undefined) wsPatch.repetition_penalty = patch.repetitionPenalty;
		if (patch.presencePenalty !== undefined) wsPatch.presence_penalty = patch.presencePenalty;
		if (patch.connectionId !== undefined) wsPatch.connection_id = patch.connectionId;
		if (patch.model !== undefined) wsPatch.model = patch.model;
		onConfigChange?.({ llm: wsPatch });
	}

	/** Whisper tunables — mirror of handleLLMOverride for the transcription side.
	 *  Persists to localStorage via captionStore.updateWhisperOverrides and, when
	 *  capturing, pushes the patch through ConfigMessage.whisper so the orchestration
	 *  session-config carries the override for subsequent transcribe calls. */
	function handleWhisperOverride(patch: {
		temperature?: number | null;
		beamSize?: number | null;
		noSpeechThreshold?: number | null;
		compressionRatioThreshold?: number | null;
		languageHint?: string | null;
		initialPrompt?: string | null;
		connectionId?: string | null;
		model?: string | null;
	}): void {
		captionStore.updateWhisperOverrides(patch);
		if (!captionStore.isCapturing) return;
		const wsPatch: NonNullable<Parameters<NonNullable<typeof onConfigChange>>[0]['whisper']> = {};
		if (patch.temperature !== undefined) wsPatch.temperature = patch.temperature;
		if (patch.beamSize !== undefined) wsPatch.beam_size = patch.beamSize;
		if (patch.noSpeechThreshold !== undefined) wsPatch.no_speech_threshold = patch.noSpeechThreshold;
		if (patch.compressionRatioThreshold !== undefined) wsPatch.compression_ratio_threshold = patch.compressionRatioThreshold;
		if (patch.languageHint !== undefined) wsPatch.language_hint = patch.languageHint;
		if (patch.initialPrompt !== undefined) wsPatch.initial_prompt = patch.initialPrompt;
		if (patch.connectionId !== undefined) wsPatch.connection_id = patch.connectionId;
		if (patch.model !== undefined) wsPatch.model = patch.model;
		onConfigChange?.({ whisper: wsPatch });
	}

	function handleDisplayModeChange(mode: DisplayMode): void {
		const wasInterpreter = captionStore.displayMode === 'interpreter';
		captionStore.displayMode = mode;
		if (!captionStore.isCapturing) return;
		if (mode === 'interpreter') {
			onConfigChange?.({ language: undefined, interpreter_languages: [interpreterLangAValue, interpreterLangBValue] });
		} else if (wasInterpreter) {
			onConfigChange?.({ interpreter_languages: null, target_language: captionStore.targetLanguage });
		}
	}

	function statusColor(status: 'up' | 'down'): string {
		// Editorial palette: sage = healthy, oxblood = down
		return status === 'up' ? 'var(--sage)' : 'var(--oxblood)';
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
					onchange={() => handleCaptionSourceChange('local')}
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
					onchange={() => handleCaptionSourceChange('screencapture')}
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
					onchange={() => handleCaptionSourceChange('fireflies')}
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
			<Select.Root type="single" value={interpreterLangAValue} onValueChange={(v) => { if (v) handleInterpreterAChange(v); }}>
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
			<Select.Root type="single" value={interpreterLangBValue} onValueChange={(v) => { if (v) handleInterpreterBChange(v); }}>
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
			<Select.Root type="single" value={sourceLanguageValue} onValueChange={(v) => { if (v) handleSourceLanguageChange(v); }}>
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
			<Select.Root type="single" value={targetLanguageValue} onValueChange={(v) => { if (v) handleTargetLanguageChange(v); }}>
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
		<Select.Root type="single" value={modelOverride} onValueChange={(v) => { if (v) handleModelChange(v); }}>
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

	<!-- LLM Sampling Tunables (per-session overrides) -->
	<details class="toolbar-llm-tunables">
		<summary class="toolbar-llm-summary">
			<span class="toolbar-label">LLM Sampling</span>
			<span class="toolbar-llm-hint">tune temperature, tokens, etc.</span>
		</summary>
		<div class="toolbar-llm-grid">
			<label class="toolbar-llm-field">
				<span>Temperature</span>
				<input
					type="number"
					min="0" max="2" step="0.05"
					placeholder="0.3"
					value={captionStore.llm.temperature ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleLLMOverride({ temperature: v });
					}}
				/>
			</label>
			<label class="toolbar-llm-field">
				<span>Max tokens</span>
				<input
					type="number"
					min="1" max="4096" step="16"
					placeholder="1024"
					value={captionStore.llm.maxTokens ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleLLMOverride({ maxTokens: v });
					}}
				/>
			</label>
			<label class="toolbar-llm-field">
				<span>Top P</span>
				<input
					type="number"
					min="0" max="1" step="0.05"
					placeholder="0.8"
					value={captionStore.llm.topP ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleLLMOverride({ topP: v });
					}}
				/>
			</label>
			<label class="toolbar-llm-field">
				<span>Top K</span>
				<input
					type="number"
					min="0" max="200" step="1"
					placeholder="20"
					value={captionStore.llm.topK ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleLLMOverride({ topK: v });
					}}
				/>
			</label>
			<label class="toolbar-llm-field">
				<span>Rep. penalty</span>
				<input
					type="number"
					min="0" max="2" step="0.05"
					placeholder="1.05"
					value={captionStore.llm.repetitionPenalty ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleLLMOverride({ repetitionPenalty: v });
					}}
				/>
			</label>
			<label class="toolbar-llm-field">
				<span>Pres. penalty</span>
				<input
					type="number"
					min="-2" max="2" step="0.05"
					placeholder="1.5"
					value={captionStore.llm.presencePenalty ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleLLMOverride({ presencePenalty: v });
					}}
				/>
			</label>
		</div>
		<button
			type="button"
			class="toolbar-llm-reset"
			onclick={() => {
				captionStore.resetLLMOverrides();
				if (captionStore.isCapturing) {
					onConfigChange?.({
						llm: {
							temperature: null, max_tokens: null,
							top_p: null, top_k: null,
							repetition_penalty: null, presence_penalty: null,
						},
					});
				}
			}}
		>Reset to backend defaults</button>
	</details>

	<!-- Whisper Decoding Tunables (per-session overrides) -->
	<details class="toolbar-llm-tunables">
		<summary class="toolbar-llm-summary">
			<span class="toolbar-label">Whisper</span>
			<span class="toolbar-llm-hint">beam, prompt, language hint…</span>
		</summary>
		<div class="toolbar-llm-grid">
			<label class="toolbar-llm-field">
				<span>Beam size</span>
				<input
					type="number"
					min="1" max="10" step="1"
					placeholder="1"
					value={captionStore.whisper.beamSize ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleWhisperOverride({ beamSize: v });
					}}
				/>
			</label>
			<label class="toolbar-llm-field">
				<span>Temperature</span>
				<input
					type="number"
					min="0" max="1" step="0.05"
					placeholder="0.0"
					value={captionStore.whisper.temperature ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleWhisperOverride({ temperature: v });
					}}
				/>
			</label>
			<label class="toolbar-llm-field">
				<span>No-speech thresh.</span>
				<input
					type="number"
					min="0" max="1" step="0.05"
					placeholder="0.6"
					value={captionStore.whisper.noSpeechThreshold ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleWhisperOverride({ noSpeechThreshold: v });
					}}
				/>
			</label>
			<label class="toolbar-llm-field">
				<span>Compression thresh.</span>
				<input
					type="number"
					min="1" max="5" step="0.1"
					placeholder="2.4"
					value={captionStore.whisper.compressionRatioThreshold ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						const v = raw === '' ? null : Number(raw);
						handleWhisperOverride({ compressionRatioThreshold: v });
					}}
				/>
			</label>
			<label class="toolbar-llm-field">
				<span>Language hint</span>
				<input
					type="text"
					maxlength="8"
					placeholder="auto"
					value={captionStore.whisper.languageHint ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value.trim();
						handleWhisperOverride({ languageHint: raw === '' ? null : raw });
					}}
				/>
			</label>
			<label class="toolbar-llm-field toolbar-llm-field-wide">
				<span>Initial prompt</span>
				<input
					type="text"
					maxlength="220"
					placeholder="(none)"
					value={captionStore.whisper.initialPrompt ?? ''}
					oninput={(e) => {
						const raw = (e.target as HTMLInputElement).value;
						handleWhisperOverride({ initialPrompt: raw === '' ? null : raw });
					}}
				/>
			</label>
		</div>
		<button
			type="button"
			class="toolbar-llm-reset"
			onclick={() => {
				captionStore.resetWhisperOverrides();
				if (captionStore.isCapturing) {
					onConfigChange?.({
						whisper: {
							temperature: null,
							beam_size: null,
							no_speech_threshold: null,
							compression_ratio_threshold: null,
							language_hint: null,
							initial_prompt: null,
						},
					});
				}
			}}
		>Reset to backend defaults</button>
	</details>

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
					onclick={() => handleDisplayModeChange(mode.value)}
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
	/* ── Editorial toolbar styling — D4.5 ─────────────────────────
	   Reads as the masthead's settings strip: small-caps eyebrow labels,
	   inked CTA buttons, and an editorial peach for the recording state. */
	.toolbar {
		display: flex;
		flex-wrap: wrap;
		align-items: end;
		gap: 1.25rem;
		padding: 0.875rem 1.25rem;
		border-bottom: 1px solid var(--rule);
		background: var(--paper-cream);
		position: relative;
	}

	.toolbar-collapsed {
		padding: 0.5rem 1.25rem;
	}

	.toolbar-toggle {
		position: absolute;
		top: 0.25rem;
		right: 0.5rem;
		padding: 0.125rem;
		border: none;
		background: transparent;
		color: var(--ink-faint);
		cursor: pointer;
		transition: color 160ms ease;
		z-index: 1;
	}
	.toolbar-toggle:hover {
		color: var(--ink-soft);
	}

	.toolbar-group {
		display: flex;
		flex-direction: column;
		gap: 0.375rem;
	}

	/* LLM sampling tunables — collapsed disclosure to keep the toolbar light.
	   Inputs are number-typed so the browser provides spinner UX; the visual
	   weight stays minimal so the section doesn't dominate when expanded. */
	.toolbar-llm-tunables {
		flex: 0 0 100%;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		margin-top: 0.25rem;
		padding: 0.5rem 0.75rem;
		border: 1px dashed var(--rule);
		background: var(--paper-tint, var(--paper-cream));
	}
	.toolbar-llm-summary {
		display: flex;
		align-items: baseline;
		gap: 0.625rem;
		cursor: pointer;
		list-style: none;
		font-variant-caps: small-caps;
		letter-spacing: 0.04em;
	}
	.toolbar-llm-summary::-webkit-details-marker {
		display: none;
	}
	.toolbar-llm-summary::before {
		content: '▸';
		color: var(--ink-faint);
		transition: transform 160ms ease;
		display: inline-block;
		font-size: 0.7rem;
	}
	.toolbar-llm-tunables[open] .toolbar-llm-summary::before {
		transform: rotate(90deg);
	}
	.toolbar-llm-hint {
		color: var(--ink-faint);
		font-size: 0.7rem;
		font-variant-caps: normal;
		letter-spacing: 0;
	}
	.toolbar-llm-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
		gap: 0.5rem 0.875rem;
	}
	.toolbar-llm-field {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		font-size: 0.7rem;
		color: var(--ink-soft);
	}
	.toolbar-llm-field-wide {
		grid-column: 1 / -1;
	}
	.toolbar-llm-field input {
		width: 100%;
		padding: 0.25rem 0.4rem;
		border: 1px solid var(--rule);
		background: var(--paper);
		color: var(--ink);
		font-family: inherit;
		font-size: 0.8rem;
	}
	.toolbar-llm-field input:focus {
		outline: 1px solid var(--ink-faint);
		outline-offset: 0;
	}
	.toolbar-llm-reset {
		align-self: flex-start;
		padding: 0.2rem 0.5rem;
		border: 1px solid var(--rule);
		background: transparent;
		color: var(--ink-soft);
		font-size: 0.7rem;
		font-variant-caps: small-caps;
		letter-spacing: 0.04em;
		cursor: pointer;
	}
	.toolbar-llm-reset:hover {
		color: var(--ink);
		border-color: var(--ink-soft);
	}

	.toolbar-label {
		font-family: var(--font-display);
		font-variation-settings: "opsz" 14;
		font-feature-settings: "smcp", "c2sc", "kern";
		letter-spacing: 0.14em;
		text-transform: lowercase;
		font-weight: 600;
		font-size: 0.6875rem;
		color: var(--ink-faint);
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

	/* ── Display mode switcher — segmented small-caps ─────────────── */
	.display-mode-switcher {
		display: inline-flex;
		border: 1px solid var(--rule);
		border-radius: 0.375rem;
		overflow: hidden;
		background: var(--paper);
	}

	.display-mode-btn {
		padding: 0.375rem 0.75rem;
		font-family: var(--font-display);
		font-variation-settings: "opsz" 14;
		font-feature-settings: "smcp", "c2sc";
		letter-spacing: 0.08em;
		text-transform: lowercase;
		font-size: 0.75rem;
		background: transparent;
		color: var(--ink-soft);
		border: none;
		cursor: pointer;
		transition: background-color 160ms ease, color 160ms ease;
	}
	.display-mode-btn:not(:last-child) {
		border-right: 1px solid var(--rule);
	}
	.display-mode-btn:hover {
		background: var(--paper-cream);
		color: var(--ink);
	}
	.display-mode-btn.active {
		background: var(--ink);
		color: var(--paper);
	}

	/* ── Status pips — earth-tone dots ───────────────────────────── */
	.status-dots {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.25rem 0;
	}
	.status-dot {
		display: inline-block;
		width: 0.4375rem;
		height: 0.4375rem;
		border-radius: 50%;
	}
	.status-label {
		font-family: var(--font-mono);
		font-size: 0.625rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--ink-faint);
		margin-right: 0.25rem;
	}

	/* ── VU meter — single peach bar on a hairline track ───────── */
	.vu-meter {
		width: 5.5rem;
		height: 0.375rem;
		background: var(--paper);
		border: 1px solid var(--rule);
		border-radius: 0.125rem;
		overflow: hidden;
		margin-top: 0.25rem;
	}
	.vu-bar {
		height: 100%;
		transition: width 80ms linear, background-color 200ms ease;
	}

	/* ── Pipeline counters ───────────────────────────────────── */
	.pipeline-counters {
		display: flex;
		align-items: baseline;
		gap: 0.4rem;
		font-family: var(--font-mono);
		font-size: 0.6875rem;
		font-variant-numeric: tabular-nums;
		padding: 0.125rem 0;
	}
	.counter { color: var(--ink); }
	.counter-label {
		color: var(--ink-faint);
		margin-left: 0.125rem;
	}
	.counter-sep {
		color: var(--ink-faint);
		font-size: 0.625rem;
	}

	.drain-spinner {
		display: inline-block;
		width: 12px;
		height: 12px;
		border: 2px solid var(--ink-soft);
		border-top-color: transparent;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
		margin-right: 4px;
		vertical-align: middle;
	}
	@keyframes spin { to { transform: rotate(360deg); } }

	/* ── Source radio options ─────────────────────────────────── */
	.source-options {
		display: flex;
		gap: 0.375rem;
		flex-wrap: wrap;
	}
	.source-option {
		display: inline-flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.375rem 0.625rem;
		font-family: var(--font-body);
		font-size: 0.8125rem;
		border: 1px solid var(--rule);
		border-radius: 9999px;
		cursor: pointer;
		color: var(--ink-soft);
		background: var(--paper);
		transition: color 160ms ease, border-color 160ms ease, background 160ms ease;
		user-select: none;
		white-space: nowrap;
	}
	.source-option:hover:not(.source-option-disabled) {
		color: var(--ink);
		border-color: var(--ink-soft);
	}
	.source-option:has(input:checked) {
		color: var(--ink);
		border-color: var(--peach-deep);
		background: color-mix(in srgb, var(--peach) 12%, var(--paper));
	}
	.source-option input[type="radio"] {
		accent-color: var(--peach-deep);
		width: 0.75rem;
		height: 0.75rem;
		flex-shrink: 0;
	}
	.source-option-disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.source-badge {
		font-family: var(--font-display);
		font-feature-settings: "smcp", "c2sc";
		letter-spacing: 0.12em;
		text-transform: lowercase;
		font-size: 0.625rem;
		font-weight: 600;
		padding: 0.0625rem 0.4rem;
		border-radius: 9999px;
		background: var(--ochre);
		color: var(--paper);
	}

	/* ── Fireflies meeting picker ──────────────────────────────── */
	.fireflies-status {
		font-family: var(--font-body);
		font-size: 0.8125rem;
		color: var(--ink-soft);
		padding: 0.25rem 0;
	}
	.fireflies-status-empty {
		font-style: italic;
	}
	.fireflies-auto-selected {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-family: var(--font-body);
		font-size: 0.8125rem;
		color: var(--ink);
		padding: 0.375rem 0.625rem;
		border: 1px solid var(--peach-deep);
		border-radius: 0.25rem;
		background: color-mix(in srgb, var(--peach) 12%, var(--paper));
		max-width: 14rem;
		overflow: hidden;
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.fireflies-time {
		font-family: var(--font-mono);
		font-size: 0.6875rem;
		color: var(--ink-faint);
		flex-shrink: 0;
	}
	.meeting-option-title {
		font-size: 0.8125rem;
	}
	.meeting-option-time {
		font-family: var(--font-mono);
		font-size: 0.6875rem;
		color: var(--ink-faint);
		margin-left: 0.375rem;
	}
</style>
