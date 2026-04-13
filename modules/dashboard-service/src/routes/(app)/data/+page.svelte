<script lang="ts">
	import { browser } from '$app/environment';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Label } from '$lib/components/ui/label';
	import { Separator } from '$lib/components/ui/separator';
	import * as Table from '$lib/components/ui/table';
	import { toastStore } from '$lib/stores/toast.svelte';

	interface TranscriptEntry {
		timestamp: string;
		speaker: string;
		text: string;
		confidence?: number;
	}

	interface TranslationEntry {
		timestamp: string;
		speaker: string;
		translated_text: string;
		language: string;
		confidence?: number;
	}

	interface TimelineEntry {
		timestamp: string;
		speaker: string;
		original: string;
		translated_text: string;
		language: string;
		confidence: number;
	}

	interface MeetingSentence {
		id: string;
		text: string;
		speaker_name: string;
		start_time: number;
		end_time: number;
		boundary_type: string;
		translations: Array<{
			translated_text: string;
			target_language: string;
			confidence: number;
			model_used: string;
		}>;
	}

	interface ApiLogEntry {
		method: string;
		endpoint: string;
		status: string;
		time: string;
	}

	interface ServiceLogEntry {
		timestamp: string;
		level: string;
		event: string;
		service: string;
		filename: string | null;
		func_name: string | null;
		lineno: number | null;
		extra: Record<string, unknown>;
	}

	let { data } = $props();

	// Data source: 'sessions' (active Fireflies) or 'meetings' (DB past meetings)
	let dataSource = $state<'sessions' | 'meetings'>('sessions');
	let selectedSession = $state('');
	let selectedMeeting = $state('');

	// Filter to only truly active sessions (not completed/disconnected/error)
	const INACTIVE_STATUSES = ['COMPLETED', 'DISCONNECTED', 'ERROR', 'completed', 'disconnected', 'error'];
	let activeSessions = $derived(
		data.sessions.filter((s: { connection_status: string }) =>
			!INACTIVE_STATUSES.includes(s.connection_status)
		)
	);

	// Sync initial selections from server load data
	$effect(() => {
		dataSource = data.meetings.length > 0 ? 'meetings' : 'sessions';
	});
	$effect(() => {
		selectedSession = data.preSelectedSession || '';
	});
	$effect(() => {
		selectedMeeting = data.preSelectedMeeting || '';
	});
	let loading = $state(false);

	// Session-based data (from /api/data/*)
	let transcripts = $state<TranscriptEntry[]>([]);
	let translations = $state<TranslationEntry[]>([]);
	let timeline = $state<TimelineEntry[]>([]);

	// Meeting-based data (from /api/meetings/*)
	let sentences = $state<MeetingSentence[]>([]);

	let apiLog = $state<ApiLogEntry[]>([]);
	let loadError = $state('');

	// Service logs (from persistent files)
	let serviceLogs = $state<ServiceLogEntry[]>([]);
	let serviceLogsLoading = $state(false);
	let logLevelFilter = $state<string>('');
	let logServiceFilter = $state<string>('');
	let logsAutoRefresh = $state(true);
	let logsRefreshInterval: ReturnType<typeof setInterval> | null = null;

	let hasData = $derived(
		transcripts.length > 0 || translations.length > 0 || timeline.length > 0 || sentences.length > 0
	);

	function formatTimestamp(iso: string): string {
		try {
			const d = new Date(iso);
			if (isNaN(d.getTime())) return iso;
			return d.toLocaleTimeString('en-US', { hour12: false });
		} catch {
			return iso;
		}
	}

	function formatSeconds(secs: number): string {
		const m = Math.floor(secs / 60);
		const s = Math.floor(secs % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}

	function formatDuration(totalSecs: number): string {
		const h = Math.floor(totalSecs / 3600);
		const m = Math.floor((totalSecs % 3600) / 60);
		if (h > 0) return `${h}h ${m}m`;
		return `${m}m`;
	}

	function nowTimestamp(): string {
		return new Date().toLocaleTimeString('en-US', { hour12: false, fractionalSecondDigits: 3 });
	}

	function truncateId(id: string): string {
		if (id.length <= 16) return id;
		return id.slice(0, 16) + '...';
	}

	async function trackedFetch(
		method: string,
		endpoint: string
	): Promise<{ ok: boolean; data: unknown }> {
		const time = nowTimestamp();
		try {
			const res = await fetch(endpoint);
			const status = res.ok ? `${res.status} OK` : `${res.status} Error`;
			apiLog = [...apiLog, { method, endpoint, status, time }];
			if (!res.ok) return { ok: false, data: null };
			const json = await res.json();
			return { ok: true, data: json };
		} catch (err) {
			const message = err instanceof TypeError && err.message === 'Failed to fetch'
				? 'Connection error. Please check your network and try again.'
				: err instanceof Error ? err.message : 'Network error';
			apiLog = [...apiLog, { method, endpoint, status: `ERR: ${message}`, time }];
			return { ok: false, data: null };
		}
	}

	async function loadSessionData() {
		if (!selectedSession || !browser) return;
		loading = true;
		loadError = '';
		sentences = [];

		const base = `/api/data/sessions/${selectedSession}`;

		const [transcriptsRes, translationsRes, timelineRes] = await Promise.all([
			trackedFetch('GET', `${base}/transcripts`),
			trackedFetch('GET', `${base}/translations`),
			trackedFetch('GET', `${base}/timeline`)
		]);

		if (transcriptsRes.ok && Array.isArray(transcriptsRes.data)) {
			transcripts = transcriptsRes.data as TranscriptEntry[];
		} else {
			transcripts = [];
		}

		if (translationsRes.ok && Array.isArray(translationsRes.data)) {
			translations = translationsRes.data as TranslationEntry[];
		} else {
			translations = [];
		}

		if (timelineRes.ok && Array.isArray(timelineRes.data)) {
			timeline = timelineRes.data as TimelineEntry[];
		} else {
			timeline = [];
		}

		if (!transcriptsRes.ok && !translationsRes.ok && !timelineRes.ok) {
			const hasConnectionError = apiLog.some((e) => e.status.includes('Connection error'));
			loadError = hasConnectionError
				? 'Connection error. Please check your network and try again.'
				: 'Failed to load any data for this session. The backend may be unavailable.';
			toastStore.error(loadError);
		} else {
			const totalRows = transcripts.length + translations.length + timeline.length;
			toastStore.success(`Loaded ${totalRows} data entries`);
		}

		loading = false;
	}

	async function loadMeetingData() {
		if (!selectedMeeting || !browser) return;
		loading = true;
		loadError = '';
		transcripts = [];
		translations = [];
		timeline = [];

		const res = await trackedFetch('GET', `/api/meetings/${selectedMeeting}/transcript`);

		if (res.ok && res.data && typeof res.data === 'object') {
			const body = res.data as { sentences?: MeetingSentence[] };
			sentences = body.sentences ?? [];
			toastStore.success(`Loaded ${sentences.length} sentences`);
		} else {
			sentences = [];
			loadError = 'Failed to load meeting transcript.';
			toastStore.error(loadError);
		}

		loading = false;
	}

	function loadData() {
		if (dataSource === 'meetings') {
			loadMeetingData();
		} else {
			loadSessionData();
		}
	}

	function clearData() {
		transcripts = [];
		translations = [];
		timeline = [];
		sentences = [];
		apiLog = [];
		loadError = '';
	}

	async function loadServiceLogs() {
		if (!browser) return;
		serviceLogsLoading = true;
		try {
			const params = new URLSearchParams({ limit: '500' });
			if (logLevelFilter) params.set('level', logLevelFilter);
			if (logServiceFilter) params.set('service', logServiceFilter);
			const res = await fetch(`/api/system/logs?${params}`);
			if (res.ok) {
				const data = await res.json();
				serviceLogs = data.entries ?? [];
			}
		} catch (e) {
			console.error('Failed to load service logs:', e);
		} finally {
			serviceLogsLoading = false;
		}
	}

	function startLogsAutoRefresh() {
		if (logsRefreshInterval) clearInterval(logsRefreshInterval);
		if (logsAutoRefresh && browser) {
			loadServiceLogs();
			logsRefreshInterval = setInterval(loadServiceLogs, 3000);
		}
	}

	function stopLogsAutoRefresh() {
		if (logsRefreshInterval) {
			clearInterval(logsRefreshInterval);
			logsRefreshInterval = null;
		}
	}

	// Auto-refresh logs when enabled
	$effect(() => {
		if (logsAutoRefresh) {
			startLogsAutoRefresh();
		} else {
			stopLogsAutoRefresh();
		}
		return () => stopLogsAutoRefresh();
	});

	// Reload when filters change
	$effect(() => {
		// Track both filters to trigger reload
		const _level = logLevelFilter;
		const _service = logServiceFilter;
		if (browser) {
			loadServiceLogs();
		}
	});

	function getLevelBadgeVariant(level: string): 'default' | 'secondary' | 'destructive' | 'outline' {
		switch (level.toLowerCase()) {
			case 'error': return 'destructive';
			case 'warning': return 'secondary';
			case 'info': return 'default';
			default: return 'outline';
		}
	}

	function formatLogTimestamp(iso: string): string {
		try {
			const d = new Date(iso);
			return d.toLocaleTimeString('en-US', { hour12: false, fractionalSecondDigits: 3 });
		} catch {
			return iso;
		}
	}
</script>

<PageHeader title="Data & Logs" description="System data, transcription logs, and analytics">
	{#snippet actions()}
		{#if hasData}
			<Button variant="outline" size="sm" onclick={clearData}>Clear</Button>
		{/if}
	{/snippet}
</PageHeader>

<!-- Controls -->
<Card.Root class="mb-6">
	<Card.Header>
		<Card.Title>Data Source</Card.Title>
		<Card.Description>Choose a data source to load transcripts, translations, and timeline data</Card.Description>
	</Card.Header>
	<Card.Content>
		<!-- Source tabs -->
		<div class="flex gap-2 mb-4">
			<Button
				variant={dataSource === 'meetings' ? 'default' : 'outline'}
				size="sm"
				onclick={() => { dataSource = 'meetings'; clearData(); }}
			>
				Meetings ({data.meetingsTotal})
			</Button>
			<Button
				variant={dataSource === 'sessions' ? 'default' : 'outline'}
				size="sm"
				onclick={() => { dataSource = 'sessions'; clearData(); }}
				disabled={activeSessions.length === 0}
			>
				Live Sessions ({activeSessions.length})
			</Button>
		</div>

		{#if dataSource === 'meetings'}
			<!-- Meeting selector -->
			<div class="flex flex-col sm:flex-row items-start sm:items-end gap-4">
				<div class="space-y-2 flex-1 w-full">
					<Label for="meeting-select">Meeting</Label>
					<select
						id="meeting-select"
						class="w-full rounded-md border bg-background px-3 py-2 text-sm"
						bind:value={selectedMeeting}
					>
						<option value="">-- Select a meeting --</option>
						{#each data.meetings as meeting (meeting.id)}
							<option value={meeting.id}>
								{meeting.title ?? 'Untitled'} — {meeting.sentence_count} sentences, {formatDuration(meeting.duration)}
								({meeting.status})
							</option>
						{/each}
					</select>
				</div>
				<Button
					onclick={loadData}
					disabled={!selectedMeeting || loading}
				>
					{#if loading}Loading...{:else}Load Transcript{/if}
				</Button>
			</div>
			{#if data.meetings.length === 0}
				<p class="text-sm text-muted-foreground mt-3">
					No past meetings found in the database. Connect to a Fireflies session to create meeting data.
				</p>
			{/if}
		{:else}
			<!-- Live session selector -->
			{#if activeSessions.length > 0}
				<div class="flex flex-col sm:flex-row items-start sm:items-end gap-4">
					<div class="space-y-2 flex-1 w-full">
						<Label for="session-select">Live Session</Label>
						<select
							id="session-select"
							class="w-full rounded-md border bg-background px-3 py-2 text-sm"
							bind:value={selectedSession}
						>
							<option value="">-- Select a live session --</option>
							{#each activeSessions as session (session.session_id)}
								<option value={session.session_id}>
									{truncateId(session.session_id)} ({session.connection_status}) - {session.chunks_received} chunks
								</option>
							{/each}
						</select>
					</div>
					<Button
						onclick={loadData}
						disabled={!selectedSession || loading}
					>
						{#if loading}Loading...{:else}Load Data{/if}
					</Button>
				</div>
			{:else}
				<p class="text-sm text-muted-foreground mt-3">
					No live sessions. Start a Fireflies connection or loopback capture to see live data here.
				</p>
			{/if}
		{/if}

		{#if loadError}
			<p class="text-sm text-destructive mt-3">{loadError}</p>
		{/if}
	</Card.Content>
</Card.Root>

<!-- Meeting Sentences view -->
{#if sentences.length > 0}
	<Card.Root class="mb-6">
		<Card.Header>
			<div class="flex items-center justify-between">
				<Card.Title>Meeting Transcript</Card.Title>
				<Badge variant="secondary">{sentences.length} sentences</Badge>
			</div>
		</Card.Header>
		<Card.Content>
			<div class="max-h-[40rem] overflow-auto">
				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head class="w-20">Time</Table.Head>
							<Table.Head class="w-32">Speaker</Table.Head>
							<Table.Head class="min-w-64">Text</Table.Head>
							<Table.Head class="min-w-48">Translation</Table.Head>
						</Table.Row>
					</Table.Header>
					<Table.Body>
						{#each sentences as s, i (s.id)}
							<Table.Row>
								<Table.Cell class="font-mono text-xs whitespace-nowrap text-muted-foreground">
									{formatSeconds(s.start_time)}
								</Table.Cell>
								<Table.Cell class="font-medium text-sm">
									{s.speaker_name || 'Unknown'}
								</Table.Cell>
								<Table.Cell class="text-sm">
									{s.text}
								</Table.Cell>
								<Table.Cell class="text-sm">
									{#if s.translations && s.translations.length > 0}
										{#each s.translations as t}
											<div>
												<Badge variant="outline" class="text-xs mr-1">{t.target_language}</Badge>
												{t.translated_text}
											</div>
										{/each}
									{:else}
										<span class="text-muted-foreground">--</span>
									{/if}
								</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
			</div>
		</Card.Content>
	</Card.Root>
{/if}

<!-- Session data views (transcripts + translations + timeline) -->
{#if transcripts.length > 0 || translations.length > 0}
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
		<!-- Transcripts -->
		<Card.Root>
			<Card.Header>
				<div class="flex items-center justify-between">
					<Card.Title>Transcripts</Card.Title>
					<Badge variant="secondary">{transcripts.length}</Badge>
				</div>
			</Card.Header>
			<Card.Content>
				{#if transcripts.length === 0}
					<p class="text-sm text-muted-foreground text-center py-6">No transcripts found</p>
				{:else}
					<div class="max-h-96 overflow-y-auto space-y-1 font-mono text-sm">
						{#each transcripts as entry, i (i)}
							<div class="py-1 px-2 rounded hover:bg-accent/50">
								<span class="text-muted-foreground">[{formatTimestamp(entry.timestamp)}]</span>
								<span class="font-semibold text-primary">{entry.speaker}:</span>
								<span>{entry.text}</span>
							</div>
						{/each}
					</div>
				{/if}
			</Card.Content>
		</Card.Root>

		<!-- Translations -->
		<Card.Root>
			<Card.Header>
				<div class="flex items-center justify-between">
					<Card.Title>Translations</Card.Title>
					<Badge variant="secondary">{translations.length}</Badge>
				</div>
			</Card.Header>
			<Card.Content>
				{#if translations.length === 0}
					<p class="text-sm text-muted-foreground text-center py-6">No translations found</p>
				{:else}
					<div class="max-h-96 overflow-y-auto space-y-1 font-mono text-sm">
						{#each translations as entry, i (i)}
							<div class="py-1 px-2 rounded hover:bg-accent/50">
								<Badge variant="outline" class="mr-1 font-mono text-xs">{entry.language}</Badge>
								<span class="font-semibold text-primary">{entry.speaker}:</span>
								<span>{entry.translated_text}</span>
							</div>
						{/each}
					</div>
				{/if}
			</Card.Content>
		</Card.Root>
	</div>

	<Separator class="mb-6" />
{/if}

{#if timeline.length > 0}
	<!-- Database Entries Table -->
	<Card.Root class="mb-6">
		<Card.Header>
			<div class="flex items-center justify-between">
				<Card.Title>Database Entries</Card.Title>
				<Badge variant="secondary">{timeline.length} rows</Badge>
			</div>
		</Card.Header>
		<Card.Content>
			<div class="max-h-[32rem] overflow-auto overflow-x-auto">
				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head>Time</Table.Head>
							<Table.Head>Speaker</Table.Head>
							<Table.Head class="min-w-48">Original</Table.Head>
							<Table.Head class="min-w-48">Translation</Table.Head>
							<Table.Head>Language</Table.Head>
							<Table.Head>Confidence</Table.Head>
						</Table.Row>
					</Table.Header>
					<Table.Body>
						{#each timeline as row, i (i)}
							<Table.Row>
								<Table.Cell class="font-mono text-xs whitespace-nowrap">
									{formatTimestamp(row.timestamp)}
								</Table.Cell>
								<Table.Cell class="font-medium">
									{row.speaker}
								</Table.Cell>
								<Table.Cell class="text-sm">
									{row.original}
								</Table.Cell>
								<Table.Cell class="text-sm">
									{row.translated_text}
								</Table.Cell>
								<Table.Cell>
									<Badge variant="outline" class="text-xs">{row.language}</Badge>
								</Table.Cell>
								<Table.Cell>
									{#if row.confidence != null}
										<span
											class="text-sm font-medium"
											class:text-green-600={row.confidence >= 0.8}
											class:text-yellow-600={row.confidence >= 0.5 && row.confidence < 0.8}
											class:text-red-600={row.confidence < 0.5}
										>
											{Math.round(row.confidence * 100)}%
										</span>
									{:else}
										<span class="text-muted-foreground">--</span>
									{/if}
								</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
			</div>
		</Card.Content>
	</Card.Root>
{/if}

<!-- API Call Log -->
{#if apiLog.length > 0}
	<Card.Root class="mb-6">
		<Card.Header>
			<div class="flex items-center justify-between">
				<Card.Title>API Call Log</Card.Title>
				<Badge variant="secondary">{apiLog.length} calls</Badge>
			</div>
		</Card.Header>
		<Card.Content>
			<div class="max-h-64 overflow-y-auto space-y-1 font-mono text-xs">
				{#each apiLog as entry, i (i)}
					<div class="flex items-center gap-3 py-1 px-2 rounded hover:bg-accent/50">
						<span class="text-muted-foreground whitespace-nowrap">{entry.time}</span>
						<Badge variant="outline" class="text-xs shrink-0">{entry.method}</Badge>
						<span class="truncate flex-1" title={entry.endpoint}>{entry.endpoint}</span>
						{#if entry.status.startsWith('2')}
							<Badge variant="default" class="text-xs shrink-0">{entry.status}</Badge>
						{:else if entry.status.startsWith('ERR')}
							<Badge variant="destructive" class="text-xs shrink-0">{entry.status}</Badge>
						{:else}
							<Badge variant="secondary" class="text-xs shrink-0">{entry.status}</Badge>
						{/if}
					</div>
				{/each}
			</div>
		</Card.Content>
	</Card.Root>
{/if}

<!-- Service Logs (Persistent) -->
<Card.Root class="mb-6">
	<Card.Header>
		<div class="flex items-center justify-between">
			<div>
				<Card.Title>Service Logs</Card.Title>
				<Card.Description>Persistent logs from /tmp/livetranslate/logs/</Card.Description>
			</div>
			<div class="flex items-center gap-2">
				<select
					class="rounded-md border bg-background px-2 py-1 text-xs"
					bind:value={logServiceFilter}
				>
					<option value="">All services</option>
					<option value="orchestration">orchestration</option>
					<option value="transcription">transcription</option>
					<option value="dashboard">dashboard</option>
				</select>
				<select
					class="rounded-md border bg-background px-2 py-1 text-xs"
					bind:value={logLevelFilter}
				>
					<option value="">All levels</option>
					<option value="debug">Debug</option>
					<option value="info">Info</option>
					<option value="warning">Warning</option>
					<option value="error">Error</option>
				</select>
				<Button
					variant={logsAutoRefresh ? 'default' : 'outline'}
					size="sm"
					onclick={() => logsAutoRefresh = !logsAutoRefresh}
				>
					{logsAutoRefresh ? 'Live' : 'Paused'}
				</Button>
				<Badge variant="secondary">{serviceLogs.length} entries</Badge>
			</div>
		</div>
	</Card.Header>
	<Card.Content>
		{#if serviceLogs.length === 0}
			<p class="text-sm text-muted-foreground text-center py-6">
				{serviceLogsLoading ? 'Loading logs...' : 'No logs yet. Logs will appear as services emit events.'}
			</p>
		{:else}
			<div class="max-h-96 overflow-y-auto space-y-1 font-mono text-xs">
				{#each serviceLogs as log, i (i)}
					<div class="flex items-start gap-2 py-1.5 px-2 rounded hover:bg-accent/50 border-l-2 {log.level === 'error' ? 'border-l-red-500 bg-red-500/5' : log.level === 'warning' ? 'border-l-yellow-500 bg-yellow-500/5' : 'border-l-transparent'}">
						<span class="text-muted-foreground whitespace-nowrap shrink-0">{formatLogTimestamp(log.timestamp)}</span>
						<Badge variant={getLevelBadgeVariant(log.level)} class="text-xs shrink-0 w-14 justify-center">{log.level}</Badge>
						<Badge variant="outline" class="text-xs shrink-0">{log.service}</Badge>
						<span class="flex-1 break-all">
							<span class="font-semibold text-primary">{log.event}</span>
							{#if Object.keys(log.extra).length > 0}
								<span class="text-muted-foreground ml-1">
									{#each Object.entries(log.extra).slice(0, 4) as [k, v]}
										<span class="ml-1">{k}=<span class="text-foreground">{typeof v === 'string' ? v : JSON.stringify(v)}</span></span>
									{/each}
									{#if Object.keys(log.extra).length > 4}
										<span class="ml-1">+{Object.keys(log.extra).length - 4} more</span>
									{/if}
								</span>
							{/if}
						</span>
						{#if log.filename}
							<span class="text-muted-foreground shrink-0 text-[10px]">{log.filename}:{log.lineno}</span>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	</Card.Content>
</Card.Root>

{#if !hasData && apiLog.length === 0}
	<Card.Root>
		<Card.Content class="py-12">
			<div class="text-center">
				<p class="text-muted-foreground mb-2">Select a data source and click "Load" to view transcripts and translations.</p>
				<p class="text-sm text-muted-foreground">
					<strong>Past Meetings</strong> shows completed sessions stored in the database.
					<strong>Active Sessions</strong> shows currently running Fireflies connections.
				</p>
			</div>
		</Card.Content>
	</Card.Root>
{/if}
