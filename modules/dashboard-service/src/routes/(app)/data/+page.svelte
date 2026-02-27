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

	interface ApiLogEntry {
		method: string;
		endpoint: string;
		status: string;
		time: string;
	}

	let { data } = $props();

	const initialSession = data.preSelectedSession || '';
	let selectedSession = $state(initialSession);
	let loading = $state(false);
	let transcripts = $state<TranscriptEntry[]>([]);
	let translations = $state<TranslationEntry[]>([]);
	let timeline = $state<TimelineEntry[]>([]);
	let apiLog = $state<ApiLogEntry[]>([]);
	let loadError = $state('');

	let hasData = $derived(transcripts.length > 0 || translations.length > 0 || timeline.length > 0);

	function formatTimestamp(iso: string): string {
		try {
			const d = new Date(iso);
			if (isNaN(d.getTime())) return iso;
			return d.toLocaleTimeString('en-US', { hour12: false });
		} catch {
			return iso;
		}
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

	async function loadData() {
		if (!selectedSession || !browser) return;
		loading = true;
		loadError = '';

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

	function clearData() {
		transcripts = [];
		translations = [];
		timeline = [];
		apiLog = [];
		loadError = '';
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
		<Card.Title>Session Selector</Card.Title>
		<Card.Description>Choose a session to load its transcripts, translations, and timeline data</Card.Description>
	</Card.Header>
	<Card.Content>
		<div class="flex flex-col sm:flex-row items-start sm:items-end gap-4">
			<div class="space-y-2 flex-1 w-full">
				<Label for="session-select">Session</Label>
				<select
					id="session-select"
					class="w-full rounded-md border bg-background px-3 py-2 text-sm"
					bind:value={selectedSession}
				>
					<option value="">-- Select a session --</option>
					{#each data.sessions as session (session.session_id)}
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
				{#if loading}
					Loading...
				{:else}
					Load Data
				{/if}
			</Button>
		</div>
		{#if data.sessions.length === 0}
			<p class="text-sm text-muted-foreground mt-3">
				No sessions available. Connect to Fireflies first to create sessions.
			</p>
		{/if}
		{#if loadError}
			<p class="text-sm text-destructive mt-3">{loadError}</p>
		{/if}
	</Card.Content>
</Card.Root>

{#if hasData}
	<!-- Two-column grid: Transcripts & Translations -->
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

	<!-- Database Entries Table -->
	<Card.Root class="mb-6">
		<Card.Header>
			<div class="flex items-center justify-between">
				<Card.Title>Database Entries</Card.Title>
				<Badge variant="secondary">{timeline.length} rows</Badge>
			</div>
		</Card.Header>
		<Card.Content>
			{#if timeline.length === 0}
				<p class="text-sm text-muted-foreground text-center py-6">No timeline entries found</p>
			{:else}
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
			{/if}
		</Card.Content>
	</Card.Root>
{/if}

<!-- API Call Log -->
{#if apiLog.length > 0}
	<Card.Root>
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

{#if !hasData && apiLog.length === 0}
	<Card.Root>
		<Card.Content class="py-12">
			<div class="text-center">
				<p class="text-muted-foreground mb-2">Select a session and click "Load Data" to view transcripts, translations, and timeline entries.</p>
				<p class="text-sm text-muted-foreground">
					Data is fetched from the orchestration service via SvelteKit API proxies.
				</p>
			</div>
		</Card.Content>
	</Card.Root>
{/if}
