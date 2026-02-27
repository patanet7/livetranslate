<script lang="ts">
	import { browser } from '$app/environment';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import * as Table from '$lib/components/ui/table';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Separator } from '$lib/components/ui/separator';
	import { toastStore } from '$lib/stores/toast.svelte';

	interface Meeting {
		id: string;
		title: string;
		date: string;
		duration: number;
		speakers: string[];
	}

	interface TranscriptEntry {
		speaker: string;
		text: string;
		start_time: number;
		end_time: number;
		translation?: string;
	}

	interface SavedTranscript {
		id: string;
		language: string;
		saved_at: string;
		entries: TranscriptEntry[];
	}

	let { data } = $props();

	// --- Section 1: Fetch Past Meetings ---
	let dateFrom = $state('');
	let dateTo = $state('');
	let meetings = $state<Meeting[]>([]);
	let fetchingMeetings = $state(false);
	let fetchError = $state('');

	async function fetchPastMeetings() {
		if (!browser) return;
		const apiKey = localStorage.getItem('fireflies_api_key');
		if (!apiKey) {
			fetchError = 'No Fireflies API key found. Set it on the Fireflies page first.';
			toastStore.error(fetchError);
			return;
		}
		fetchError = '';
		fetchingMeetings = true;
		try {
			const res = await fetch('/api/fireflies/transcripts', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					api_key: apiKey,
					date_from: dateFrom || undefined,
					date_to: dateTo || undefined
				})
			});
			const result = await res.json();
			if (!res.ok) {
				fetchError = result.error ?? `Request failed (${res.status})`;
				toastStore.error(fetchError);
				return;
			}
			meetings = Array.isArray(result) ? result : (result.transcripts ?? []);
			toastStore.success(`Fetched ${meetings.length} meeting${meetings.length === 1 ? '' : 's'}`);
		} catch (err) {
			if (err instanceof TypeError && err.message === 'Failed to fetch') {
				fetchError = 'Connection error. Please check your network and try again.';
			} else {
				fetchError = err instanceof Error ? err.message : 'Unknown error';
			}
			toastStore.error(fetchError);
		} finally {
			fetchingMeetings = false;
		}
	}

	function formatDuration(seconds: number): string {
		if (!seconds) return '--';
		const m = Math.floor(seconds / 60);
		const s = seconds % 60;
		return `${m}m ${s}s`;
	}

	// --- Section 2: Transcript Viewer Modal ---
	let viewerOpen = $state(false);
	let viewerMeeting = $state<Meeting | null>(null);
	let transcriptEntries = $state<TranscriptEntry[]>([]);
	let loadingTranscript = $state(false);
	let transcriptError = $state('');
	let targetLanguage = $state('es');
	let translating = $state(false);
	let translationDone = $state(0);
	let translationTotal = $state(0);
	let translationPercent = $derived(
		translationTotal > 0 ? Math.round((translationDone / translationTotal) * 100) : 0
	);

	async function openViewer(meeting: Meeting) {
		viewerMeeting = meeting;
		viewerOpen = true;
		transcriptEntries = [];
		transcriptError = '';
		loadingTranscript = true;
		translationDone = 0;
		translationTotal = 0;

		if (!browser) return;
		const apiKey = localStorage.getItem('fireflies_api_key');
		if (!apiKey) {
			transcriptError = 'No API key found';
			loadingTranscript = false;
			return;
		}

		try {
			const res = await fetch(`/api/fireflies/transcript/${meeting.id}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ api_key: apiKey })
			});
			const result = await res.json();
			if (!res.ok) {
				transcriptError = result.error ?? `Request failed (${res.status})`;
				return;
			}
			transcriptEntries = Array.isArray(result)
				? result
				: (result.sentences ?? result.entries ?? []);
		} catch (err) {
			if (err instanceof TypeError && err.message === 'Failed to fetch') {
				transcriptError = 'Connection error. Please check your network and try again.';
			} else {
				transcriptError = err instanceof Error ? err.message : 'Unknown error';
			}
		} finally {
			loadingTranscript = false;
		}
	}

	async function translateAll() {
		if (!transcriptEntries.length) return;
		translating = true;
		translationDone = 0;
		translationTotal = transcriptEntries.length;

		const batchSize = 5;
		for (let i = 0; i < transcriptEntries.length; i += batchSize) {
			const batch = transcriptEntries.slice(i, i + batchSize);
			const promises = batch.map(async (entry, idx) => {
				try {
					const res = await fetch('/api/translation/translate', {
						method: 'POST',
						headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify({ text: entry.text, target_language: targetLanguage })
					});
					const result = await res.json();
					if (res.ok && result.translated_text) {
						transcriptEntries[i + idx] = {
							...transcriptEntries[i + idx],
							translation: result.translated_text
						};
					}
				} catch {
					// translation failed for this entry, leave it empty
				}
				translationDone++;
			});
			await Promise.all(promises);
		}
		translating = false;
		toastStore.success(`Translation complete: ${translationDone}/${translationTotal} entries`);
	}

	function formatTimestamp(seconds: number): string {
		if (seconds == null) return '--:--';
		const m = Math.floor(seconds / 60);
		const s = Math.floor(seconds % 60);
		return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
	}

	function saveLocally() {
		if (!browser || !viewerMeeting) return;
		const saved: SavedTranscript = {
			id: viewerMeeting.id,
			language: targetLanguage,
			saved_at: new Date().toISOString(),
			entries: transcriptEntries
		};
		localStorage.setItem(`saved_transcript_${viewerMeeting.id}`, JSON.stringify(saved));
		refreshSavedTranscripts();
		toastStore.success('Transcript saved locally');
	}

	async function importToDb() {
		if (!browser || !viewerMeeting) return;
		const apiKey = localStorage.getItem('fireflies_api_key');
		if (!apiKey) {
			toastStore.error('No API key found');
			return;
		}
		try {
			const res = await fetch(`/api/fireflies/import/${viewerMeeting.id}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ api_key: apiKey })
			});
			if (res.ok) {
				toastStore.success('Transcript imported to database');
			} else {
				toastStore.error('Failed to import transcript');
			}
		} catch {
			toastStore.error('Network error importing transcript');
		}
	}

	// --- Section 3: Saved Transcripts ---
	let savedTranscripts = $state<SavedTranscript[]>([]);
	let savedViewerOpen = $state(false);
	let savedViewerTranscript = $state<SavedTranscript | null>(null);
	let deleteConfirmId = $state<string | null>(null);

	function refreshSavedTranscripts() {
		if (!browser) return;
		const items: SavedTranscript[] = [];
		for (let i = 0; i < localStorage.length; i++) {
			const key = localStorage.key(i);
			if (key?.startsWith('saved_transcript_')) {
				try {
					const raw = localStorage.getItem(key);
					if (raw) items.push(JSON.parse(raw) as SavedTranscript);
				} catch {
					// skip invalid entries
				}
			}
		}
		items.sort((a, b) => new Date(b.saved_at).getTime() - new Date(a.saved_at).getTime());
		savedTranscripts = items;
	}

	$effect(() => {
		if (browser) {
			refreshSavedTranscripts();
		}
	});

	function viewSaved(transcript: SavedTranscript) {
		savedViewerTranscript = transcript;
		savedViewerOpen = true;
	}

	function exportJson(transcript: SavedTranscript) {
		if (!browser) return;
		const blob = new Blob([JSON.stringify(transcript, null, 2)], {
			type: 'application/json'
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `transcript_${transcript.id}.json`;
		a.click();
		URL.revokeObjectURL(url);
	}

	function deleteSaved(id: string) {
		if (!browser) return;
		localStorage.removeItem(`saved_transcript_${id}`);
		deleteConfirmId = null;
		refreshSavedTranscripts();
		toastStore.success('Saved transcript deleted');
	}
</script>

<PageHeader title="Session History" description="Browse past Fireflies meetings, view transcripts, and translate" />

<!-- Section 1: Fetch Past Meetings -->
<Card.Root class="mb-6">
	<Card.Header>
		<Card.Title>Fetch Past Meetings</Card.Title>
	</Card.Header>
	<Card.Content>
		<div class="flex flex-wrap items-end gap-4">
			<div class="space-y-2 w-full sm:w-auto">
				<Label for="date-from">From</Label>
				<Input id="date-from" type="date" bind:value={dateFrom} class="w-full sm:w-44" />
			</div>
			<div class="space-y-2 w-full sm:w-auto">
				<Label for="date-to">To</Label>
				<Input id="date-to" type="date" bind:value={dateTo} class="w-full sm:w-44" />
			</div>
			<Button class="w-full sm:w-auto" onclick={fetchPastMeetings} disabled={fetchingMeetings}>
				{fetchingMeetings ? 'Fetching...' : 'Fetch Past Meetings'}
			</Button>
		</div>
		{#if fetchError}
			<p class="text-sm text-destructive mt-3">{fetchError}</p>
		{/if}
	</Card.Content>
</Card.Root>

{#if meetings.length > 0}
	<Card.Root class="mb-6">
		<Card.Header>
			<Card.Title>Past Meetings ({meetings.length})</Card.Title>
		</Card.Header>
		<Card.Content class="p-0">
			<div class="overflow-x-auto">
				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head>Date</Table.Head>
							<Table.Head>Title</Table.Head>
							<Table.Head>Duration</Table.Head>
							<Table.Head>Speakers</Table.Head>
							<Table.Head>Actions</Table.Head>
						</Table.Row>
					</Table.Header>
					<Table.Body>
						{#each meetings as meeting (meeting.id)}
							<Table.Row>
								<Table.Cell class="text-xs text-muted-foreground whitespace-nowrap">
									{new Date(meeting.date).toLocaleDateString()}
								</Table.Cell>
								<Table.Cell class="font-medium">{meeting.title}</Table.Cell>
								<Table.Cell>{formatDuration(meeting.duration)}</Table.Cell>
								<Table.Cell>
									{#if meeting.speakers?.length}
										<div class="flex flex-wrap gap-1">
											{#each meeting.speakers as speaker}
												<Badge variant="secondary">{speaker}</Badge>
											{/each}
										</div>
									{:else}
										<span class="text-muted-foreground">--</span>
									{/if}
								</Table.Cell>
								<Table.Cell>
									<div class="flex gap-2">
										<Button variant="outline" size="sm" onclick={() => openViewer(meeting)}>
											View
										</Button>
										<Button variant="outline" size="sm" onclick={() => openViewer(meeting)}>
											Translate
										</Button>
									</div>
								</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
			</div>
		</Card.Content>
	</Card.Root>
{:else if !fetchingMeetings && !fetchError}
	<Card.Root class="mb-6">
		<Card.Content class="py-12">
			<div class="text-center">
				<p class="text-muted-foreground">Enter a date range and click "Fetch Past Meetings" to browse your meeting history.</p>
			</div>
		</Card.Content>
	</Card.Root>
{/if}

<!-- Section 2: Transcript Viewer Modal -->
<Dialog.Root bind:open={viewerOpen}>
	<Dialog.Content class="max-w-3xl max-h-[85vh] overflow-y-auto">
		<Dialog.Header>
			<Dialog.Title>{viewerMeeting?.title ?? 'Transcript'}</Dialog.Title>
			<Dialog.Description>
				View and translate transcript content
			</Dialog.Description>
		</Dialog.Header>

		{#if loadingTranscript}
			<div class="py-8 text-center text-muted-foreground">Loading transcript...</div>
		{:else if transcriptError}
			<div class="py-4 text-center text-destructive">{transcriptError}</div>
		{:else}
			<!-- Transcript entries -->
			<div class="max-h-[40vh] overflow-y-auto border rounded-md p-3 space-y-3 mb-4">
				{#if transcriptEntries.length === 0}
					<p class="text-muted-foreground text-center py-4">No transcript entries</p>
				{:else}
					{#each transcriptEntries as entry, i (i)}
						<div class="text-sm space-y-1">
							<div class="flex items-center gap-2">
								<Badge variant="outline">{entry.speaker ?? 'Unknown'}</Badge>
								<span class="text-xs text-muted-foreground">
									{formatTimestamp(entry.start_time)} - {formatTimestamp(entry.end_time)}
								</span>
							</div>
							<p>{entry.text}</p>
							{#if entry.translation}
								<p class="text-primary italic">{entry.translation}</p>
							{/if}
						</div>
						{#if i < transcriptEntries.length - 1}
							<Separator />
						{/if}
					{/each}
				{/if}
			</div>

			<!-- Translation controls -->
			<div class="space-y-3">
				<div class="flex items-end gap-4">
					<div class="space-y-2 flex-1">
						<Label for="target-lang">Target Language</Label>
						<select
							id="target-lang"
							bind:value={targetLanguage}
							class="w-full rounded-md border bg-background px-3 py-2 text-sm"
						>
							{#if data.uiConfig?.languages}
								{#each data.uiConfig.languages as lang}
									<option value={lang.code}>{lang.name}</option>
								{/each}
							{:else}
								<option value="es">Spanish</option>
								<option value="fr">French</option>
								<option value="de">German</option>
							{/if}
						</select>
					</div>
					<Button
						onclick={translateAll}
						disabled={translating || transcriptEntries.length === 0}
					>
						{translating ? 'Translating...' : 'Translate All'}
					</Button>
				</div>

				{#if translating || translationDone > 0}
					<div class="space-y-1">
						<div class="flex justify-between text-xs text-muted-foreground">
							<span>{translationDone}/{translationTotal} ({translationPercent}%)</span>
						</div>
						<div class="h-2 bg-secondary rounded-full overflow-hidden">
							<div
								class="h-full bg-primary transition-all duration-300"
								style="width: {translationPercent}%"
							></div>
						</div>
					</div>
				{/if}
			</div>
		{/if}

		<Dialog.Footer class="mt-4">
			<Button variant="outline" onclick={saveLocally} disabled={transcriptEntries.length === 0}>
				Save Locally
			</Button>
			<Button variant="outline" onclick={importToDb} disabled={transcriptEntries.length === 0}>
				Import to DB
			</Button>
			<Dialog.Close>
				<Button variant="secondary">Close</Button>
			</Dialog.Close>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Section 3: Saved Transcripts (Local) -->
<Card.Root>
	<Card.Header>
		<Card.Title>Saved Transcripts (Local)</Card.Title>
	</Card.Header>
	<Card.Content class="p-0">
		{#if savedTranscripts.length === 0}
			<div class="p-6 text-center text-muted-foreground">No saved transcripts yet</div>
		{:else}
			<div class="overflow-x-auto">
			<Table.Root>
				<Table.Header>
					<Table.Row>
						<Table.Head>ID</Table.Head>
						<Table.Head>Language</Table.Head>
						<Table.Head>Saved At</Table.Head>
						<Table.Head>Items</Table.Head>
						<Table.Head>Actions</Table.Head>
					</Table.Row>
				</Table.Header>
				<Table.Body>
					{#each savedTranscripts as transcript (transcript.id)}
						<Table.Row>
							<Table.Cell class="font-mono text-xs">
								{transcript.id.length > 20 ? transcript.id.slice(0, 20) + '...' : transcript.id}
							</Table.Cell>
							<Table.Cell>
								<Badge variant="secondary">{transcript.language}</Badge>
							</Table.Cell>
							<Table.Cell class="text-xs text-muted-foreground whitespace-nowrap">
								{new Date(transcript.saved_at).toLocaleString()}
							</Table.Cell>
							<Table.Cell>{transcript.entries.length}</Table.Cell>
							<Table.Cell>
								<div class="flex gap-2">
									<Button variant="outline" size="sm" onclick={() => viewSaved(transcript)}>
										View
									</Button>
									<Button variant="outline" size="sm" onclick={() => exportJson(transcript)}>
										Export JSON
									</Button>
									{#if deleteConfirmId === transcript.id}
										<Button
											variant="destructive"
											size="sm"
											onclick={() => deleteSaved(transcript.id)}
										>
											Confirm
										</Button>
										<Button
											variant="outline"
											size="sm"
											onclick={() => (deleteConfirmId = null)}
										>
											Cancel
										</Button>
									{:else}
										<Button
											variant="outline"
											size="sm"
											onclick={() => (deleteConfirmId = transcript.id)}
										>
											Delete
										</Button>
									{/if}
								</div>
							</Table.Cell>
						</Table.Row>
					{/each}
				</Table.Body>
			</Table.Root>
			</div>
		{/if}
	</Card.Content>
</Card.Root>

<!-- Saved Transcript Viewer Modal (side-by-side original/translated) -->
<Dialog.Root bind:open={savedViewerOpen}>
	<Dialog.Content class="max-w-4xl max-h-[85vh] overflow-y-auto">
		<Dialog.Header>
			<Dialog.Title>Saved Transcript: {savedViewerTranscript?.id ?? ''}</Dialog.Title>
			<Dialog.Description>
				Side-by-side view of original and translated content
			</Dialog.Description>
		</Dialog.Header>

		{#if savedViewerTranscript}
			<div class="flex gap-2 mb-3 text-xs text-muted-foreground">
				<span>Language: <Badge variant="secondary">{savedViewerTranscript.language}</Badge></span>
				<span>Saved: {new Date(savedViewerTranscript.saved_at).toLocaleString()}</span>
				<span>Items: {savedViewerTranscript.entries.length}</span>
			</div>

			<div class="max-h-[55vh] overflow-y-auto border rounded-md">
				<table class="w-full text-sm">
					<thead class="sticky top-0 bg-background border-b">
						<tr>
							<th class="text-left p-2 w-24">Speaker</th>
							<th class="text-left p-2">Original</th>
							<th class="text-left p-2">Translation</th>
						</tr>
					</thead>
					<tbody>
						{#each savedViewerTranscript.entries as entry, i (i)}
							<tr class="border-b last:border-b-0">
								<td class="p-2 align-top">
									<Badge variant="outline">{entry.speaker ?? 'Unknown'}</Badge>
								</td>
								<td class="p-2 align-top">{entry.text}</td>
								<td class="p-2 align-top text-primary italic">
									{entry.translation ?? '--'}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}

		<Dialog.Footer class="mt-4">
			<Dialog.Close>
				<Button variant="secondary">Close</Button>
			</Dialog.Close>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
