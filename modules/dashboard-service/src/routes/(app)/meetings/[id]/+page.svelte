<script lang="ts">
	import { browser } from '$app/environment';
	import { marked } from 'marked';
	import * as Tabs from '$lib/components/ui/tabs';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { Input } from '$lib/components/ui/input';
	import { Separator } from '$lib/components/ui/separator';
	import ErrorBoundary from '$lib/components/ErrorBoundary.svelte';
	import { toastStore } from '$lib/stores/toast.svelte';
	import type { MeetingSpeaker, MeetingSentence, MeetingInsight } from '$lib/types';
	import type { DiarizationJob, TranscriptComparison } from '$lib/api/diarization';
	import { exportApi } from '$lib/api/export';

	let { data } = $props();

	// --- Export dropdown state ---
	let exportOpen = $state(false);

	const meeting = $derived(data.meeting);
	const sentences: MeetingSentence[] = $derived(data.transcript?.sentences ?? []);
	const insights: MeetingInsight[] = $derived(data.insights?.insights ?? []);

	// --- Transcript tab state ---
	let transcriptSearch = $state('');
	let speakerFilter = $state<string | null>(null);

	const speakers = $derived(
		[...new Set(sentences.map((s) => s.speaker_name).filter(Boolean))] as string[]
	);

	const speakerColors: Record<string, string> = {};
	const palette = [
		'#4CAF50',
		'#2196F3',
		'#FF9800',
		'#9C27B0',
		'#F44336',
		'#00BCD4',
		'#795548',
		'#607D8B'
	];

	function getSpeakerColor(name: string): string {
		if (!speakerColors[name]) {
			speakerColors[name] = palette[Object.keys(speakerColors).length % palette.length];
		}
		return speakerColors[name];
	}

	const filteredSentences = $derived(
		sentences.filter((s) => {
			if (speakerFilter && s.speaker_name !== speakerFilter) return false;
			if (transcriptSearch) {
				const q = transcriptSearch.toLowerCase();
				return (
					s.text.toLowerCase().includes(q) ||
					s.translations?.some((t) => t.translated_text.toLowerCase().includes(q))
				);
			}
			return true;
		})
	);

	function formatTimestamp(seconds: number): string {
		if (seconds == null) return '--:--';
		const m = Math.floor(seconds / 60);
		const s = Math.floor(seconds % 60);
		return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
	}

	// --- Insights tab ---
	function getInsightsByType(type: string): MeetingInsight[] {
		return insights.filter((i) => i.insight_type === type);
	}

	/** Parse insight content — handles JSONB strings from PostgreSQL. */
	function parseContent(raw: unknown): unknown {
		if (typeof raw === 'string') {
			try {
				return JSON.parse(raw);
			} catch {
				return raw; // plain text, not JSON
			}
		}
		return raw;
	}

	function getInsightText(insight: MeetingInsight): string {
		const content = parseContent(insight.content);
		if (typeof content === 'string') return content;
		if (typeof content === 'object' && content && 'text' in content) {
			const text = (content as Record<string, unknown>).text;
			if (typeof text === 'string') return text;
		}
		return JSON.stringify(content, null, 2);
	}

	function getInsightItems(insight: MeetingInsight): string[] {
		const content: unknown = parseContent(insight.content);
		if (Array.isArray(content)) {
			return content.map((item: unknown) => {
				if (typeof item === 'string') return item;
				if (typeof item === 'object' && item && 'text' in item) return String((item as Record<string, unknown>).text);
				return JSON.stringify(item);
			});
		}
		if (typeof content === 'string') return content.split('\n').filter(Boolean);
		if (typeof content === 'object' && content && 'text' in content) {
			const text = (content as Record<string, unknown>).text;
			if (typeof text === 'string') return text.split('\n').filter(Boolean);
		}
		return [];
	}

	function getKeywords(insight: MeetingInsight): string[] {
		const content: unknown = parseContent(insight.content);
		if (Array.isArray(content)) return content.map(String);
		if (typeof content === 'string') return content.split(/[,\n]/).map((s: string) => s.trim()).filter(Boolean);
		if (typeof content === 'object' && content && 'text' in content) {
			const text = (content as Record<string, unknown>).text;
			if (typeof text === 'string') return text.split(/[,\n]/).map((s: string) => s.trim()).filter(Boolean);
		}
		return [];
	}

	function getSentiment(insight: MeetingInsight): { positive: number; neutral: number; negative: number } | null {
		const content = parseContent(insight.content);
		if (typeof content === 'object' && content && !Array.isArray(content)) {
			const c = content as Record<string, unknown>;
			// Fireflies uses _pct suffix (positive_pct, neutral_pct, negative_pct)
			if ('positive_pct' in c || 'neutral_pct' in c || 'negative_pct' in c) {
				return {
					positive: Number(c.positive_pct ?? 0),
					neutral: Number(c.neutral_pct ?? 0),
					negative: Number(c.negative_pct ?? 0)
				};
			}
			if ('positive' in c || 'neutral' in c || 'negative' in c) {
				return {
					positive: Number(c.positive ?? 0),
					neutral: Number(c.neutral ?? 0),
					negative: Number(c.negative ?? 0)
				};
			}
			if ('text' in c && typeof c.text === 'string') {
				// Try to parse sentiment from text
				const match = c.text.match(/positive[:\s]+(\d+)/i);
				if (match) return { positive: Number(match[1]), neutral: 0, negative: 0 };
			}
		}
		return null;
	}

	/** Render markdown text to sanitised HTML. Content is from Fireflies API (trusted). */
	function renderMarkdown(text: string): string {
		if (!text) return '';
		const html = marked.parse(text, { async: false }) as string;
		// Strip <script> tags as a defence-in-depth precaution
		return html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
	}

	/** Extract ai_filters stats (tasks, metrics, questions, date_times). */
	function getAiFilters(insight: MeetingInsight): { label: string; value: number }[] | null {
		const c = parseContent(insight.content);
		if (typeof c !== 'object' || c === null || Array.isArray(c)) return null;
		const obj = c as Record<string, unknown>;
		const labels: Record<string, string> = {
			tasks: 'Tasks',
			metrics: 'Metrics',
			questions: 'Questions',
			date_times: 'Date/Times',
			prices: 'Prices'
		};
		const items: { label: string; value: number }[] = [];
		for (const [key, label] of Object.entries(labels)) {
			if (key in obj && typeof obj[key] === 'number' && (obj[key] as number) > 0) {
				items.push({ label, value: obj[key] as number });
			}
		}
		return items.length > 0 ? items : null;
	}

	/** Extract attendance data (attendees list with join/leave times). */
	function getAttendance(
		insight: MeetingInsight
	): { attendees: string[]; attendance: { name: string; join_time: string; leave_time: string }[] } | null {
		const c = parseContent(insight.content);
		if (typeof c !== 'object' || c === null || Array.isArray(c)) return null;
		const obj = c as Record<string, unknown>;
		if (!('attendees' in obj) && !('attendance' in obj)) return null;
		return {
			attendees: Array.isArray(obj.attendees) ? obj.attendees.map(String) : [],
			attendance: Array.isArray(obj.attendance)
				? (obj.attendance as { name: string; join_time: string; leave_time: string }[])
				: []
		};
	}

	/** Extract structured fields from Fireflies summary insight.
	 *  Fireflies summary content: {gist, overview, keywords, bullet_gist, short_summary, ...} */
	function getSummaryData(insight: MeetingInsight): {
		gist: string;
		shortSummary: string;
		overview: string;
		bulletGist: string;
	} | null {
		const c = parseContent(insight.content);
		if (typeof c !== 'object' || c === null || Array.isArray(c)) return null;
		const obj = c as Record<string, unknown>;
		const gist = typeof obj.gist === 'string' ? obj.gist : '';
		const shortSummary = typeof obj.short_summary === 'string' ? obj.short_summary : '';
		const overview = typeof obj.overview === 'string' ? obj.overview : '';
		const bulletGist = typeof obj.bullet_gist === 'string' ? obj.bullet_gist : '';
		if (!gist && !shortSummary && !overview && !bulletGist) return null;
		return { gist, shortSummary, overview, bulletGist };
	}

	let generating = $state(false);

	async function generateInsights() {
		if (!browser) return;
		generating = true;
		try {
			const res = await fetch(`/api/meetings/${meeting.id}/insights/generate`, {
				method: 'POST'
			});
			if (!res.ok) throw new Error(`${res.status}`);
			toastStore.success('Insights generated. Refresh to view.');
		} catch {
			toastStore.error('Failed to generate insights');
		} finally {
			generating = false;
		}
	}

	// --- Speakers tab (lazy loaded) ---
	let speakersData = $state<MeetingSpeaker[]>([]);
	let speakersLoaded = $state(false);
	let loadingSpeakers = $state(false);

	async function loadSpeakers() {
		if (speakersLoaded || !browser) return;
		loadingSpeakers = true;
		try {
			const res = await fetch(`/api/meetings/${meeting.id}/speakers`);
			if (!res.ok) throw new Error(`${res.status}`);
			const result = await res.json();
			speakersData = result.speakers;
			speakersLoaded = true;
		} catch {
			toastStore.error('Failed to load speaker data');
		} finally {
			loadingSpeakers = false;
		}
	}

	const totalTalkTime = $derived(speakersData.reduce((sum, s) => sum + s.talk_time_seconds, 0));

	// --- Active tab ---
	let activeTab = $state('transcript');

	// --- Diarization tab ---

	const diarizationJobs: DiarizationJob[] = $derived(data.diarizationJobs ?? []);
	const completedDiarizationJob = $derived(
		diarizationJobs.find((j) => j.status === 'completed') ?? null
	);

	let compareData = $state<TranscriptComparison | null>(null);
	let loadingCompare = $state(false);
	let applyingDiarization = $state(false);
	let compareLoaded = $state(false);

	async function loadCompare() {
		if (compareLoaded || !completedDiarizationJob || !browser) return;
		loadingCompare = true;
		try {
			const res = await fetch(`/api/diarization/meetings/${meeting.id}/compare`);
			if (!res.ok) throw new Error(`${res.status}`);
			compareData = await res.json();
			compareLoaded = true;
		} catch {
			toastStore.error('Failed to load diarization comparison');
		} finally {
			loadingCompare = false;
		}
	}

	async function applyDiarization() {
		applyingDiarization = true;
		try {
			const res = await fetch(`/api/diarization/meetings/${meeting.id}/apply`, { method: 'POST' });
			if (!res.ok) throw new Error(`${res.status}`);
			toastStore.success('Diarization applied — transcript updated with speaker names.');
		} catch {
			toastStore.error('Failed to apply diarization');
		} finally {
			applyingDiarization = false;
		}
	}
</script>

<!-- Page header with title and export -->
<div class="mb-4 flex items-center justify-between">
	<div>
		<h1 class="text-2xl font-bold">{meeting?.title ?? 'Meeting'}</h1>
		{#if meeting?.start_time}
			<p class="text-sm text-muted-foreground">
				{new Date(meeting.start_time).toLocaleDateString(undefined, {
					year: 'numeric',
					month: 'long',
					day: 'numeric'
				})}
			</p>
		{/if}
	</div>

	<div class="relative">
		<Button
			variant="outline"
			size="sm"
			onclick={() => (exportOpen = !exportOpen)}
		>
			Export &#9662;
		</Button>

		{#if exportOpen}
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<div
				class="absolute right-0 z-50 mt-1 w-56 rounded-md border bg-popover p-1 shadow-md"
				onmouseleave={() => (exportOpen = false)}
			>
				<p class="px-2 py-1.5 text-xs font-semibold text-muted-foreground">Transcript</p>
				<a href={exportApi.transcriptUrl(meeting.id, 'srt')} download class="block rounded-sm px-2 py-1.5 text-sm hover:bg-accent">SRT (subtitles)</a>
				<a href={exportApi.transcriptUrl(meeting.id, 'vtt')} download class="block rounded-sm px-2 py-1.5 text-sm hover:bg-accent">VTT (web subtitles)</a>
				<a href={exportApi.transcriptUrl(meeting.id, 'txt')} download class="block rounded-sm px-2 py-1.5 text-sm hover:bg-accent">Plain text</a>
				<a href={exportApi.transcriptUrl(meeting.id, 'pdf')} download class="block rounded-sm px-2 py-1.5 text-sm hover:bg-accent">PDF</a>

				<div class="my-1 h-px bg-border"></div>

				<p class="px-2 py-1.5 text-xs font-semibold text-muted-foreground">Translations</p>
				{#each sentences
					.flatMap((s) => s.translations ?? [])
					.map((t) => t.target_language)
					.filter((v, i, a) => a.indexOf(v) === i) as lang}
					<a href={exportApi.translationsUrl(meeting.id, lang, 'srt')} download class="block rounded-sm px-2 py-1.5 text-sm hover:bg-accent">{lang} (SRT)</a>
				{:else}
					<p class="px-2 py-1.5 text-xs text-muted-foreground italic">No translations available</p>
				{/each}

				<div class="my-1 h-px bg-border"></div>

				<a href={exportApi.archiveUrl(meeting.id)} download class="block rounded-sm px-2 py-1.5 text-sm font-medium hover:bg-accent">Download All (ZIP)</a>
			</div>
		{/if}
	</div>
</div>

<Tabs.Root bind:value={activeTab}>
	<Tabs.List>
		<Tabs.Trigger value="transcript">Transcript</Tabs.Trigger>
		<Tabs.Trigger value="translations">Translations</Tabs.Trigger>
		<Tabs.Trigger value="insights">Summary & Insights</Tabs.Trigger>
		<Tabs.Trigger value="speakers" onclick={loadSpeakers}>Speakers</Tabs.Trigger>
		<Tabs.Trigger value="diarization" onclick={loadCompare}>Diarization</Tabs.Trigger>
		<Tabs.Trigger value="media">Media & Links</Tabs.Trigger>
	</Tabs.List>

	<!-- Tab 1: Transcript -->
	<Tabs.Content value="transcript">
		<ErrorBoundary>
			{#snippet children()}
				<Card.Root>
					<Card.Header>
						<div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
							<Card.Title>Transcript ({filteredSentences.length})</Card.Title>
							<div class="flex gap-2">
								<Input
									placeholder="Search transcript..."
									aria-label="Search transcript"
									bind:value={transcriptSearch}
									class="w-48"
								/>
								{#if speakerFilter}
									<Button
										variant="ghost"
										size="sm"
										onclick={() => (speakerFilter = null)}
									>
										Clear filter
									</Button>
								{/if}
							</div>
						</div>
						{#if speakers.length > 0}
							<div class="mt-2 flex flex-wrap gap-1" role="group" aria-label="Filter by speaker">
								{#each speakers as speaker}
									<button
										class="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors hover:opacity-80"
										style="background-color: {getSpeakerColor(speaker)}20; color: {getSpeakerColor(
											speaker
										)}; border: 1px solid {getSpeakerColor(speaker)}40"
										class:ring-2={speakerFilter === speaker}
										aria-pressed={speakerFilter === speaker}
										onclick={() =>
											(speakerFilter =
												speakerFilter === speaker ? null : speaker)}
									>
										{speaker}
									</button>
								{/each}
							</div>
						{/if}
					</Card.Header>
					<Card.Content>
						{#if filteredSentences.length === 0}
							<div class="py-12 text-center">
								<p class="text-muted-foreground">
									{sentences.length === 0
										? 'Transcript is still being processed...'
										: 'No results match your search.'}
								</p>
							</div>
						{:else}
							<div class="max-h-[60vh] space-y-3 overflow-y-auto">
								{#each filteredSentences as sentence (sentence.id)}
									<div class="space-y-1 text-sm">
										<div class="flex items-center gap-2">
											{#if sentence.speaker_name}
												<button
													class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium"
													style="background-color: {getSpeakerColor(
														sentence.speaker_name
													)}20; color: {getSpeakerColor(
														sentence.speaker_name
													)}"
													onclick={() =>
														(speakerFilter = sentence.speaker_name)}
												>
													{sentence.speaker_name}
												</button>
											{/if}
											<span class="text-xs text-muted-foreground">
												{formatTimestamp(sentence.start_time)}
												{#if sentence.end_time}
													- {formatTimestamp(sentence.end_time)}
												{/if}
											</span>
										</div>
										<p>{sentence.text}</p>
										{#if sentence.translations?.length}
											{#each sentence.translations as translation}
												<p class="text-xs italic text-primary/80">
													{translation.translated_text}
													<span class="text-muted-foreground">
														({translation.target_language}
														{#if translation.confidence < 1}
															&middot; {Math.round(
																translation.confidence * 100
															)}%
														{/if})
													</span>
												</p>
											{/each}
										{/if}
									</div>
									<Separator />
								{/each}
							</div>
						{/if}
					</Card.Content>
				</Card.Root>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>

	<!-- Tab 2: Translations -->
	<Tabs.Content value="translations">
		<ErrorBoundary>
			{#snippet children()}
				<Card.Root>
					<Card.Header>
						<Card.Title>Translations</Card.Title>
					</Card.Header>
					<Card.Content>
						{#if sentences.length === 0}
							<p class="py-8 text-center text-muted-foreground">
								No transcript data available.
							</p>
						{:else}
							<div class="max-h-[60vh] overflow-y-auto">
								<table class="w-full text-sm">
									<thead class="sticky top-0 border-b bg-background">
										<tr>
											<th class="w-24 p-2 text-left">Speaker</th>
											<th class="p-2 text-left">Original</th>
											<th class="p-2 text-left">Translation</th>
											<th class="w-20 p-2 text-left">Confidence</th>
										</tr>
									</thead>
									<tbody>
										{#each sentences as sentence (sentence.id)}
											<tr class="border-b">
												<td class="p-2 align-top">
													<Badge variant="outline"
														>{sentence.speaker_name ??
															'Unknown'}</Badge
													>
												</td>
												<td class="p-2 align-top">{sentence.text}</td>
												<td class="p-2 align-top italic text-primary">
													{#if sentence.translations?.length}
														{sentence.translations[0].translated_text}
													{:else}
														<span class="text-muted-foreground"
															>--</span
														>
													{/if}
												</td>
												<td
													class="p-2 align-top text-xs text-muted-foreground"
												>
													{#if sentence.translations?.length}
														{Math.round(
															sentence.translations[0].confidence *
																100
														)}%
													{:else}
														--
													{/if}
												</td>
											</tr>
										{/each}
									</tbody>
								</table>
							</div>
						{/if}
					</Card.Content>
				</Card.Root>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>

	<!-- Tab 3: Summary & Insights -->
	<Tabs.Content value="insights">
		<ErrorBoundary>
			{#snippet children()}
				<div class="space-y-4">
					{#if insights.length === 0}
						<Card.Root>
							<Card.Content class="py-12">
								<div class="space-y-4 text-center">
									<div class="text-4xl">&#x1F4A1;</div>
									<h3 class="font-semibold">No insights yet</h3>
									<p class="text-sm text-muted-foreground">
										Generate AI insights from the meeting transcript.
									</p>
									<Button onclick={generateInsights} disabled={generating}>
										{generating ? 'Generating...' : 'Generate Insights'}
									</Button>
								</div>
							</Card.Content>
						</Card.Root>
					{:else}
						<!-- Summary — gist quote + short summary + markdown body -->
						{@const summaryItems = getInsightsByType('summary')}
						{#if summaryItems.length > 0}
							<Card.Root>
								<Card.Header>
									<div class="flex items-center justify-between">
										<Card.Title>Summary</Card.Title>
										<Badge variant="outline" class="text-xs">{summaryItems[0].source}</Badge>
									</div>
								</Card.Header>
								<Card.Content class="space-y-3">
									{#each summaryItems as insight (insight.id)}
										{@const sd = getSummaryData(insight)}
										{#if sd}
											{#if sd.gist}
												<blockquote class="border-l-4 border-primary/40 pl-4 italic text-muted-foreground">
													{sd.gist}
												</blockquote>
											{/if}
											{#if sd.shortSummary}
												<div class="prose prose-sm max-w-none dark:prose-invert">
													{@html renderMarkdown(sd.shortSummary)}
												</div>
											{/if}
											{#if sd.overview}
												<div class="prose prose-sm max-w-none dark:prose-invert">
													{@html renderMarkdown(sd.overview)}
												</div>
											{/if}
										{:else}
											<div class="prose prose-sm max-w-none dark:prose-invert">
												{@html renderMarkdown(getInsightText(insight))}
											</div>
										{/if}
									{/each}
								</Card.Content>
							</Card.Root>
						{/if}

						<!-- Overview — markdown rendered -->
						{@const overviewItems = getInsightsByType('overview')}
						{#if overviewItems.length > 0}
							<Card.Root>
								<Card.Header>
									<div class="flex items-center justify-between">
										<Card.Title>Overview</Card.Title>
										<Badge variant="outline" class="text-xs">{overviewItems[0].source}</Badge>
									</div>
								</Card.Header>
								<Card.Content>
									{#each overviewItems as insight (insight.id)}
										<div class="prose prose-sm max-w-none dark:prose-invert">
											{@html renderMarkdown(getInsightText(insight))}
										</div>
									{/each}
								</Card.Content>
							</Card.Root>
						{/if}

						<!-- Action Items — markdown rendered -->
						{@const actionItems = getInsightsByType('action_items')}
						{#if actionItems.length > 0}
							<Card.Root>
								<Card.Header>
									<div class="flex items-center justify-between">
										<Card.Title>Action Items</Card.Title>
										<Badge variant="outline" class="text-xs">{actionItems[0].source}</Badge>
									</div>
								</Card.Header>
								<Card.Content>
									{#each actionItems as insight (insight.id)}
										<div class="prose prose-sm max-w-none dark:prose-invert">
											{@html renderMarkdown(getInsightText(insight))}
										</div>
									{/each}
								</Card.Content>
							</Card.Root>
						{/if}

						<!-- Keywords — tag/badge list -->
						{@const keywordInsights = getInsightsByType('keywords')}
						{#if keywordInsights.length > 0}
							<Card.Root>
								<Card.Header>
									<Card.Title>Keywords</Card.Title>
								</Card.Header>
								<Card.Content>
									{#each keywordInsights as insight (insight.id)}
										{@const kws = getKeywords(insight)}
										<div class="flex flex-wrap gap-2">
											{#each kws as kw}
												<Badge variant="secondary">{kw}</Badge>
											{/each}
										</div>
									{/each}
								</Card.Content>
							</Card.Root>
						{/if}

						<!-- Outline / Shorthand Bullet — markdown rendered -->
						{#each ['outline', 'shorthand_bullet'] as insightType}
							{@const items = getInsightsByType(insightType)}
							{#if items.length > 0}
								<Card.Root>
									<Card.Header>
										<Card.Title class="capitalize">{insightType.replace(/_/g, ' ')}</Card.Title>
									</Card.Header>
									<Card.Content>
										{#each items as insight (insight.id)}
											<div class="prose prose-sm max-w-none dark:prose-invert">
												{@html renderMarkdown(getInsightText(insight))}
											</div>
										{/each}
									</Card.Content>
								</Card.Root>
							{/if}
						{/each}

						<!-- Sentiment — percentage bars -->
						{@const sentimentInsights = getInsightsByType('sentiment')}
						{#if sentimentInsights.length > 0}
							<Card.Root>
								<Card.Header>
									<Card.Title>Sentiment</Card.Title>
								</Card.Header>
								<Card.Content>
									{#each sentimentInsights as insight (insight.id)}
										{@const sent = getSentiment(insight)}
										{#if sent}
											<div class="space-y-2">
												{#each [
													{ label: 'Positive', value: sent.positive, color: 'bg-green-500' },
													{ label: 'Neutral', value: sent.neutral, color: 'bg-gray-400' },
													{ label: 'Negative', value: sent.negative, color: 'bg-red-500' }
												] as bar}
													<div class="flex items-center gap-2">
														<span class="w-16 text-sm">{bar.label}</span>
														<div class="h-4 flex-1 overflow-hidden rounded-full bg-muted">
															<div
																class="h-full rounded-full {bar.color}"
																style="width: {bar.value}%"
															></div>
														</div>
														<span class="w-10 text-right text-xs text-muted-foreground">
															{Math.round(bar.value)}%
														</span>
													</div>
												{/each}
											</div>
										{:else}
											<p class="whitespace-pre-wrap text-sm">{getInsightText(insight)}</p>
										{/if}
									{/each}
								</Card.Content>
							</Card.Root>
						{/if}

						<!-- AI Filters — stat badges -->
						{@const aiFilterInsights = getInsightsByType('ai_filters')}
						{#if aiFilterInsights.length > 0}
							<Card.Root>
								<Card.Header>
									<Card.Title>AI Filters</Card.Title>
								</Card.Header>
								<Card.Content>
									{#each aiFilterInsights as insight (insight.id)}
										{@const filters = getAiFilters(insight)}
										{#if filters}
											<div class="flex flex-wrap gap-3">
												{#each filters as f}
													<div class="flex items-center gap-1.5 rounded-lg border px-3 py-2">
														<span class="text-lg font-semibold">{f.value}</span>
														<span class="text-sm text-muted-foreground">{f.label}</span>
													</div>
												{/each}
											</div>
										{:else}
											<div class="prose prose-sm max-w-none dark:prose-invert">
												{@html renderMarkdown(getInsightText(insight))}
											</div>
										{/if}
									{/each}
								</Card.Content>
							</Card.Root>
						{/if}

						<!-- Attendance — attendee table -->
						{@const attendanceInsights = getInsightsByType('attendance')}
						{#if attendanceInsights.length > 0}
							<Card.Root>
								<Card.Header>
									<Card.Title>Attendance</Card.Title>
								</Card.Header>
								<Card.Content>
									{#each attendanceInsights as insight (insight.id)}
										{@const att = getAttendance(insight)}
										{#if att}
											{#if att.attendance.length > 0}
												<div class="overflow-x-auto">
													<table class="w-full text-sm">
														<thead class="border-b">
															<tr>
																<th class="p-2 text-left">Name</th>
																<th class="p-2 text-left">Joined</th>
																<th class="p-2 text-left">Left</th>
															</tr>
														</thead>
														<tbody>
															{#each att.attendance as person}
																<tr class="border-b">
																	<td class="p-2">{person.name}</td>
																	<td class="p-2 text-muted-foreground">{person.join_time ?? '—'}</td>
																	<td class="p-2 text-muted-foreground">{person.leave_time ?? '—'}</td>
																</tr>
															{/each}
														</tbody>
													</table>
												</div>
											{:else if att.attendees.length > 0}
												<div class="flex flex-wrap gap-2">
													{#each att.attendees as name}
														<Badge variant="outline">{name}</Badge>
													{/each}
												</div>
											{/if}
										{:else}
											<div class="prose prose-sm max-w-none dark:prose-invert">
												{@html renderMarkdown(getInsightText(insight))}
											</div>
										{/if}
									{/each}
								</Card.Content>
							</Card.Root>
						{/if}

						<!-- Decisions / Questions / other types — markdown rendered -->
						{#each ['decisions', 'questions'] as insightType}
							{@const items = getInsightsByType(insightType)}
							{#if items.length > 0}
								<Card.Root>
									<Card.Header>
										<Card.Title class="capitalize">{insightType.replace(/_/g, ' ')}</Card.Title>
									</Card.Header>
									<Card.Content>
										{#each items as insight (insight.id)}
											<div class="prose prose-sm max-w-none dark:prose-invert">
												{@html renderMarkdown(getInsightText(insight))}
											</div>
										{/each}
									</Card.Content>
								</Card.Root>
							{/if}
						{/each}

						<div class="flex justify-center">
							<Button
								variant="outline"
								onclick={generateInsights}
								disabled={generating}
							>
								{generating ? 'Regenerating...' : 'Regenerate Insights'}
							</Button>
						</div>
					{/if}
				</div>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>

	<!-- Tab 4: Speakers -->
	<Tabs.Content value="speakers">
		<ErrorBoundary>
			{#snippet children()}
				<Card.Root>
					<Card.Header>
						<Card.Title>Speaker Analytics</Card.Title>
					</Card.Header>
					<Card.Content>
						{#if loadingSpeakers}
							<div
								class="animate-pulse py-8 text-center text-muted-foreground"
							>
								Loading speakers...
							</div>
						{:else if speakersData.length === 0}
							<div class="py-12 text-center">
								<p class="text-muted-foreground">
									Speaker data will appear after the meeting completes and
									syncs.
								</p>
							</div>
						{:else}
							<!-- Talk time bar chart -->
							{#if totalTalkTime > 0}
								<div class="mb-6 space-y-2">
									<h4 class="text-sm font-medium">Talk Time Distribution</h4>
									{#each speakersData as speaker (speaker.id)}
										{@const percent = Math.round(
											(speaker.talk_time_seconds / totalTalkTime) * 100
										)}
										<div class="flex items-center gap-2">
											<span class="w-24 truncate text-sm"
												>{speaker.speaker_name}</span
											>
											<div
												class="h-5 flex-1 overflow-hidden rounded-full bg-muted"
											role="progressbar"
											aria-valuenow={percent}
											aria-valuemin={0}
											aria-valuemax={100}
											aria-label="{speaker.speaker_name} talk time: {percent}%"
										>
												<div
													class="h-full rounded-full transition-all"
													style="width: {percent}%; background-color: {getSpeakerColor(
														speaker.speaker_name
													)}"
												></div>
											</div>
											<span
												class="w-12 text-right text-xs text-muted-foreground"
												>{percent}%</span
											>
										</div>
									{/each}
								</div>
								<Separator class="my-4" />
							{/if}

							<!-- Speaker cards -->
							<div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
								{#each speakersData as speaker (speaker.id)}
									<Card.Root>
										<Card.Content class="p-4">
											<div class="mb-2 flex items-center gap-2">
												<div
													class="size-3 rounded-full"
													style="background-color: {getSpeakerColor(
														speaker.speaker_name
													)}"
												></div>
												<span class="font-medium"
													>{speaker.speaker_name}</span
												>
											</div>
											{#if speaker.email}
												<p
													class="mb-2 text-xs text-muted-foreground"
												>
													{speaker.email}
												</p>
											{/if}
											{@const a = speaker.analytics ?? {}}
											<div class="grid grid-cols-2 gap-2 text-sm">
												<div>
													<span class="text-muted-foreground">Talk time:</span>
													<span>{Math.round(speaker.talk_time_seconds / 60)}m</span>
												</div>
												<div>
													<span class="text-muted-foreground">Words:</span>
													<span>{speaker.word_count.toLocaleString()}</span>
												</div>
												{#if a.words_per_minute}
													<div>
														<span class="text-muted-foreground">WPM:</span>
														<span>{Math.round(Number(a.words_per_minute))}</span>
													</div>
												{/if}
												{#if a.filler_words != null}
													<div>
														<span class="text-muted-foreground">Fillers:</span>
														<span>{a.filler_words}</span>
													</div>
												{/if}
												{#if a.questions != null}
													<div>
														<span class="text-muted-foreground">Questions:</span>
														<span>{a.questions}</span>
													</div>
												{/if}
												{#if a.longest_monologue != null}
													<div>
														<span class="text-muted-foreground">Longest monologue:</span>
														<span>{Math.round(Number(a.longest_monologue))}s</span>
													</div>
												{/if}
												{#if speaker.sentiment_score != null}
													<div class="col-span-2">
														<span class="text-muted-foreground">Sentiment:</span>
														<span>
															{speaker.sentiment_score > 0.3
																? 'Positive'
																: speaker.sentiment_score < -0.3
																	? 'Negative'
																	: 'Neutral'}
															({speaker.sentiment_score.toFixed(2)})
														</span>
													</div>
												{/if}
											</div>
										</Card.Content>
									</Card.Root>
								{/each}
							</div>
						{/if}
					</Card.Content>
				</Card.Root>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>

	<!-- Tab 5: Diarization -->
	<Tabs.Content value="diarization">
		<ErrorBoundary>
			{#snippet children()}
				<div class="space-y-4">
					{#if diarizationJobs.length === 0}
						<Card.Root>
							<Card.Content class="py-12">
								<div class="space-y-4 text-center">
									<div class="text-4xl">&#x1F3A4;</div>
									<h3 class="font-semibold">No diarization jobs yet</h3>
									<p class="text-sm text-muted-foreground">
										Use the "Diarize" button in the header to run offline speaker diarization
										with VibeVoice-ASR on this meeting's audio.
									</p>
								</div>
							</Card.Content>
						</Card.Root>
					{:else}
						<!-- Job Status Cards -->
						<Card.Root>
							<Card.Header>
								<Card.Title>Diarization Jobs</Card.Title>
							</Card.Header>
							<Card.Content>
								<div class="space-y-3">
									{#each diarizationJobs as job (job.job_id)}
										<div class="flex items-start justify-between rounded-md border p-3">
											<div class="space-y-1">
												<div class="flex items-center gap-2">
													<Badge
														variant={job.status === 'completed' ? 'default' : job.status === 'failed' ? 'destructive' : 'secondary'}
														class={['processing', 'downloading', 'mapping'].includes(job.status) ? 'animate-pulse' : ''}
													>
														{job.status}
													</Badge>
													{#if job.detected_language}
														<span class="text-xs text-muted-foreground">
															Language: {job.detected_language}
														</span>
													{/if}
													{#if job.num_speakers_detected != null}
														<span class="text-xs text-muted-foreground">
															{job.num_speakers_detected} speaker{job.num_speakers_detected !== 1 ? 's' : ''} detected
														</span>
													{/if}
												</div>
												{#if job.created_at}
													<p class="text-xs text-muted-foreground">
														Started: {new Date(job.created_at).toLocaleString()}
													</p>
												{/if}
												{#if job.completed_at}
													<p class="text-xs text-muted-foreground">
														Completed: {new Date(job.completed_at).toLocaleString()}
														{#if job.processing_time_seconds != null}
															({Math.round(job.processing_time_seconds)}s)
														{/if}
													</p>
												{/if}
												{#if job.error_message}
													<p class="text-xs text-destructive">{job.error_message}</p>
												{/if}
											</div>
											{#if job.status === 'completed'}
												<Button
													variant="outline"
													size="sm"
													onclick={loadCompare}
													disabled={loadingCompare}
												>
													{loadingCompare ? 'Loading...' : 'Compare'}
												</Button>
											{/if}
										</div>
									{/each}
								</div>
							</Card.Content>
						</Card.Root>

						<!-- Speaker Map (if completed job has one) -->
						{#if completedDiarizationJob?.speaker_map && Object.keys(completedDiarizationJob.speaker_map).length > 0}
							<Card.Root>
								<Card.Header>
									<Card.Title>Speaker Map</Card.Title>
								</Card.Header>
								<Card.Content>
									<div class="overflow-x-auto">
										<table class="w-full text-sm">
											<thead class="border-b">
												<tr>
													<th class="p-2 text-left">Diarization Label</th>
													<th class="p-2 text-left">Matched Name</th>
													<th class="p-2 text-left">Confidence</th>
													<th class="p-2 text-left">Method</th>
												</tr>
											</thead>
											<tbody>
												{#each Object.entries(completedDiarizationJob.speaker_map) as [label, entry]}
													<tr class="border-b">
														<td class="p-2 font-mono text-xs">{label}</td>
														<td class="p-2">{entry.name}</td>
														<td class="p-2 text-muted-foreground">
															{Math.round(entry.confidence * 100)}%
														</td>
														<td class="p-2 text-xs text-muted-foreground">{entry.method}</td>
													</tr>
												{/each}
											</tbody>
										</table>
									</div>
								</Card.Content>
							</Card.Root>
						{/if}

						<!-- Compare View -->
						{#if compareData}
							<Card.Root>
								<Card.Header>
									<div class="flex items-center justify-between">
										<Card.Title>Transcript Comparison</Card.Title>
										<Button
											variant="default"
											size="sm"
											onclick={applyDiarization}
											disabled={applyingDiarization}
										>
											{applyingDiarization ? 'Applying...' : 'Apply Diarization'}
										</Button>
									</div>
									<p class="text-sm text-muted-foreground">
										Side-by-side view of Fireflies transcript vs VibeVoice-ASR diarization.
										Click "Apply Diarization" to update the meeting transcript with the new speaker names.
									</p>
								</Card.Header>
								<Card.Content>
									<div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
										<!-- Fireflies column -->
										<div>
											<h4 class="mb-2 text-sm font-semibold">Fireflies Transcript</h4>
											<div class="max-h-[60vh] space-y-2 overflow-y-auto rounded-md border p-3">
												{#if compareData.fireflies_sentences.length === 0}
													<p class="text-sm text-muted-foreground">No Fireflies sentences available.</p>
												{:else}
													{#each compareData.fireflies_sentences as sentence}
														<div class="space-y-0.5 text-sm">
															{#if sentence.speaker_name}
																<span class="text-xs font-medium text-primary">
																	{String(sentence.speaker_name)}
																</span>
															{/if}
															<p class="text-muted-foreground">{String(sentence.text ?? '')}</p>
														</div>
														<Separator />
													{/each}
												{/if}
											</div>
										</div>

										<!-- VibeVoice column -->
										<div>
											<h4 class="mb-2 text-sm font-semibold">VibeVoice-ASR Diarization</h4>
											<div class="max-h-[60vh] space-y-2 overflow-y-auto rounded-md border p-3">
												{#if compareData.vibevoice_segments.length === 0}
													<p class="text-sm text-muted-foreground">No VibeVoice segments available.</p>
												{:else}
													{#each compareData.vibevoice_segments as segment}
														<div class="space-y-0.5 text-sm">
															{#if segment.speaker}
																<span class="text-xs font-medium text-primary">
																	{String(segment.speaker)}
																	{#if compareData.speaker_map?.[String(segment.speaker)]}
																		<span class="text-muted-foreground">
																			→ {compareData.speaker_map[String(segment.speaker)].name}
																		</span>
																	{/if}
																</span>
															{/if}
															{#if segment.start != null || segment.end != null}
																<span class="text-xs text-muted-foreground">
																	{segment.start != null ? String(Math.floor(Number(segment.start) / 60)).padStart(2, '0') + ':' + String(Math.floor(Number(segment.start) % 60)).padStart(2, '0') : '--'}
																</span>
															{/if}
															<p class="text-muted-foreground">{String(segment.text ?? '')}</p>
														</div>
														<Separator />
													{/each}
												{/if}
											</div>
										</div>
									</div>
								</Card.Content>
							</Card.Root>
						{:else if completedDiarizationJob && !loadingCompare && !compareLoaded}
							<Card.Root>
								<Card.Content class="py-8">
									<div class="space-y-3 text-center">
										<p class="text-sm text-muted-foreground">
											A completed diarization job is available. Load the side-by-side comparison to review
											and apply speaker names to the transcript.
										</p>
										<Button variant="outline" onclick={loadCompare} disabled={loadingCompare}>
											{loadingCompare ? 'Loading comparison...' : 'Load Comparison'}
										</Button>
									</div>
								</Card.Content>
							</Card.Root>
						{/if}
					{/if}
				</div>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>

	<!-- Tab 6: Media & Links -->
	<Tabs.Content value="media">
		<ErrorBoundary>
			{#snippet children()}
				<Card.Root>
					<Card.Header>
						<Card.Title>Media & Links</Card.Title>
					</Card.Header>
					<Card.Content class="space-y-4">
						{@const hasMedia =
							meeting.audio_url ||
							meeting.video_url ||
							meeting.transcript_url ||
							meeting.meeting_link}

						{#if !hasMedia && !meeting.participants?.length && !meeting.organizer_email}
							<div class="py-12 text-center">
								<p class="text-muted-foreground">
									Media links will appear after the meeting syncs from
									Fireflies.
								</p>
							</div>
						{:else}
							{#if meeting.audio_url}
								<div
									class="flex items-center justify-between rounded-md border p-3"
								>
									<div>
										<p class="text-sm font-medium">Audio Recording</p>
										<p
											class="max-w-md truncate text-xs text-muted-foreground"
										>
											{meeting.audio_url}
										</p>
									</div>
									<Button
										variant="outline"
										size="sm"
										href={meeting.audio_url}
										target="_blank"
									>
										Open
									</Button>
								</div>
							{/if}

							{#if meeting.video_url}
								<div
									class="flex items-center justify-between rounded-md border p-3"
								>
									<div>
										<p class="text-sm font-medium">Video Recording</p>
										<p
											class="max-w-md truncate text-xs text-muted-foreground"
										>
											{meeting.video_url}
										</p>
									</div>
									<Button
										variant="outline"
										size="sm"
										href={meeting.video_url}
										target="_blank"
									>
										Open
									</Button>
								</div>
							{/if}

							{#if meeting.transcript_url}
								<div
									class="flex items-center justify-between rounded-md border p-3"
								>
									<div>
										<p class="text-sm font-medium">Fireflies Transcript</p>
										<p
											class="max-w-md truncate text-xs text-muted-foreground"
										>
											{meeting.transcript_url}
										</p>
									</div>
									<Button
										variant="outline"
										size="sm"
										href={meeting.transcript_url}
										target="_blank"
									>
										Open
									</Button>
								</div>
							{/if}

							{#if meeting.meeting_link}
								<div
									class="flex items-center justify-between rounded-md border p-3"
								>
									<div>
										<p class="text-sm font-medium">Meeting Link</p>
										<p
											class="max-w-md truncate text-xs text-muted-foreground"
										>
											{meeting.meeting_link}
										</p>
									</div>
									<Button
										variant="outline"
										size="sm"
										href={meeting.meeting_link}
										target="_blank"
									>
										Open
									</Button>
								</div>
							{/if}

							{#if meeting.organizer_email}
								<Separator />
								<div>
									<p class="mb-1 text-sm font-medium">Organizer</p>
									<p class="text-sm text-muted-foreground">
										{meeting.organizer_email}
									</p>
								</div>
							{/if}

							{#if meeting.participants?.length}
								<div>
									<p class="mb-2 text-sm font-medium">Participants</p>
									<div class="flex flex-wrap gap-1">
										{#each meeting.participants as participant}
											<Badge variant="outline">{participant}</Badge>
										{/each}
									</div>
								</div>
							{/if}
						{/if}
					</Card.Content>
				</Card.Root>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>
</Tabs.Root>
