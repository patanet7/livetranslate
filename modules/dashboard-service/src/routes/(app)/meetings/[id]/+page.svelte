<script lang="ts">
	import { browser } from '$app/environment';
	import * as Tabs from '$lib/components/ui/tabs';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { Input } from '$lib/components/ui/input';
	import { Separator } from '$lib/components/ui/separator';
	import ErrorBoundary from '$lib/components/ErrorBoundary.svelte';
	import { toastStore } from '$lib/stores/toast.svelte';
	import type { MeetingSpeaker, MeetingSentence, MeetingInsight } from '$lib/types';

	let { data } = $props();

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

	function getInsightText(insight: MeetingInsight): string {
		const content = insight.content;
		if (typeof content === 'string') return content;
		if (content?.text && typeof content.text === 'string') return content.text;
		return JSON.stringify(content, null, 2);
	}

	function getInsightItems(insight: MeetingInsight): string[] {
		const content: unknown = insight.content;
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
		const content: unknown = insight.content;
		if (Array.isArray(content)) return content.map(String);
		if (typeof content === 'string') return content.split(/[,\n]/).map((s: string) => s.trim()).filter(Boolean);
		if (typeof content === 'object' && content && 'text' in content) {
			const text = (content as Record<string, unknown>).text;
			if (typeof text === 'string') return text.split(/[,\n]/).map((s: string) => s.trim()).filter(Boolean);
		}
		return [];
	}

	function getSentiment(insight: MeetingInsight): { positive: number; neutral: number; negative: number } | null {
		const content = insight.content;
		if (typeof content === 'object' && content && !Array.isArray(content)) {
			const c = content as Record<string, unknown>;
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
</script>

<Tabs.Root bind:value={activeTab}>
	<Tabs.List>
		<Tabs.Trigger value="transcript">Transcript</Tabs.Trigger>
		<Tabs.Trigger value="translations">Translations</Tabs.Trigger>
		<Tabs.Trigger value="insights">Summary & Insights</Tabs.Trigger>
		<Tabs.Trigger value="speakers" onclick={loadSpeakers}>Speakers</Tabs.Trigger>
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
						<!-- Overview / Summary — full text paragraphs -->
						{#each ['overview', 'summary'] as insightType}
							{@const items = getInsightsByType(insightType)}
							{#if items.length > 0}
								<Card.Root>
									<Card.Header>
										<div class="flex items-center justify-between">
											<Card.Title class="capitalize">{insightType}</Card.Title>
											<Badge variant="outline" class="text-xs">{items[0].source}</Badge>
										</div>
									</Card.Header>
									<Card.Content>
										{#each items as insight (insight.id)}
											<div class="prose prose-sm max-w-none whitespace-pre-wrap dark:prose-invert">
												{getInsightText(insight)}
											</div>
										{/each}
									</Card.Content>
								</Card.Root>
							{/if}
						{/each}

						<!-- Action Items — bullet list -->
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
										{@const items = getInsightItems(insight)}
										{#if items.length > 0}
											<ul class="list-disc space-y-1 pl-5 text-sm">
												{#each items as item}
													<li>{item}</li>
												{/each}
											</ul>
										{:else}
											<p class="whitespace-pre-wrap text-sm">{getInsightText(insight)}</p>
										{/if}
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

						<!-- Outline / Shorthand Bullet — structured list -->
						{#each ['outline', 'shorthand_bullet'] as insightType}
							{@const items = getInsightsByType(insightType)}
							{#if items.length > 0}
								<Card.Root>
									<Card.Header>
										<Card.Title class="capitalize">{insightType.replace(/_/g, ' ')}</Card.Title>
									</Card.Header>
									<Card.Content>
										{#each items as insight (insight.id)}
											{@const bullets = getInsightItems(insight)}
											{#if bullets.length > 0}
												<ul class="list-disc space-y-1 pl-5 text-sm">
													{#each bullets as bullet}
														<li>{bullet}</li>
													{/each}
												</ul>
											{:else}
												<p class="whitespace-pre-wrap text-sm">{getInsightText(insight)}</p>
											{/if}
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

						<!-- Decisions / other types — fallback text display -->
						{#each ['decisions', 'questions', 'ai_filters'] as insightType}
							{@const items = getInsightsByType(insightType)}
							{#if items.length > 0}
								<Card.Root>
									<Card.Header>
										<Card.Title class="capitalize">{insightType.replace(/_/g, ' ')}</Card.Title>
									</Card.Header>
									<Card.Content>
										{#each items as insight (insight.id)}
											<div class="prose prose-sm max-w-none whitespace-pre-wrap dark:prose-invert">
												{getInsightText(insight)}
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

	<!-- Tab 5: Media & Links -->
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
