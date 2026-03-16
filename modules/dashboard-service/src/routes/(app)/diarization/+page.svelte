<script lang="ts">
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Tabs from '$lib/components/ui/tabs';
	import { toastStore } from '$lib/stores/toast.svelte';
	import type { DiarizationJob, SpeakerProfile, DiarizationRules } from '$lib/api/diarization';

	// ── Data ───────────────────────────────────────────────────────────

	let { data } = $props();

	// ── Jobs State ─────────────────────────────────────────────────────

	let jobs = $state<DiarizationJob[]>([]);
	let cancellingJobId = $state<string | null>(null);

	// ── Speakers State ─────────────────────────────────────────────────

	let speakers = $state<SpeakerProfile[]>([]);
	let showAddSpeaker = $state(false);
	let newSpeakerName = $state('');
	let newSpeakerEmail = $state('');
	let addingSpeaker = $state(false);

	// ── Rules State ────────────────────────────────────────────────────

	let rulesEnabled = $state(false);
	let participantPatterns = $state('');
	let titlePatterns = $state('');
	let minDurationMinutes = $state(0);
	let excludeEmpty = $state(false);

	// Sync from server load data
	$effect(() => { jobs = data.jobs; });
	$effect(() => { speakers = data.speakers; });
	$effect(() => { rulesEnabled = data.rules.enabled; });
	$effect(() => { participantPatterns = data.rules.participant_patterns.join(', '); });
	$effect(() => { titlePatterns = data.rules.title_patterns.join(', '); });
	$effect(() => { minDurationMinutes = data.rules.min_duration_minutes; });
	$effect(() => { excludeEmpty = data.rules.exclude_empty; });
	let savingRules = $state(false);

	// ── Helpers ────────────────────────────────────────────────────────

	function statusBadgeClass(status: DiarizationJob['status']): string {
		switch (status) {
			case 'queued':
				return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
			case 'downloading':
			case 'processing':
			case 'mapping':
				return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
			case 'completed':
				return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
			case 'failed':
				return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
			case 'cancelled':
				return 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300';
			default:
				return '';
		}
	}

	function formatDate(iso: string): string {
		try {
			return new Date(iso).toLocaleString();
		} catch {
			return iso;
		}
	}

	function splitPatterns(value: string): string[] {
		return value
			.split(',')
			.map((s) => s.trim())
			.filter(Boolean);
	}

	// ── Jobs Actions ───────────────────────────────────────────────────

	async function cancelJob(id: string) {
		cancellingJobId = id;
		try {
			const res = await fetch(`/api/diarization/jobs/${id}/cancel`, { method: 'POST' });
			if (res.ok) {
				toastStore.success('Job cancelled');
				jobs = jobs.map((j) => (j.job_id === id ? { ...j, status: 'cancelled' as const } : j));
			} else {
				const body = await res.json().catch(() => null);
				toastStore.error(body?.detail ?? 'Failed to cancel job');
			}
		} catch (err) {
			toastStore.error(
				err instanceof TypeError && err.message === 'Failed to fetch'
					? 'Connection error. Please check your network and try again.'
					: 'Network error cancelling job'
			);
		} finally {
			cancellingJobId = null;
		}
	}

	// ── Speakers Actions ───────────────────────────────────────────────

	async function addSpeaker() {
		if (!newSpeakerName.trim()) return;
		addingSpeaker = true;
		try {
			const res = await fetch('/api/diarization/speakers', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					name: newSpeakerName.trim(),
					email: newSpeakerEmail.trim() || null
				})
			});
			if (res.ok) {
				const created: SpeakerProfile = await res.json();
				speakers = [...speakers, created];
				newSpeakerName = '';
				newSpeakerEmail = '';
				showAddSpeaker = false;
				toastStore.success('Speaker added');
			} else {
				const body = await res.json().catch(() => null);
				toastStore.error(body?.detail ?? 'Failed to add speaker');
			}
		} catch (err) {
			toastStore.error(
				err instanceof TypeError && err.message === 'Failed to fetch'
					? 'Connection error. Please check your network and try again.'
					: 'Network error adding speaker'
			);
		} finally {
			addingSpeaker = false;
		}
	}

	// ── Rules Actions ──────────────────────────────────────────────────

	async function saveRules() {
		savingRules = true;
		try {
			const payload: DiarizationRules = {
				enabled: rulesEnabled,
				participant_patterns: splitPatterns(participantPatterns),
				title_patterns: splitPatterns(titlePatterns),
				min_duration_minutes: minDurationMinutes,
				exclude_empty: excludeEmpty
			};
			const res = await fetch('/api/diarization/rules', {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload)
			});
			if (res.ok) {
				toastStore.success('Rules saved');
			} else {
				const body = await res.json().catch(() => null);
				toastStore.error(body?.detail ?? 'Failed to save rules');
			}
		} catch (err) {
			toastStore.error(
				err instanceof TypeError && err.message === 'Failed to fetch'
					? 'Connection error. Please check your network and try again.'
					: 'Network error saving rules'
			);
		} finally {
			savingRules = false;
		}
	}
</script>

<PageHeader
	title="Diarization"
	description="Offline speaker diarization with VibeVoice-ASR"
/>

<Tabs.Root value="jobs">
	<Tabs.List>
		<Tabs.Trigger value="jobs">Jobs</Tabs.Trigger>
		<Tabs.Trigger value="speakers">Speakers</Tabs.Trigger>
		<Tabs.Trigger value="rules">Rules</Tabs.Trigger>
	</Tabs.List>

	<!-- ════════════════════════════════════════════════════════════════ -->
	<!-- Tab 1: Jobs                                                     -->
	<!-- ════════════════════════════════════════════════════════════════ -->
	<Tabs.Content value="jobs">
		<div class="mt-4">
			<Card.Root>
				<Card.Header>
					<Card.Title>Diarization Jobs</Card.Title>
					<Card.Description>
						Offline speaker diarization job queue and history
					</Card.Description>
				</Card.Header>
				<Card.Content>
					{#if jobs.length === 0}
						<p class="text-sm text-muted-foreground py-8 text-center">
							No diarization jobs found.
						</p>
					{:else}
						<div class="overflow-x-auto">
							<table class="w-full text-sm">
								<thead>
									<tr class="border-b text-left text-xs text-muted-foreground uppercase tracking-wide">
										<th class="pb-3 pr-4 font-medium">Meeting ID</th>
										<th class="pb-3 pr-4 font-medium">Status</th>
										<th class="pb-3 pr-4 font-medium">Triggered By</th>
										<th class="pb-3 pr-4 font-medium">Speakers</th>
										<th class="pb-3 pr-4 font-medium">Created</th>
										<th class="pb-3 font-medium"></th>
									</tr>
								</thead>
								<tbody class="divide-y">
									{#each jobs as job (job.job_id)}
										<tr class="hover:bg-muted/40 transition-colors">
											<td class="py-3 pr-4 font-mono text-xs">
												{job.meeting_id}
											</td>
											<td class="py-3 pr-4">
												<span
													class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {statusBadgeClass(
														job.status
													)}"
												>
													{job.status}
												</span>
											</td>
											<td class="py-3 pr-4 text-muted-foreground">
												{job.triggered_by}
											</td>
											<td class="py-3 pr-4 text-muted-foreground">
												{job.num_speakers_detected ?? '—'}
											</td>
											<td class="py-3 pr-4 text-muted-foreground whitespace-nowrap">
												{formatDate(job.created_at ?? '')}
											</td>
											<td class="py-3">
												{#if job.status === 'queued'}
													<Button
														variant="outline"
														size="sm"
														disabled={cancellingJobId === job.job_id}
														onclick={() => cancelJob(job.job_id)}
													>
														{cancellingJobId === job.job_id ? 'Cancelling...' : 'Cancel'}
													</Button>
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
		</div>
	</Tabs.Content>

	<!-- ════════════════════════════════════════════════════════════════ -->
	<!-- Tab 2: Speakers                                                 -->
	<!-- ════════════════════════════════════════════════════════════════ -->
	<Tabs.Content value="speakers">
		<div class="mt-4 space-y-4">
			<Card.Root>
				<Card.Header>
					<div class="flex items-center justify-between">
						<div>
							<Card.Title>Speaker Profiles</Card.Title>
							<Card.Description>
								Known speakers used for identity mapping during diarization
							</Card.Description>
						</div>
						<Button
							variant="outline"
							size="sm"
							onclick={() => {
								showAddSpeaker = !showAddSpeaker;
							}}
						>
							{showAddSpeaker ? 'Cancel' : 'Add Speaker'}
						</Button>
					</div>
				</Card.Header>
				<Card.Content>
					{#if showAddSpeaker}
						<div class="mb-6 rounded-lg border p-4 bg-muted/30 space-y-3">
							<p class="text-sm font-medium">New Speaker</p>
							<div class="grid gap-3 sm:grid-cols-2">
								<div class="space-y-1.5">
									<Label for="speaker-name">Name</Label>
									<Input
										id="speaker-name"
										placeholder="Full name"
										bind:value={newSpeakerName}
										onkeydown={(e: KeyboardEvent) => {
											if (e.key === 'Enter') addSpeaker();
										}}
									/>
								</div>
								<div class="space-y-1.5">
									<Label for="speaker-email">Email (optional)</Label>
									<Input
										id="speaker-email"
										type="email"
										placeholder="email@example.com"
										bind:value={newSpeakerEmail}
										onkeydown={(e: KeyboardEvent) => {
											if (e.key === 'Enter') addSpeaker();
										}}
									/>
								</div>
							</div>
							<Button
								disabled={addingSpeaker || !newSpeakerName.trim()}
								onclick={addSpeaker}
							>
								{addingSpeaker ? 'Adding...' : 'Add Speaker'}
							</Button>
						</div>
					{/if}

					{#if speakers.length === 0}
						<p class="text-sm text-muted-foreground py-8 text-center">
							No speaker profiles yet. Add a speaker to enable identity mapping.
						</p>
					{:else}
						<div class="overflow-x-auto">
							<table class="w-full text-sm">
								<thead>
									<tr class="border-b text-left text-xs text-muted-foreground uppercase tracking-wide">
										<th class="pb-3 pr-4 font-medium">Name</th>
										<th class="pb-3 pr-4 font-medium">Email</th>
										<th class="pb-3 pr-4 font-medium">Source</th>
										<th class="pb-3 font-medium">Samples</th>
									</tr>
								</thead>
								<tbody class="divide-y">
									{#each speakers as speaker (speaker.id)}
										<tr class="hover:bg-muted/40 transition-colors">
											<td class="py-3 pr-4 font-medium">{speaker.name}</td>
											<td class="py-3 pr-4 text-muted-foreground">
												{speaker.email ?? '—'}
											</td>
											<td class="py-3 pr-4">
												<Badge variant="outline">{speaker.enrollment_source}</Badge>
											</td>
											<td class="py-3 text-muted-foreground">{speaker.sample_count}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{/if}
				</Card.Content>
			</Card.Root>
		</div>
	</Tabs.Content>

	<!-- ════════════════════════════════════════════════════════════════ -->
	<!-- Tab 3: Rules                                                    -->
	<!-- ════════════════════════════════════════════════════════════════ -->
	<Tabs.Content value="rules">
		<div class="mt-4">
			<Card.Root>
				<Card.Header>
					<Card.Title>Auto-Trigger Rules</Card.Title>
					<Card.Description>
						Configure when offline diarization is automatically triggered for new meetings
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<div class="space-y-6 max-w-lg">
						<!-- Enabled toggle -->
						<div class="flex items-center justify-between rounded-lg border p-4">
							<div>
								<p class="text-sm font-medium">Auto-trigger enabled</p>
								<p class="text-xs text-muted-foreground mt-0.5">
									Automatically queue diarization when meetings match the rules below
								</p>
							</div>
							<button
								type="button"
								role="switch"
								aria-label="Auto-trigger enabled"
								aria-checked={rulesEnabled}
								onclick={() => {
									rulesEnabled = !rulesEnabled;
								}}
								class="relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background
									{rulesEnabled ? 'bg-primary' : 'bg-input'}"
							>
								<span
									class="pointer-events-none block size-5 rounded-full bg-background shadow-lg ring-0 transition-transform
										{rulesEnabled ? 'translate-x-5' : 'translate-x-0'}"
								></span>
							</button>
						</div>

						<!-- Participant patterns -->
						<div class="space-y-1.5">
							<Label for="participant-patterns">Participant Email Patterns</Label>
							<Input
								id="participant-patterns"
								placeholder="@company.com, specific@email.com"
								bind:value={participantPatterns}
								disabled={!rulesEnabled}
							/>
							<p class="text-xs text-muted-foreground">
								Comma-separated patterns. Meetings with matching participants will be queued.
							</p>
						</div>

						<!-- Title patterns -->
						<div class="space-y-1.5">
							<Label for="title-patterns">Meeting Title Patterns</Label>
							<Input
								id="title-patterns"
								placeholder="All Hands, Sprint Review"
								bind:value={titlePatterns}
								disabled={!rulesEnabled}
							/>
							<p class="text-xs text-muted-foreground">
								Comma-separated patterns matched against meeting titles.
							</p>
						</div>

						<!-- Min duration -->
						<div class="space-y-1.5">
							<Label for="min-duration">Minimum Duration (minutes)</Label>
							<Input
								id="min-duration"
								type="number"
								min={0}
								bind:value={minDurationMinutes}
								disabled={!rulesEnabled}
								class="max-w-32"
							/>
							<p class="text-xs text-muted-foreground">
								Skip meetings shorter than this duration.
							</p>
						</div>

						<!-- Exclude empty -->
						<div class="flex items-center gap-3">
							<input
								id="exclude-empty"
								type="checkbox"
								bind:checked={excludeEmpty}
								disabled={!rulesEnabled}
								class="size-4 rounded border-input accent-primary cursor-pointer"
							/>
							<div>
								<Label for="exclude-empty" class="cursor-pointer">Exclude empty transcripts</Label>
								<p class="text-xs text-muted-foreground">
									Skip meetings with no transcript content.
								</p>
							</div>
						</div>

						<!-- Save button -->
						<Button
							disabled={savingRules}
							onclick={saveRules}
						>
							{savingRules ? 'Saving...' : 'Save Rules'}
						</Button>
					</div>
				</Card.Content>
			</Card.Root>
		</div>
	</Tabs.Content>
</Tabs.Root>
