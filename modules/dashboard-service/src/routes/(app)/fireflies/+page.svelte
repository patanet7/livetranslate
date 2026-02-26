<script lang="ts">
	import { enhance } from '$app/forms';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';

	let { data, form } = $props();
</script>

<PageHeader
	title="Fireflies"
	description="Connect to a live Fireflies transcript for real-time translation"
/>

<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
	<!-- Connect Form -->
	<div class="lg:col-span-2">
		<Card.Root>
			<Card.Header>
				<Card.Title>Connect to Transcript</Card.Title>
			</Card.Header>
			<Card.Content>
				<form method="POST" action="?/connect" use:enhance class="space-y-4">
					<div class="space-y-2">
						<Label for="transcript_id">Transcript ID</Label>
						<Input
							id="transcript_id"
							name="transcript_id"
							placeholder="Enter Fireflies transcript ID"
							value={form?.transcript_id ?? ''}
							required
						/>
						{#if form?.errors?.transcript_id}
							<p class="text-sm text-destructive">{form.errors.transcript_id}</p>
						{/if}
					</div>

					<div class="space-y-2">
						<Label for="api_key">API Key (optional)</Label>
						<Input
							id="api_key"
							name="api_key"
							type="password"
							placeholder="Uses env default if blank"
						/>
					</div>

					<div class="space-y-2">
						<Label for="target_languages">Target Languages (comma-separated)</Label>
						<Input
							id="target_languages"
							name="target_languages"
							placeholder="es,fr,de"
						/>
					</div>

					<div class="space-y-2">
						<Label for="domain">Domain</Label>
						<select
							id="domain"
							name="domain"
							class="w-full rounded-md border bg-background px-3 py-2 text-sm"
						>
							<option value="">General</option>
							{#if data.uiConfig?.domains}
								{#each data.uiConfig.domains as d}
									<option value={d}>{d}</option>
								{/each}
							{/if}
						</select>
					</div>

					{#if form?.errors?.form}
						<p class="text-sm text-destructive">{form.errors.form}</p>
					{/if}

					<Button type="submit">Connect</Button>
				</form>
			</Card.Content>
		</Card.Root>
	</div>

	<!-- Active Sessions -->
	<div>
		<Card.Root>
			<Card.Header>
				<Card.Title>Active Sessions</Card.Title>
			</Card.Header>
			<Card.Content>
				{#if data.sessions.length === 0}
					<p class="text-sm text-muted-foreground">No active sessions</p>
				{:else}
					<ul class="space-y-2">
						{#each data.sessions as session}
							<li>
								<a
									href="/fireflies/connect?session={session.session_id}"
									class="block p-2 rounded border hover:bg-accent transition-colors"
								>
									<div class="flex items-center justify-between">
										<span class="text-sm font-mono truncate"
											>{session.session_id.slice(0, 16)}...</span
										>
										<StatusIndicator
											status={session.connection_status === 'CONNECTED'
												? 'connected'
												: 'disconnected'}
										/>
									</div>
									<p class="text-xs text-muted-foreground mt-1">
										{session.chunks_received} chunks · {session.translations_completed} translations
									</p>
								</a>
							</li>
						{/each}
					</ul>
				{/if}
			</Card.Content>
		</Card.Root>
	</div>
</div>
