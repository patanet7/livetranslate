<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
	import CaptionStream from '$lib/components/captions/CaptionStream.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { wsStore } from '$lib/stores/websocket.svelte';
	import { captionStore } from '$lib/stores/captions.svelte';
	import { WS_BASE } from '$lib/config';
	import type { CaptionEvent } from '$lib/types';

	let { data } = $props();

	onMount(() => {
		const sessionId = data.session.session_id;
		wsStore.connect(`${WS_BASE}/api/captions/stream/${sessionId}`);

		wsStore.onMessage = (event) => {
			const msg: CaptionEvent = JSON.parse(event.data);
			switch (msg.event) {
				case 'connected':
					msg.current_captions.forEach((c) => captionStore.addCaption(c));
					break;
				case 'caption_added':
					captionStore.addCaption(msg.caption);
					break;
				case 'caption_updated':
					captionStore.updateCaption(msg.caption);
					break;
				case 'caption_expired':
					captionStore.removeCaption(msg.caption_id);
					break;
				case 'interim_caption':
					captionStore.updateInterim(msg.caption.text);
					break;
				case 'session_cleared':
					captionStore.clear();
					break;
			}
		};

		captionStore.start();

		return () => {
			wsStore.disconnect();
			captionStore.stop();
			captionStore.clear();
		};
	});

	async function handleDisconnect() {
		await fetch(`/api/fireflies/disconnect`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ session_id: data.session.session_id })
		});
		goto('/fireflies');
	}
</script>

<PageHeader title="Live Session">
	{#snippet actions()}
		<StatusIndicator status={wsStore.status} label={wsStore.status} />
		<Button variant="destructive" size="sm" onclick={handleDisconnect}>Disconnect</Button>
	{/snippet}
</PageHeader>

<div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
	<div class="lg:col-span-3">
		<Card.Root>
			<Card.Header>
				<Card.Title>Live Captions</Card.Title>
			</Card.Header>
			<Card.Content>
				<CaptionStream captions={captionStore.captions} interim={captionStore.interim} />
			</Card.Content>
		</Card.Root>
	</div>

	<div>
		<Card.Root>
			<Card.Header>
				<Card.Title>Session Info</Card.Title>
			</Card.Header>
			<Card.Content class="space-y-2 text-sm">
				<div class="flex justify-between">
					<span class="text-muted-foreground">Status</span>
					<StatusIndicator
						status={data.session.connection_status === 'CONNECTED'
							? 'connected'
							: 'disconnected'}
						label={data.session.connection_status}
					/>
				</div>
				<div class="flex justify-between">
					<span class="text-muted-foreground">Chunks</span>
					<span>{data.session.chunks_received}</span>
				</div>
				<div class="flex justify-between">
					<span class="text-muted-foreground">Translations</span>
					<span>{data.session.translations_completed}</span>
				</div>
				<div class="flex justify-between">
					<span class="text-muted-foreground">Speakers</span>
					<span>{data.session.speakers_detected.length}</span>
				</div>
				{#if data.session.speakers_detected.length > 0}
					<div class="pt-2 border-t">
						<p class="text-xs text-muted-foreground mb-1">Detected speakers:</p>
						{#each data.session.speakers_detected as speaker}
							<span class="text-xs bg-accent px-1.5 py-0.5 rounded mr-1">{speaker}</span>
						{/each}
					</div>
				{/if}
			</Card.Content>
		</Card.Root>
	</div>
</div>
