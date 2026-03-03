<script lang="ts">
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import CaptionStream from '$lib/components/captions/CaptionStream.svelte';
	import ConnectionBanner from '$lib/components/meetings/ConnectionBanner.svelte';
	import { wsStore } from '$lib/stores/websocket.svelte';
	import { captionStore } from '$lib/stores/captions.svelte';
	import { toastStore } from '$lib/stores/toast.svelte';
	import { WS_BASE } from '$lib/config';
	import type { CaptionEvent } from '$lib/types';

	let { data } = $props();

	const meeting = $derived(data.meeting);
	const sessionId = $derived(data.sessionId ?? data.session?.session_id);

	let disconnecting = $state(false);

	// Connect WebSocket on mount
	$effect(() => {
		if (!browser || !sessionId) return;

		captionStore.start();
		const wsUrl = `${WS_BASE}/api/captions/stream/${sessionId}`;
		wsStore.onMessage = handleWsMessage;
		wsStore.connect(wsUrl);

		return () => {
			wsStore.disconnect();
			captionStore.stop();
			captionStore.clear();
		};
	});

	function handleWsMessage(event: MessageEvent) {
		try {
			const msg: CaptionEvent = JSON.parse(event.data);

			switch (msg.event) {
				case 'connected':
					if (msg.current_captions) {
						for (const c of msg.current_captions) {
							captionStore.addCaption({ ...c, receivedAt: Date.now() });
						}
					}
					break;
				case 'caption_added':
					captionStore.addCaption({ ...msg.caption, receivedAt: Date.now() });
					break;
				case 'caption_updated':
					captionStore.updateCaption(msg.caption);
					break;
				case 'caption_expired':
					captionStore.removeCaption(msg.caption_id);
					break;
				case 'interim_caption':
					captionStore.updateInterim(msg.caption?.text ?? '');
					break;
				case 'session_cleared':
					captionStore.clear();
					break;
			}
		} catch {
			// Ignore malformed messages
		}
	}

	async function handleDisconnect() {
		if (!sessionId) return;
		disconnecting = true;
		try {
			const res = await fetch(`/api/fireflies/disconnect`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ session_id: sessionId })
			});
			if (!res.ok) {
				const errorData = await res.json().catch(() => null);
				toastStore.error(errorData?.detail ?? errorData?.message ?? `Disconnect failed (${res.status})`);
				return;
			}
			wsStore.disconnect();
			toastStore.success('Session disconnected');
			goto(`/meetings/${meeting.id}`);
		} catch {
			toastStore.error('Failed to disconnect');
		} finally {
			disconnecting = false;
		}
	}
</script>

<!-- Connection Banner -->
<ConnectionBanner
	status={wsStore.status}
	reconnectAttempt={wsStore.reconnectAttempt}
	onretry={() => wsStore.retry()}
/>

<div class="mt-4 grid grid-cols-1 lg:grid-cols-4 gap-6">
	<!-- Live Captions -->
	<div class="lg:col-span-3">
		<Card.Root>
			<Card.Header>
				<div class="flex items-center justify-between">
					<Card.Title>Live Captions</Card.Title>
					<Badge variant="outline">
						{captionStore.captions.length} captions
					</Badge>
				</div>
			</Card.Header>
			<Card.Content>
				{#if captionStore.captions.length === 0 && !captionStore.interim}
					<div class="py-12 text-center text-muted-foreground">
						<p>Waiting for captions...</p>
						<p class="text-xs mt-1">Captions will appear here as the meeting progresses.</p>
					</div>
				{:else}
					<CaptionStream captions={captionStore.captions} interim={captionStore.interim} />
				{/if}
			</Card.Content>
		</Card.Root>
	</div>

	<!-- Session Info Sidebar -->
	<div class="space-y-4">
		<Card.Root>
			<Card.Header>
				<Card.Title>Session Info</Card.Title>
			</Card.Header>
			<Card.Content class="space-y-3 text-sm">
				{#if data.session}
					<div class="flex justify-between">
						<span class="text-muted-foreground">Status</span>
						<Badge variant={data.session.connection_status === 'CONNECTED' ? 'default' : 'secondary'}>
							{data.session.connection_status}
						</Badge>
					</div>
					<div class="flex justify-between">
						<span class="text-muted-foreground">Chunks</span>
						<span>{data.session.chunks_received}</span>
					</div>
					<div class="flex justify-between">
						<span class="text-muted-foreground">Translations</span>
						<span>{data.session.translations_completed}</span>
					</div>
					{#if data.session.speakers_detected?.length}
						<div>
							<span class="text-muted-foreground">Speakers</span>
							<div class="flex flex-wrap gap-1 mt-1">
								{#each data.session.speakers_detected as speaker}
									<Badge variant="outline" class="text-xs">{speaker}</Badge>
								{/each}
							</div>
						</div>
					{/if}
				{:else}
					<p class="text-muted-foreground">No session data available</p>
				{/if}
			</Card.Content>
		</Card.Root>

		<Button
			variant="destructive"
			class="w-full"
			onclick={handleDisconnect}
			disabled={disconnecting}
		>
			{disconnecting ? 'Disconnecting...' : 'Disconnect'}
		</Button>
	</div>
</div>
