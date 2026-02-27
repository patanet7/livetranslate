<script lang="ts">
	import type { Caption } from '$lib/types';
	import CaptionBox from './CaptionBox.svelte';
	import InterimCaption from './InterimCaption.svelte';

	interface Props {
		captions: (Caption & { receivedAt: number })[];
		interim: string;
		showOriginal?: boolean;
		showTranslated?: boolean;
	}

	let { captions, interim, showOriginal = true, showTranslated = true }: Props = $props();

	let container: HTMLDivElement;

	$effect(() => {
		// Auto-scroll to bottom when new captions arrive
		if (captions.length && container) {
			container.scrollTop = container.scrollHeight;
		}
	});
</script>

<div bind:this={container} class="caption-stream space-y-2 max-h-[70vh] overflow-y-auto">
	{#each captions as caption (caption.id)}
		<CaptionBox {caption} {showOriginal} {showTranslated} />
	{/each}
	<InterimCaption text={interim} />
</div>
