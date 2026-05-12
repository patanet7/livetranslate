<script lang="ts">
	import type { Caption } from '$lib/types';
	import type { Attachment } from 'svelte/attachments';
	import CaptionBox from './CaptionBox.svelte';
	import InterimCaption from './InterimCaption.svelte';

	interface Props {
		captions: (Caption & { receivedAt: number })[];
		interim: string;
		showOriginal?: boolean;
		showTranslated?: boolean;
	}

	let { captions, interim, showOriginal = true, showTranslated = true }: Props = $props();

	/**
	 * Auto-scroll the container to the bottom whenever the caption count grows.
	 * Uses an attachment (Svelte 5 idiom) to read `captions.length` reactively
	 * without `bind:this` + a separate `$effect`.
	 */
	const autoScrollOnGrow: Attachment<HTMLDivElement> = (node) => {
		let prev = 0;
		$effect(() => {
			const c = captions.length;
			if (c > prev) {
				prev = c;
				node.scrollTop = node.scrollHeight;
			}
		});
	};
</script>

<div {@attach autoScrollOnGrow} class="caption-stream space-y-2 max-h-[70vh] overflow-y-auto" role="log" aria-live="polite" aria-label="Live captions">
	{#each captions as caption (caption.id)}
		<CaptionBox {caption} {showOriginal} {showTranslated} />
	{/each}
	<InterimCaption text={interim} />
</div>
