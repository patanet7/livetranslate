<script lang="ts">
	import type { Caption } from '$lib/types';

	interface Props {
		caption: Caption & { receivedAt: number };
		showOriginal?: boolean;
		showTranslated?: boolean;
	}

	let { caption, showOriginal = true, showTranslated = true }: Props = $props();
</script>

<div
	class="caption-box border rounded-lg p-3 space-y-1 transition-opacity"
	data-caption-id={caption.id}
>
	<div class="flex items-center gap-2">
		<span
			class="speaker-name text-xs font-medium px-1.5 py-0.5 rounded"
			style="background-color: {caption.speaker_color}20; color: {caption.speaker_color}"
		>
			{caption.speaker_name}
		</span>
		<span class="text-xs text-muted-foreground">{caption.target_language}</span>
		{#if caption.confidence}
			<span class="text-xs text-muted-foreground ml-auto"
				>{Math.round(caption.confidence * 100)}%</span
			>
		{/if}
	</div>
	{#if showOriginal && caption.original_text}
		<p class="original-text text-sm text-muted-foreground">{caption.original_text}</p>
	{/if}
	{#if showTranslated && caption.text}
		<p class="translated-text text-sm font-medium">{caption.text}</p>
	{/if}
</div>
