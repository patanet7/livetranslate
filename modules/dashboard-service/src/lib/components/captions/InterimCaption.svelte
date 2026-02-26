<script lang="ts">
	interface Props {
		text: string;
	}

	let { text }: Props = $props();
</script>

{#if text}
	<div class="interim-caption border border-dashed rounded-lg p-3 opacity-70">
		<p class="text-sm italic">{text}</p>
	</div>
{/if}
