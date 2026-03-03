<script lang="ts">
	interface Props {
		rows?: number;
	}

	let { rows = 5 }: Props = $props();
</script>

<div class="space-y-3" role="status" aria-busy="true" aria-label="Loading meetings">
	{#each Array(rows) as _, i (i)}
		<div class="flex items-center gap-4 rounded-lg border p-4 animate-pulse">
			<div class="flex-1 space-y-2">
				<div class="h-4 rounded bg-muted" style="width: {60 + Math.random() * 30}%"></div>
				<div class="h-3 rounded bg-muted" style="width: {30 + Math.random() * 20}%"></div>
			</div>
			<div class="h-6 w-16 rounded-full bg-muted"></div>
		</div>
	{/each}
</div>
