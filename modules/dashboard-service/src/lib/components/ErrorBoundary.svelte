<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { toastStore } from '$lib/stores/toast.svelte';

	let { children } = $props();

	let error = $state<Error | null>(null);

	function handleError(e: unknown) {
		error = e instanceof Error ? e : new Error(String(e));
		toastStore.error(`Something went wrong: ${error.message}`);
	}

	function retry() {
		error = null;
	}
</script>

{#if error}
	<div class="flex flex-col items-center justify-center gap-4 py-12 text-center" role="alert">
		<div class="text-4xl">⚠</div>
		<h3 class="text-lg font-semibold">Something went wrong</h3>
		<p class="text-sm text-muted-foreground max-w-md">{error.message}</p>
		<Button variant="outline" onclick={retry}>Try again</Button>
	</div>
{:else}
	<svelte:boundary onerror={handleError}>
		{@render children()}
	</svelte:boundary>
{/if}
