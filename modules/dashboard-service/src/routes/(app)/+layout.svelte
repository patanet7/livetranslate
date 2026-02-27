<script lang="ts">
	import { onMount } from 'svelte';
	import { healthStore } from '$lib/stores/health.svelte';
	import { demoStore } from '$lib/stores/demo.svelte';
	import Sidebar from '$lib/components/layout/Sidebar.svelte';
	import TopBar from '$lib/components/layout/TopBar.svelte';

	let { children } = $props();

	let sidebarOpen = $state(false);

	onMount(() => {
		healthStore.startPolling();
		return () => healthStore.stopPolling();
	});

	$effect(() => {
		demoStore.checkStatus();
	});
</script>

<div class="flex h-screen bg-background text-foreground">
	<Sidebar open={sidebarOpen} onclose={() => (sidebarOpen = false)} />
	<div class="flex flex-col flex-1 overflow-hidden">
		<TopBar health={healthStore.status} onMenuToggle={() => (sidebarOpen = !sidebarOpen)} />
		<main class="flex-1 overflow-y-auto p-4 md:p-6">
			{@render children()}
		</main>
	</div>
</div>
