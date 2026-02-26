<script lang="ts">
	import StatusIndicator from './StatusIndicator.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		health: 'healthy' | 'degraded' | 'down' | 'unknown';
	}

	let { health }: Props = $props();

	// Derive connection statuses from overall health
	let apiConnected = $derived(health === 'healthy' || health === 'degraded');
	let translationOnline = $derived(health === 'healthy');

	// Demo controls state
	let demoOpen = $state(false);
	let demoMode = $state<'live' | 'pretranslated'>('live');
	let demoRunning = $state(false);

	const demoModes = [
		{ value: 'live' as const, label: 'Live Passthrough' },
		{ value: 'pretranslated' as const, label: 'Pre-translated ES' }
	];

	function toggleDemo() {
		demoRunning = !demoRunning;
		// Actual demo API integration will come in Batch 3 Task 3.6
	}

	function selectMode(mode: 'live' | 'pretranslated') {
		demoMode = mode;
		demoOpen = false;
	}

	function handleClickOutside(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if (!target.closest('[data-demo-menu]')) {
			demoOpen = false;
		}
	}
</script>

<svelte:document onclick={demoOpen ? handleClickOutside : undefined} />

<header class="h-12 border-b bg-card flex items-center justify-between px-4">
	<!-- Left: App name / breadcrumb -->
	<div class="flex items-center gap-2">
		<span class="text-sm font-semibold text-foreground">LiveTranslate</span>
		<span class="text-xs text-muted-foreground">/</span>
		<span class="text-xs text-muted-foreground">Dashboard</span>
	</div>

	<!-- Right: Status badges, demo controls, services indicator -->
	<div class="flex items-center gap-3">
		<!-- API status badge -->
		<Badge variant="outline" class="gap-1.5 text-xs font-normal">
			<span
				class="h-1.5 w-1.5 rounded-full {apiConnected ? 'bg-green-500' : 'bg-red-500'}"
			></span>
			{apiConnected ? 'API Connected' : 'API Disconnected'}
		</Badge>

		<!-- Translation status badge -->
		<Badge variant="outline" class="gap-1.5 text-xs font-normal">
			<span
				class="h-1.5 w-1.5 rounded-full {translationOnline ? 'bg-green-500' : 'bg-red-500'}"
			></span>
			{translationOnline ? 'Translation' : 'Translation Offline'}
		</Badge>

		<!-- Separator -->
		<div class="h-5 w-px bg-border"></div>

		<!-- Demo controls -->
		<div class="relative" data-demo-menu>
			<Button
				variant="outline"
				size="sm"
				class="h-7 gap-1.5 text-xs"
				onclick={() => (demoOpen = !demoOpen)}
			>
				<span
					class="h-1.5 w-1.5 rounded-full {demoRunning
						? 'bg-green-500 animate-pulse'
						: 'bg-muted-foreground'}"
				></span>
				Demo
				<svg
					class="h-3 w-3 text-muted-foreground transition-transform {demoOpen
						? 'rotate-180'
						: ''}"
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 20 20"
					fill="currentColor"
				>
					<path
						fill-rule="evenodd"
						d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"
						clip-rule="evenodd"
					/>
				</svg>
			</Button>

			{#if demoOpen}
				<div
					class="absolute right-0 top-full mt-1 z-50 w-56 rounded-md border bg-popover p-1 shadow-md"
				>
					<div class="px-2 py-1.5 text-xs font-medium text-muted-foreground">
						Demo Mode
					</div>
					{#each demoModes as mode}
						<button
							class="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-xs transition-colors hover:bg-accent hover:text-accent-foreground {demoMode ===
							mode.value
								? 'bg-accent/50 text-accent-foreground'
								: 'text-foreground'}"
							onclick={() => selectMode(mode.value)}
						>
							<span
								class="h-1.5 w-1.5 rounded-full {demoMode === mode.value
									? 'bg-primary'
									: 'bg-transparent'}"
							></span>
							{mode.label}
						</button>
					{/each}
					<div class="my-1 h-px bg-border"></div>
					<button
						class="flex w-full items-center justify-center rounded-sm px-2 py-1.5 text-xs font-medium transition-colors {demoRunning
							? 'bg-red-500/10 text-red-400 hover:bg-red-500/20'
							: 'bg-green-500/10 text-green-400 hover:bg-green-500/20'}"
						onclick={toggleDemo}
					>
						{demoRunning ? 'Stop Demo' : 'Launch Demo'}
					</button>
				</div>
			{/if}
		</div>

		<!-- Separator -->
		<div class="h-5 w-px bg-border"></div>

		<!-- Services overall status -->
		<StatusIndicator status={health} label="Services" />
	</div>
</header>
