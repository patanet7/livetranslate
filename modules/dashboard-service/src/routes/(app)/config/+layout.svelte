<script lang="ts">
	import { page } from '$app/stores';
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();

	const tabs = [
		{ href: '/config/audio', label: 'Audio' },
		{ href: '/config/translation', label: 'Translation' },
		{ href: '/config/system', label: 'System' },
		{ href: '/config/connections', label: 'Connections' },
		{ href: '/config/settings', label: 'Settings' },
	];

	function isActive(href: string): boolean {
		return $page.url.pathname === href || $page.url.pathname.startsWith(href + '/');
	}
</script>

{#if $page.url.pathname !== '/config'}
	<nav class="config-tabs" aria-label="Configuration sections">
		{#each tabs as tab (tab.href)}
			<a
				href={tab.href}
				class="config-tab"
				class:active={isActive(tab.href)}
				aria-current={isActive(tab.href) ? 'page' : undefined}
			>
				{tab.label}
			</a>
		{/each}
	</nav>
{/if}

{@render children()}

<style>
	.config-tabs {
		display: flex;
		gap: 0;
		border-bottom: 1px solid var(--border, #333);
		padding: 0 1rem;
		margin-bottom: 1.5rem;
	}

	.config-tab {
		padding: 0.5rem 1rem;
		font-size: 0.8125rem;
		color: var(--text-muted, #94a3b8);
		text-decoration: none;
		border-bottom: 2px solid transparent;
		transition: color 0.15s, border-color 0.15s;
	}

	.config-tab:hover {
		color: var(--text, #e2e8f0);
	}

	.config-tab.active {
		color: var(--primary, #3b82f6);
		border-bottom-color: var(--primary, #3b82f6);
		font-weight: 500;
	}
</style>
