<script lang="ts">
	import { page } from '$app/stores';
	import { APP_NAME } from '$lib/config';
	import HomeIcon from '@lucide/svelte/icons/house';
	import MicIcon from '@lucide/svelte/icons/mic';
	import DatabaseIcon from '@lucide/svelte/icons/database';
	import LightbulbIcon from '@lucide/svelte/icons/lightbulb';
	import GlobeIcon from '@lucide/svelte/icons/globe';
	import SettingsIcon from '@lucide/svelte/icons/settings';
	import TerminalIcon from '@lucide/svelte/icons/terminal';
	import ChevronRightIcon from '@lucide/svelte/icons/chevron-right';
	import type { Component } from 'svelte';

	interface NavChild {
		label: string;
		href: string;
	}

	interface NavItem {
		label: string;
		href: string;
		icon: Component;
		children?: NavChild[];
	}

	const navItems: NavItem[] = [
		{ label: 'Dashboard', href: '/', icon: HomeIcon },
		{
			label: 'Fireflies',
			href: '/fireflies',
			icon: MicIcon,
			children: [
				{ label: 'Connect', href: '/fireflies' },
				{ label: 'Live Feed', href: '/fireflies/live-feed' },
				{ label: 'Sessions', href: '/fireflies/sessions' },
				{ label: 'History', href: '/fireflies/history' },
				{ label: 'Glossary', href: '/fireflies/glossary' }
			]
		},
		{ label: 'Data & Logs', href: '/data', icon: DatabaseIcon },
		{ label: 'Intelligence', href: '/intelligence', icon: LightbulbIcon },
		{
			label: 'Translation',
			href: '/translation',
			icon: GlobeIcon,
			children: [
				{ label: 'Test Bench', href: '/translation/test' },
				{ label: 'Config', href: '/config/translation' }
			]
		},
		{
			label: 'Config',
			href: '/config',
			icon: SettingsIcon,
			children: [
				{ label: 'Audio', href: '/config/audio' },
				{ label: 'System', href: '/config/system' },
				{ label: 'Settings', href: '/config/settings' }
			]
		},
		{ label: 'Session Manager', href: '/sessions', icon: TerminalIcon }
	];

	function isActive(href: string): boolean {
		const pathname = $page.url.pathname;
		if (href === '/') return pathname === '/';
		return pathname === href || pathname.startsWith(href + '/');
	}

	function isParentActive(item: NavItem): boolean {
		if (isActive(item.href)) return true;
		if (item.children) {
			return item.children.some((child) => isActive(child.href));
		}
		return false;
	}

	function isChildExact(href: string): boolean {
		return $page.url.pathname === href;
	}
</script>

<aside class="w-56 border-r bg-card flex flex-col h-full">
	<div class="p-4 border-b">
		<h1 class="text-lg font-semibold">{APP_NAME}</h1>
	</div>
	<nav class="flex-1 p-2 space-y-1 overflow-y-auto">
		{#each navItems as item}
			{@const parentActive = isParentActive(item)}
			<a
				href={item.children ? item.children[0].href : item.href}
				aria-current={parentActive ? 'page' : undefined}
				class="flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors
					{parentActive
					? 'bg-accent text-accent-foreground font-medium'
					: 'text-muted-foreground hover:bg-accent/50'}"
			>
				<svelte:component this={item.icon} class="size-4 shrink-0" />
				<span class="truncate">{item.label}</span>
				{#if item.children}
					<ChevronRightIcon
						class="size-3 ml-auto shrink-0 transition-transform {parentActive
							? 'rotate-90'
							: ''}"
					/>
				{/if}
			</a>
			{#if item.children && parentActive}
				<div class="ml-8 space-y-0.5">
					{#each item.children as child}
						<a
							href={child.href}
							class="block px-3 py-1.5 rounded text-xs transition-colors
								{isChildExact(child.href)
								? 'text-foreground font-medium'
								: 'text-muted-foreground hover:text-foreground'}"
						>
							{child.label}
						</a>
					{/each}
				</div>
			{/if}
		{/each}
	</nav>
	<div class="p-3 border-t">
		<p class="text-[10px] text-muted-foreground/60 text-center">SvelteKit Dashboard v2.0.0</p>
	</div>
</aside>
