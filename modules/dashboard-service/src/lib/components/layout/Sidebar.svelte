<script lang="ts">
	import { page } from '$app/stores';
	import { APP_NAME } from '$lib/config';
	import HomeIcon from '@lucide/svelte/icons/house';
	import CalendarDaysIcon from '@lucide/svelte/icons/calendar-days';
	import MicIcon from '@lucide/svelte/icons/mic';
	import DatabaseIcon from '@lucide/svelte/icons/database';
	import LightbulbIcon from '@lucide/svelte/icons/lightbulb';
	import AudioWaveformIcon from '@lucide/svelte/icons/audio-waveform';
	import HeadphonesIcon from '@lucide/svelte/icons/headphones';
	import GlobeIcon from '@lucide/svelte/icons/globe';
	import SettingsIcon from '@lucide/svelte/icons/settings';
	import MessageSquareIcon from '@lucide/svelte/icons/message-square';
	import TerminalIcon from '@lucide/svelte/icons/terminal';
	import ChevronRightIcon from '@lucide/svelte/icons/chevron-right';
	import type { Component } from 'svelte';

	interface Props {
		open?: boolean;
		onclose?: () => void;
	}

	let { open = false, onclose }: Props = $props();

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
		{ label: 'Meetings', href: '/meetings', icon: CalendarDaysIcon },
		{ label: 'Loopback', href: '/loopback', icon: HeadphonesIcon },
		{
			label: 'Fireflies',
			href: '/fireflies',
			icon: MicIcon,
			children: [
				{ label: 'Connect', href: '/fireflies' },
				{ label: 'Live Feed', href: '/fireflies/live-feed' },
				{ label: 'Glossary', href: '/fireflies/glossary' }
			]
		},
		{ label: 'Data & Logs', href: '/data', icon: DatabaseIcon },
		{ label: 'Intelligence', href: '/intelligence', icon: LightbulbIcon },
		{ label: 'Diarization', href: '/diarization', icon: AudioWaveformIcon },
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
				{ label: 'Connections', href: '/config/connections' },
				{ label: 'Audio', href: '/config/audio' },
				{ label: 'System', href: '/config/system' },
				{ label: 'Settings', href: '/config/settings' }
			]
		},
		{ label: 'Chat', href: '/chat', icon: MessageSquareIcon },
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

	function handleNavClick() {
		onclose?.();
	}
</script>

<!-- Mobile backdrop -->
{#if open}
	<button
		class="fixed inset-0 z-40 bg-black/50 md:hidden"
		aria-label="Close sidebar"
		onclick={onclose}
		tabindex="-1"
	></button>
{/if}

<aside
	class="w-56 border-r bg-card flex flex-col h-full
		fixed inset-y-0 left-0 z-50 transition-transform duration-200 ease-in-out
		md:relative md:translate-x-0 md:z-auto
		{open ? 'translate-x-0' : '-translate-x-full'}"
>
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
				onclick={handleNavClick}
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
							onclick={handleNavClick}
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
