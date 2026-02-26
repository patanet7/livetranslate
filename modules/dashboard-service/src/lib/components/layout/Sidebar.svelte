<script lang="ts">
	import { page } from '$app/stores';
	import { APP_NAME } from '$lib/config';

	const navItems = [
		{ label: 'Dashboard', href: '/', icon: '⌂' },
		{
			label: 'Fireflies',
			href: '/fireflies',
			icon: '🎙',
			children: [
				{ label: 'Connect', href: '/fireflies' },
				{ label: 'History', href: '/fireflies/history' },
				{ label: 'Glossary', href: '/fireflies/glossary' }
			]
		},
		{
			label: 'Config',
			href: '/config',
			icon: '⚙',
			children: [
				{ label: 'Audio', href: '/config/audio' },
				{ label: 'Translation', href: '/config/translation' },
				{ label: 'System', href: '/config/system' }
			]
		},
		{ label: 'Translation', href: '/translation/test', icon: '🌐' }
	];

	function isActive(href: string): boolean {
		if (href === '/') return $page.url.pathname === '/';
		return $page.url.pathname.startsWith(href);
	}
</script>

<aside class="w-56 border-r bg-card flex flex-col h-full">
	<div class="p-4 border-b">
		<h1 class="text-lg font-semibold">{APP_NAME}</h1>
	</div>
	<nav class="flex-1 p-2 space-y-1 overflow-y-auto">
		{#each navItems as item}
			<a
				href={item.children ? item.children[0].href : item.href}
				aria-current={isActive(item.href) ? 'page' : undefined}
				class="flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors
					{isActive(item.href)
					? 'bg-accent text-accent-foreground font-medium'
					: 'text-muted-foreground hover:bg-accent/50'}"
			>
				<span>{item.icon}</span>
				<span>{item.label}</span>
			</a>
			{#if item.children && isActive(item.href)}
				<div class="ml-8 space-y-0.5">
					{#each item.children as child}
						<a
							href={child.href}
							class="block px-3 py-1.5 rounded text-xs transition-colors
								{$page.url.pathname === child.href
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
</aside>
