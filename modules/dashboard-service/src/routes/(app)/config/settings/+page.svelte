<script lang="ts">
	import { browser } from '$app/environment';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button, buttonVariants } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Badge } from '$lib/components/ui/badge';
	import * as Dialog from '$lib/components/ui/dialog';
	import { toastStore } from '$lib/stores/toast.svelte';
	import EyeIcon from '@lucide/svelte/icons/eye';
	import EyeOffIcon from '@lucide/svelte/icons/eye-off';
	import KeyIcon from '@lucide/svelte/icons/key-round';
	import Loader2Icon from '@lucide/svelte/icons/loader-circle';
	import CheckCircleIcon from '@lucide/svelte/icons/circle-check';
	import TrashIcon from '@lucide/svelte/icons/trash-2';
	import WifiIcon from '@lucide/svelte/icons/wifi';

	const STORAGE_KEY = 'fireflies_api_key';

	let apiKey = $state('');
	let savedKey = $state('');
	let showKey = $state(false);
	let saving = $state(false);
	let testing = $state(false);
	let clearDialogOpen = $state(false);
	let statusMessage = $state('');
	let statusType = $state<'success' | 'error' | ''>('');
	let meetingCount = $state<number | null>(null);

	let maskedKey = $derived(
		savedKey.length > 4
			? '****...' + savedKey.slice(-4)
			: savedKey.length > 0
				? '****'
				: ''
	);

	let hasSavedKey = $derived(savedKey.length > 0);
	let canSave = $derived(apiKey.trim().length > 0 && !saving);
	let canTest = $derived(hasSavedKey && !testing);

	$effect(() => {
		if (browser) {
			savedKey = localStorage.getItem(STORAGE_KEY) ?? '';
		}
	});

	function clearStatus() {
		statusMessage = '';
		statusType = '';
		meetingCount = null;
	}

	async function validateKey(key: string): Promise<{ valid: boolean; meeting_count?: number; error?: string }> {
		const res = await fetch('/api/fireflies/validate-key', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ api_key: key })
		});

		if (!res.ok) {
			const text = await res.text().catch(() => 'Request failed');
			return { valid: false, error: text };
		}

		return res.json();
	}

	async function handleSave() {
		if (!canSave) return;
		clearStatus();
		saving = true;

		try {
			const result = await validateKey(apiKey.trim());

			if (result.valid) {
				localStorage.setItem(STORAGE_KEY, apiKey.trim());
				savedKey = apiKey.trim();
				apiKey = '';
				showKey = false;
				statusMessage = 'API key saved and validated successfully';
				statusType = 'success';
				meetingCount = result.meeting_count ?? null;
				toastStore.success('Fireflies API key saved');
			} else {
				statusMessage = result.error ?? 'API key validation failed';
				statusType = 'error';
				toastStore.error('API key validation failed');
			}
		} catch {
			statusMessage = 'Network error: could not reach validation endpoint';
			statusType = 'error';
			toastStore.error('Network error during validation');
		} finally {
			saving = false;
		}
	}

	async function handleTest() {
		if (!canTest) return;
		clearStatus();
		testing = true;

		try {
			const result = await validateKey(savedKey);

			if (result.valid) {
				meetingCount = result.meeting_count ?? 0;
				statusMessage = `Connection successful - ${meetingCount} meeting${meetingCount === 1 ? '' : 's'} found`;
				statusType = 'success';
				toastStore.success(`Connection verified: ${meetingCount} meeting${meetingCount === 1 ? '' : 's'} found`);
			} else {
				statusMessage = result.error ?? 'Connection test failed';
				statusType = 'error';
				meetingCount = null;
				toastStore.error('Connection test failed');
			}
		} catch {
			statusMessage = 'Network error: could not reach validation endpoint';
			statusType = 'error';
			meetingCount = null;
			toastStore.error('Network error during connection test');
		} finally {
			testing = false;
		}
	}

	function handleClear() {
		localStorage.removeItem(STORAGE_KEY);
		savedKey = '';
		apiKey = '';
		showKey = false;
		clearDialogOpen = false;
		clearStatus();
		toastStore.info('API key removed');
	}
</script>

<PageHeader title="Settings" description="Manage API keys and service integrations" />

<div class="space-y-6 max-w-2xl">
	<!-- Saved Key Status -->
	<Card.Root>
		<Card.Header>
			<div class="flex items-center gap-2">
				<KeyIcon class="size-5 text-muted-foreground" />
				<Card.Title>Fireflies API Key</Card.Title>
			</div>
			<Card.Description>
				Your Fireflies API key is stored locally in your browser. It is never sent to our servers.
			</Card.Description>
		</Card.Header>
		<Card.Content class="space-y-4">
			<!-- Current Key Status -->
			<div class="flex items-center gap-3 p-3 rounded-md bg-muted/50 border">
				{#if hasSavedKey}
					<CheckCircleIcon class="size-4 text-green-500 shrink-0" />
					<span class="text-sm font-mono">{maskedKey}</span>
					<Badge variant="secondary" class="bg-green-500/10 text-green-500 border-green-500/20">
						Saved
					</Badge>
				{:else}
					<KeyIcon class="size-4 text-muted-foreground shrink-0" />
					<span class="text-sm text-muted-foreground">No API key saved</span>
				{/if}
			</div>

			<!-- Input Section -->
			<div class="space-y-2">
				<Label for="api-key">{hasSavedKey ? 'Replace API Key' : 'Enter API Key'}</Label>
				<div class="flex gap-2">
					<div class="relative flex-1">
						<Input
							id="api-key"
							type={showKey ? 'text' : 'password'}
							placeholder="Enter your Fireflies API key"
							bind:value={apiKey}
							onkeydown={(e: KeyboardEvent) => { if (e.key === 'Enter') handleSave(); }}
						/>
						<button
							type="button"
							class="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors p-1"
							onclick={() => (showKey = !showKey)}
							aria-label={showKey ? 'Hide API key' : 'Show API key'}
						>
							{#if showKey}
								<EyeOffIcon class="size-4" />
							{:else}
								<EyeIcon class="size-4" />
							{/if}
						</button>
					</div>
					<Button onclick={handleSave} disabled={!canSave}>
						{#if saving}
							<Loader2Icon class="size-4 animate-spin" />
							Validating...
						{:else}
							Save API Key
						{/if}
					</Button>
				</div>
			</div>

			<!-- Action Buttons -->
			{#if hasSavedKey}
				<div class="flex gap-2 pt-2 border-t">
					<Button variant="outline" onclick={handleTest} disabled={!canTest}>
						{#if testing}
							<Loader2Icon class="size-4 animate-spin" />
							Testing...
						{:else}
							<WifiIcon class="size-4" />
							Test Connection
						{/if}
					</Button>

					<Dialog.Dialog bind:open={clearDialogOpen}>
						<Dialog.DialogTrigger class={buttonVariants({ variant: 'destructive' })}>
							<TrashIcon class="size-4" />
							Clear
						</Dialog.DialogTrigger>
						<Dialog.DialogContent>
							<Dialog.DialogHeader>
								<Dialog.DialogTitle>Remove API Key</Dialog.DialogTitle>
								<Dialog.DialogDescription>
									This will remove your saved Fireflies API key from local storage. You will need to
									enter it again to use Fireflies features.
								</Dialog.DialogDescription>
							</Dialog.DialogHeader>
							<Dialog.DialogFooter>
								<Button variant="outline" onclick={() => (clearDialogOpen = false)}>
									Cancel
								</Button>
								<Button variant="destructive" onclick={handleClear}>
									Remove Key
								</Button>
							</Dialog.DialogFooter>
						</Dialog.DialogContent>
					</Dialog.Dialog>
				</div>
			{/if}

			<!-- Status Message -->
			{#if statusMessage}
				<div
					class="flex items-start gap-2 p-3 rounded-md text-sm {statusType === 'success'
						? 'bg-green-500/10 text-green-500 border border-green-500/20'
						: 'bg-destructive/10 text-destructive border border-destructive/20'}"
				>
					{#if statusType === 'success'}
						<CheckCircleIcon class="size-4 mt-0.5 shrink-0" />
					{/if}
					<div>
						<p>{statusMessage}</p>
						{#if meetingCount !== null && statusType === 'success'}
							<p class="mt-1 text-xs opacity-80">
								{meetingCount} meeting{meetingCount === 1 ? '' : 's'} accessible with this key
							</p>
						{/if}
					</div>
				</div>
			{/if}
		</Card.Content>
	</Card.Root>
</div>
