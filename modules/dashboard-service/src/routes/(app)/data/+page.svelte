<script lang="ts">
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
</script>

<PageHeader title="Data & Logs" description="System data, transcription logs, and analytics" />

<Card.Root>
	<Card.Header>
		<Card.Title>Data & Logs</Card.Title>
	</Card.Header>
	<Card.Content>
		<p class="text-sm text-muted-foreground">
			Data exploration and log viewing coming soon.
		</p>
	</Card.Content>
</Card.Root>
