<script lang="ts">
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
</script>

<PageHeader title="Live Feed" description="Real-time Fireflies transcript stream" />

<Card.Root>
	<Card.Header>
		<Card.Title>Live Feed</Card.Title>
	</Card.Header>
	<Card.Content>
		<p class="text-sm text-muted-foreground">
			Live transcript feed will appear here when a Fireflies session is active.
		</p>
	</Card.Content>
</Card.Root>
