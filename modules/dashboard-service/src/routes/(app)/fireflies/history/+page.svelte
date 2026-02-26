<script lang="ts">
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
  import * as Card from '$lib/components/ui/card';
  import * as Table from '$lib/components/ui/table';

  let { data } = $props();
</script>

<PageHeader title="Session History" description="Past and active Fireflies sessions" />

<Card.Root>
  <Card.Content class="p-0">
    {#await data.sessions}
      <div class="p-6 text-center text-muted-foreground">Loading sessions...</div>
    {:then sessions}
      {#if sessions.length === 0}
        <div class="p-6 text-center text-muted-foreground">No sessions found</div>
      {:else}
        <Table.Root>
          <Table.Header>
            <Table.Row>
              <Table.Head>Session ID</Table.Head>
              <Table.Head>Status</Table.Head>
              <Table.Head>Chunks</Table.Head>
              <Table.Head>Translations</Table.Head>
              <Table.Head>Speakers</Table.Head>
              <Table.Head>Connected</Table.Head>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {#each sessions as session}
              <Table.Row>
                <Table.Cell>
                  <a href="/fireflies/connect?session={session.session_id}" class="text-primary hover:underline font-mono text-xs">
                    {session.session_id.slice(0, 20)}...
                  </a>
                </Table.Cell>
                <Table.Cell>
                  <StatusIndicator
                    status={session.connection_status === 'CONNECTED' ? 'connected' : 'disconnected'}
                    label={session.connection_status}
                  />
                </Table.Cell>
                <Table.Cell>{session.chunks_received}</Table.Cell>
                <Table.Cell>{session.translations_completed}</Table.Cell>
                <Table.Cell>{session.speakers_detected.length}</Table.Cell>
                <Table.Cell class="text-xs text-muted-foreground">
                  {new Date(session.connected_at).toLocaleString()}
                </Table.Cell>
              </Table.Row>
            {/each}
          </Table.Body>
        </Table.Root>
      {/if}
    {:catch error}
      <div class="p-6 text-center text-destructive">Failed to load sessions: {error.message}</div>
    {/await}
  </Card.Content>
</Card.Root>
