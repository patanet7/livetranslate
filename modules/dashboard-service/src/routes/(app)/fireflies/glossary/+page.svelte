<script lang="ts">
	import { enhance } from '$app/forms';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import * as Table from '$lib/components/ui/table';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';

	let { data, form } = $props();
</script>

<PageHeader title="Glossary" description="Manage translation glossary terms" />

<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
	<!-- Add Entry Form -->
	<div>
		<Card.Root>
			<Card.Header>
				<Card.Title>Add Term</Card.Title>
			</Card.Header>
			<Card.Content>
				<form method="POST" action="?/addEntry" use:enhance class="space-y-3">
					<input type="hidden" name="glossary_id" value={data.activeGlossaryId ?? ''} />

					<div class="space-y-1">
						<Label for="source_term">Source Term</Label>
						<Input id="source_term" name="source_term" placeholder="heart attack" required />
					</div>

					<div class="space-y-1">
						<Label for="translation">Translation</Label>
						<Input id="translation" name="translation" placeholder="infarto de miocardio" required />
					</div>

					<div class="space-y-1">
						<Label for="target_language">Target Language</Label>
						<Input id="target_language" name="target_language" value="es" />
					</div>

					{#if form?.errors?.form}
						<p class="text-sm text-destructive">{form.errors.form}</p>
					{/if}

					{#if form?.success}
						<p class="text-sm text-green-600">Term added successfully</p>
					{/if}

					<Button type="submit" class="w-full">Add Term</Button>
				</form>
			</Card.Content>
		</Card.Root>
	</div>

	<!-- Entries Table -->
	<div class="lg:col-span-2">
		<Card.Root>
			<Card.Header>
				<Card.Title>
					Glossary Entries
					{#if data.glossaries.length > 0}
						<span class="text-sm font-normal text-muted-foreground ml-2">
							({data.entries.length} terms)
						</span>
					{/if}
				</Card.Title>
			</Card.Header>
			<Card.Content class="p-0">
				{#if data.entries.length === 0}
					<div class="p-6 text-center text-muted-foreground">No glossary entries yet</div>
				{:else}
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head>Source Term</Table.Head>
								<Table.Head>Translations</Table.Head>
								<Table.Head>Priority</Table.Head>
								<Table.Head class="w-16"></Table.Head>
							</Table.Row>
						</Table.Header>
						<Table.Body>
							{#each data.entries as entry}
								<Table.Row>
									<Table.Cell class="font-medium">{entry.source_term}</Table.Cell>
									<Table.Cell>
										{#each Object.entries(entry.translations) as [lang, text]}
											<span class="text-xs bg-accent px-1.5 py-0.5 rounded mr-1">
												{lang}: {text}
											</span>
										{/each}
									</Table.Cell>
									<Table.Cell>{entry.priority}</Table.Cell>
									<Table.Cell>
										<form method="POST" action="?/deleteEntry" use:enhance>
											<input type="hidden" name="glossary_id" value={data.activeGlossaryId} />
											<input type="hidden" name="entry_id" value={entry.entry_id} />
											<Button variant="ghost" size="sm" type="submit">x</Button>
										</form>
									</Table.Cell>
								</Table.Row>
							{/each}
						</Table.Body>
					</Table.Root>
				{/if}
			</Card.Content>
		</Card.Root>
	</div>
</div>
