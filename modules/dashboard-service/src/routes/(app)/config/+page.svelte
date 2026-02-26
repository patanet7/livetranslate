<script lang="ts">
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';

  const sections = [
    { title: 'Audio', href: '/config/audio', description: 'Audio processing, sample rates, and device settings' },
    { title: 'Translation', href: '/config/translation', description: 'Translation models, languages, and quality settings' },
    { title: 'System', href: '/config/system', description: 'System settings, feature flags, and service management' }
  ];
</script>

<PageHeader title="Configuration" description="Manage system settings" />

<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
  {#each sections as section}
    <a href={section.href} class="block">
      <Card.Root class="hover:border-primary transition-colors h-full">
        <Card.Header>
          <Card.Title>{section.title}</Card.Title>
          <Card.Description>{section.description}</Card.Description>
        </Card.Header>
      </Card.Root>
    </a>
  {/each}
</div>
