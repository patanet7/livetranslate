<script lang="ts">
	import { page } from '$app/stores';
</script>

<div class="flex items-center justify-center h-full">
	<div class="text-center">
		<h1 class="text-4xl font-bold">{$page.status}</h1>
		<p class="mt-2 text-muted-foreground">{$page.error?.message ?? 'Something went wrong'}</p>
		<a href="/" class="mt-4 inline-block text-sm underline">Back to Dashboard</a>
	</div>
</div>
