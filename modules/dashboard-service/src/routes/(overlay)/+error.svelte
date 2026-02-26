<script lang="ts">
	import { page } from '$app/stores';
</script>

<div class="flex items-center justify-center h-screen text-white">
	<p>Error {$page.status}: {$page.error?.message}</p>
</div>
