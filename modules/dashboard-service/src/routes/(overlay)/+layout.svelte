<script lang="ts">
	let { children } = $props();
</script>

<div class="overlay-root">
	{@render children()}
</div>

<style>
	.overlay-root {
		background: transparent;
		width: 100vw;
		height: 100vh;
		overflow: hidden;
	}
</style>
