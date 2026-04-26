<script lang="ts">
	import type { HTMLAttributes } from "svelte/elements";
	import { cn, type WithElementRef } from "$lib/utils.js";

	let {
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> = $props();
</script>

<!--
  Card titles use the editorial small-caps eyebrow treatment so every
  `Card.Title` across the app reads as a magazine "department" label.
  Override the default by passing a className with `font-display` etc.
-->
<div
	bind:this={ref}
	data-slot="card-title"
	class={cn(
		"leading-none font-display tracking-[0.04em]",
		"text-[0.9375rem] text-ink",
		className
	)}
	style="font-variation-settings: 'opsz' 36, 'SOFT' 30; font-weight: 500;"
	{...restProps}
>
	{@render children?.()}
</div>
