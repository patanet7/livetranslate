/**
 * Vitest config — explicitly excludes Playwright e2e specs and agent worktree
 * snapshots. Without this, `npm run test` picks up `tests/e2e/*.spec.ts` (which
 * are Playwright tests, not vitest) and fails them at the suite level because
 * `test.describe` from `@playwright/test` rejects being called outside a
 * Playwright run.
 *
 * Playwright runs separately via `npm run test:e2e` (which delegates to
 * `npx playwright test`).
 */
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	test: {
		include: ['tests/unit/**/*.{test,spec}.{js,ts}'],
		exclude: [
			'tests/e2e/**',
			'node_modules/**',
			'.svelte-kit/**',
			'**/.claude/**',
			'build/**',
			'dist/**',
		],
	},
});
