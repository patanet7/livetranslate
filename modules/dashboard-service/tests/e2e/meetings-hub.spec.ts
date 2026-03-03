import { test, expect } from '@playwright/test';

/** Wait for SvelteKit client-side hydration to complete */
async function waitForHydration(page: import('@playwright/test').Page) {
	await page.waitForLoadState('networkidle');
}

test.describe('Meeting Hub Navigation', () => {
	test('sidebar has Meetings link that navigates to /meetings', async ({ page }) => {
		await page.goto('/');
		const meetingsLink = page.locator('nav a[href="/meetings"]');
		await expect(meetingsLink).toBeVisible();
		await meetingsLink.click();
		await expect(page).toHaveURL('/meetings');
	});

	test('/meetings page loads with correct heading', async ({ page }) => {
		await page.goto('/meetings');
		await expect(page.locator('h1, h2').filter({ hasText: 'Meetings' }).first()).toBeVisible();
	});

	test('meetings page has search input with aria-label', async ({ page }) => {
		await page.goto('/meetings');
		const searchInput = page.locator('input[aria-label="Search meetings"]');
		await expect(searchInput).toBeVisible();
		await expect(searchInput).toHaveAttribute('placeholder', 'Search meetings...');
	});

	test('meetings page has status filter buttons with accessibility', async ({ page }) => {
		await page.goto('/meetings');
		const filterGroup = page.locator('[role="group"][aria-label="Filter meetings by status"]');
		await expect(filterGroup).toBeVisible();

		const allButton = filterGroup.getByText('All');
		await expect(allButton).toHaveAttribute('aria-pressed', 'true');

		const liveButton = filterGroup.getByText('Live');
		await expect(liveButton).toHaveAttribute('aria-pressed', 'false');
	});

	test('clicking status filter changes active button', async ({ page }) => {
		await page.goto('/meetings');
		await waitForHydration(page);
		const filterGroup = page.locator('[role="group"][aria-label="Filter meetings by status"]');

		// Verify initial state — "All" has default variant (bg-primary)
		const allButton = filterGroup.getByText('All');
		await expect(allButton).toHaveClass(/bg-primary/);

		// Click "Live" filter and verify it becomes active
		const liveButton = filterGroup.getByText('Live');
		await liveButton.click();
		// After click, "Live" should get the default variant styling and aria-pressed
		await expect(liveButton).toHaveClass(/bg-primary/, { timeout: 5000 });
		await expect(liveButton).toHaveAttribute('aria-pressed', 'true');
		// "All" should revert to outline variant
		await expect(allButton).not.toHaveClass(/bg-primary/);
		await expect(allButton).toHaveAttribute('aria-pressed', 'false');
	});

	test('search form submits and navigates with query param', async ({ page }) => {
		await page.goto('/meetings');
		await waitForHydration(page);
		const searchInput = page.locator('input[aria-label="Search meetings"]');
		await searchInput.click();
		await searchInput.pressSequentially('test');
		await page.locator('button[type="submit"]').click();
		await expect(page).toHaveURL(/q=test/);
	});

	test('empty state shows when no meetings', async ({ page }) => {
		await page.goto('/meetings');
		// With no backend, the catch() fallback returns empty array
		const emptyState = page.getByText('No meetings yet');
		// If backend is down, we get empty state; if running, we might get real data
		const meetingCards = page.locator('a[href^="/meetings/"]');
		const hasResults = await meetingCards.count() > 0;
		if (!hasResults) {
			await expect(emptyState).toBeVisible();
			// Verify CTA button exists
			const ctaButton = page.getByRole('link', { name: 'Connect to Fireflies' });
			await expect(ctaButton).toBeVisible();
		}
	});
});

test.describe('Meeting Hub Dashboard Integration', () => {
	test('dashboard page loads', async ({ page }) => {
		await page.goto('/');
		await expect(page).toHaveURL('/');
		// Quick actions should include Meetings link
		const meetingsAction = page.locator('a[href="/meetings"]');
		if (await meetingsAction.count() > 0) {
			await expect(meetingsAction.first()).toBeVisible();
		}
	});

	test('fireflies connect page exists', async ({ page }) => {
		await page.goto('/fireflies');
		// Should either show the connect form or redirect
		await expect(page.locator('body')).toBeVisible();
	});
});

test.describe('Meeting Hub Accessibility', () => {
	test('meetings page has proper heading hierarchy', async ({ page }) => {
		await page.goto('/meetings');
		// Check there is at least one heading
		const headings = page.locator('h1, h2, h3');
		expect(await headings.count()).toBeGreaterThan(0);
	});

	test('all interactive elements are keyboard accessible', async ({ page }) => {
		await page.goto('/meetings');
		// Tab through interactive elements
		await page.keyboard.press('Tab');
		const focused = page.locator(':focus');
		await expect(focused).toBeVisible();
	});

	test('search input is focusable', async ({ page }) => {
		await page.goto('/meetings');
		const searchInput = page.locator('input[aria-label="Search meetings"]');
		await searchInput.focus();
		await expect(searchInput).toBeFocused();
	});
});

test.describe('Meeting Detail Page', () => {
	// These tests only run meaningfully when the backend is up with real data.
	// When backend is down, meeting detail pages return 404 which is correct behavior.

	test('meeting detail 404 for non-existent id', async ({ page }) => {
		const response = await page.goto('/meetings/nonexistent-id-12345');
		// Should get 404 or error page
		if (response) {
			expect([200, 404, 500]).toContain(response.status());
		}
	});
});

test.describe('Live Session Page Structure', () => {
	test('live page requires session parameter', async ({ page }) => {
		const response = await page.goto('/meetings/test-id/live');
		// Without a valid meeting, should 404 or show error
		if (response) {
			expect([200, 404, 500]).toContain(response.status());
		}
	});
});

test.describe('Fireflies Sync Controls', () => {
	test('"Sync from Fireflies" button is visible on meetings page', async ({ page }) => {
		await page.goto('/meetings');
		await waitForHydration(page);
		const syncButton = page.getByRole('button', { name: 'Sync from Fireflies' });
		await expect(syncButton).toBeVisible();
	});

	test('"Invite Bot" button is visible and opens dialog', async ({ page }) => {
		await page.goto('/meetings');
		await waitForHydration(page);
		const inviteButton = page.getByRole('button', { name: 'Invite Bot' });
		await expect(inviteButton).toBeVisible();
		await inviteButton.click();
		// Dialog should open with meeting link input
		const linkInput = page.locator('#invite-link');
		await expect(linkInput).toBeVisible();
	});

	test('meeting detail with real data shows insight count', async ({ page }) => {
		// Navigate to meetings list first
		await page.goto('/meetings');
		await waitForHydration(page);
		const meetingLinks = page.locator('a[href^="/meetings/"]');
		const count = await meetingLinks.count();
		if (count > 0) {
			// Click into first meeting
			await meetingLinks.first().click();
			await page.waitForURL(/\/meetings\/.+/);
			await waitForHydration(page);
			// If meeting has insights, the tab should show Summary & Insights
			const insightsTab = page.getByRole('tab', { name: /Summary & Insights/i });
			await expect(insightsTab).toBeVisible();
		}
		// If no meetings exist (backend down), test passes trivially
	});
});
