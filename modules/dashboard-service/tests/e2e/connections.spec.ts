import { test, expect } from '@playwright/test';

test.describe('AI Connections', () => {
	test.beforeEach(async ({ page }) => {
		await page.goto('/config/connections');
		await page.waitForTimeout(2000);
	});

	test('page loads and shows connections section', async ({ page }) => {
		await expect(page.getByText('AI Connections', { exact: true })).toBeVisible();
		await expect(page.getByText('Connections', { exact: true })).toBeVisible();
	});

	test('add connection dialog opens and submits', async ({ page }) => {
		await page.getByRole('button', { name: /add connection/i }).click();
		const dialog = page.getByRole('dialog');
		await expect(dialog).toBeVisible({ timeout: 3000 });

		await page.getByLabel('Name').fill('Test Ollama');
		await page.getByLabel('URL').clear();
		await page.getByLabel('URL').fill('http://test:11434');
		await page.getByLabel('Prefix ID').fill('test');
		await dialog.getByRole('button', { name: /add connection/i }).click();
	});

	test('delete connection shows confirmation', async ({ page }) => {
		// Add a connection first
		await page.getByRole('button', { name: /add connection/i }).click();
		const dialog = page.getByRole('dialog');
		await expect(dialog).toBeVisible({ timeout: 3000 });
		await page.getByLabel('Name').fill('Delete Me');
		await page.getByLabel('URL').clear();
		await page.getByLabel('URL').fill('http://deleteme:11434');
		await page.getByLabel('Prefix ID').fill('del');
		await dialog.getByRole('button', { name: /add connection/i }).click();
		await page.waitForTimeout(1000);

		// Click delete
		const cards = page.locator('[data-testid="connection-card"]');
		const lastCard = cards.last();
		await lastCard
			.locator('button')
			.filter({ has: page.locator('svg.lucide-trash-2') })
			.click();

		// Expect confirmation dialog
		const confirmDialog = page.getByRole('dialog');
		await expect(confirmDialog).toBeVisible({ timeout: 3000 });
		await expect(page.getByText('Delete Connection')).toBeVisible();
	});

	test('feature preferences section is visible', async ({ page }) => {
		await expect(page.getByText('Feature Model Preferences')).toBeVisible();
		await expect(page.getByText('chat', { exact: false })).toBeVisible();
		await expect(page.getByText('translation', { exact: false })).toBeVisible();
		await expect(page.getByText('intelligence', { exact: false })).toBeVisible();
	});
});
