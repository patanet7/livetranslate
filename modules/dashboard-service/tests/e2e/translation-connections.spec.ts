import { test, expect } from '@playwright/test';

test.describe('Translation Connections', () => {
	test.beforeEach(async ({ page }) => {
		await page.goto('/config/translation');
		// Wait for auto-verify and aggregation to settle
		await page.waitForTimeout(2000);
	});

	test('page loads and shows connections section', async ({ page }) => {
		await expect(page.getByText('Translation Connections')).toBeVisible();
		await expect(page.getByRole('button', { name: /add connection/i })).toBeVisible();
	});

	test('default connection is visible', async ({ page }) => {
		await expect(page.locator('[data-testid="connection-card"]').first()).toBeVisible();
	});

	test('add connection dialog opens and submits', async ({ page }) => {
		await page.getByRole('button', { name: /add connection/i }).click();

		// Wait for dialog to appear (portal rendering)
		const dialog = page.getByRole('dialog');
		await expect(dialog).toBeVisible({ timeout: 3000 });
		await expect(page.getByText('Add a new translation backend')).toBeVisible();

		// Fill form
		await page.getByLabel('Name').fill('Test Ollama');
		await page.getByLabel('URL').clear();
		await page.getByLabel('URL').fill('http://192.168.1.100:11434');
		await page.getByLabel('Prefix ID').fill('test-ollama');

		// Submit via the dialog's action button
		await dialog.getByRole('button', { name: /add connection/i }).click();

		// New card should appear (use exact match to avoid matching toast)
		await expect(page.getByText('Test Ollama', { exact: true })).toBeVisible();
		await expect(page.getByText('http://192.168.1.100:11434')).toBeVisible();
	});

	test('verify connection shows status', async ({ page }) => {
		// Click verify on first connection
		const firstVerify = page.getByRole('button', { name: /verify/i }).first();
		await firstVerify.click();

		// Should show either connected or error status (depending on backend availability)
		await expect(
			page
				.locator(
					'[data-testid="connection-card"] .text-green-500, [data-testid="connection-card"] .text-red-500'
				)
				.first()
		).toBeVisible({ timeout: 15000 });
	});

	test('edit connection via configure button', async ({ page }) => {
		// First add a connection so we have a second one to edit
		await page.getByRole('button', { name: /add connection/i }).click();
		const dialog = page.getByRole('dialog');
		await expect(dialog).toBeVisible({ timeout: 3000 });

		await page.getByLabel('Name').fill('Edit Test');
		await page.getByLabel('URL').clear();
		await page.getByLabel('URL').fill('http://edit-test:11434');
		await page.getByLabel('Prefix ID').fill('edit-test');
		await dialog.getByRole('button', { name: /add connection/i }).click();
		await expect(page.getByText('Edit Test', { exact: true })).toBeVisible();

		// Click gear icon on the new connection (last one)
		const lastCard = page.locator('[data-testid="connection-card"]').last();
		await lastCard.locator('button').filter({ has: page.locator('svg.lucide-settings') }).click();

		// Dialog should open with Edit title
		await expect(page.getByRole('dialog')).toBeVisible({ timeout: 3000 });
		await expect(page.getByText('Edit Connection')).toBeVisible();

		// Modify name
		const nameInput = page.getByLabel('Name');
		await nameInput.clear();
		await nameInput.fill('Renamed Connection');

		await page.getByRole('dialog').getByRole('button', { name: /save changes/i }).click();

		// Card should show updated name
		await expect(page.getByText('Renamed Connection')).toBeVisible();
	});

	test('delete connection removes it', async ({ page }) => {
		// First add a connection so we have something to delete
		await page.getByRole('button', { name: /add connection/i }).click();
		const dialog = page.getByRole('dialog');
		await expect(dialog).toBeVisible({ timeout: 3000 });

		await page.getByLabel('Name').fill('Temp Connection');
		await page.getByLabel('URL').clear();
		await page.getByLabel('URL').fill('http://temp:11434');
		await page.getByLabel('Prefix ID').fill('temp');
		await dialog.getByRole('button', { name: /add connection/i }).click();
		await expect(page.getByText('Temp Connection')).toBeVisible();

		// Count cards before delete
		const cardsBefore = await page.locator('[data-testid="connection-card"]').count();

		// Click delete on that connection (last card's trash icon)
		const lastCard = page.locator('[data-testid="connection-card"]').last();
		await lastCard.locator('button').filter({ has: page.locator('svg.lucide-trash-2') }).click();

		// Confirm deletion in the confirmation dialog
		const confirmDialog = page.getByRole('dialog');
		await expect(confirmDialog).toBeVisible({ timeout: 3000 });
		await expect(page.getByText('Delete Connection')).toBeVisible();
		await confirmDialog.getByRole('button', { name: /delete/i }).click();

		// Should have one fewer card
		await expect(page.locator('[data-testid="connection-card"]')).toHaveCount(cardsBefore - 1);
	});

	test('toggle connection enable/disable', async ({ page }) => {
		// Add a fresh connection that won't auto-verify
		await page.getByRole('button', { name: /add connection/i }).click();
		const dialog = page.getByRole('dialog');
		await expect(dialog).toBeVisible({ timeout: 3000 });

		await page.getByLabel('Name').fill('Toggle Test');
		await page.getByLabel('URL').clear();
		await page.getByLabel('URL').fill('http://toggle-test:11434');
		await page.getByLabel('Prefix ID').fill('toggle');
		await dialog.getByRole('button', { name: /add connection/i }).click();
		// Use exact match to avoid matching toast text like "Toggle Test: Connection failed"
		await expect(page.getByText('Toggle Test', { exact: true })).toBeVisible();

		const lastCard = page.locator('[data-testid="connection-card"]').last();

		// Click power toggle to disable
		const powerButton = lastCard
			.locator('button')
			.filter({ has: page.locator('svg.lucide-power') });
		await powerButton.click();

		// Card should show dimmed
		await expect(lastCard).toHaveClass(/opacity-50/);

		// Click again to re-enable
		await powerButton.click();
		await expect(lastCard).not.toHaveClass(/opacity-50/);
	});

	test('engine selection changes URL default', async ({ page }) => {
		await page.getByRole('button', { name: /add connection/i }).click();
		const dialog = page.getByRole('dialog');
		await expect(dialog).toBeVisible({ timeout: 3000 });

		const urlInput = dialog.getByLabel('URL');

		// Default engine is Ollama, URL should be Ollama default
		await expect(urlInput).toHaveValue('http://localhost:11434');

		// Change engine to vLLM
		await dialog.getByLabel('Engine').selectOption('vllm');
		await expect(urlInput).toHaveValue('http://localhost:8000');

		// Change to Triton
		await dialog.getByLabel('Engine').selectOption('triton');
		await expect(urlInput).toHaveValue('http://localhost:8001');
	});

	test('existing settings sections are preserved', async ({ page }) => {
		await expect(page.getByText('Current Model')).toBeVisible();
		await expect(page.getByText('Translation Settings')).toBeVisible();
		await expect(page.getByText('Prompt Template', { exact: true })).toBeVisible();
	});
});
