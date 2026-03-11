import { test, expect } from '@playwright/test';

test.describe('Translation Connections', () => {
	test.beforeEach(async ({ page }) => {
		await page.goto('/config/translation');
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

		// Dialog should open
		await expect(page.getByRole('dialog')).toBeVisible();
		await expect(page.getByText('Add Connection')).toBeVisible();

		// Fill form
		await page.getByLabel('Name').fill('Test Ollama');
		await page.getByLabel('URL').clear();
		await page.getByLabel('URL').fill('http://192.168.1.100:11434');
		await page.getByLabel('Prefix ID').fill('test-ollama');

		// Submit via the dialog's action button
		await page.getByRole('dialog').getByRole('button', { name: /add connection/i }).click();

		// New card should appear
		await expect(page.getByText('Test Ollama')).toBeVisible();
		await expect(page.getByText('http://192.168.1.100:11434')).toBeVisible();
	});

	test('verify connection shows status', async ({ page }) => {
		// Click verify on first connection
		const firstVerify = page.getByRole('button', { name: /verify/i }).first();
		await firstVerify.click();

		// Should show either connected or error status (depending on backend availability)
		// The spinner appears first, then resolves to green or red
		await expect(
			page.locator('[data-testid="connection-card"] .text-green-500, [data-testid="connection-card"] .text-red-500').first()
		).toBeVisible({ timeout: 15000 });
	});

	test('edit connection via configure button', async ({ page }) => {
		// First add a connection so we have a second one to edit
		await page.getByRole('button', { name: /add connection/i }).click();
		await page.getByLabel('Name').fill('Edit Test');
		await page.getByLabel('URL').clear();
		await page.getByLabel('URL').fill('http://edit-test:11434');
		await page.getByLabel('Prefix ID').fill('edit-test');
		await page.getByRole('dialog').getByRole('button', { name: /add connection/i }).click();
		await expect(page.getByText('Edit Test')).toBeVisible();

		// Click gear icon on the new connection (last one)
		const cards = page.locator('[data-testid="connection-card"]');
		const lastCard = cards.last();
		await lastCard.locator('button').filter({ has: page.locator('svg.lucide-settings') }).click();

		// Dialog should open with pre-filled values
		await expect(page.getByRole('dialog')).toBeVisible();
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
		await page.getByLabel('Name').fill('Temp Connection');
		await page.getByLabel('URL').clear();
		await page.getByLabel('URL').fill('http://temp:11434');
		await page.getByLabel('Prefix ID').fill('temp');
		await page.getByRole('dialog').getByRole('button', { name: /add connection/i }).click();
		await expect(page.getByText('Temp Connection')).toBeVisible();

		// Click delete on that connection (last card's trash icon)
		const cards = page.locator('[data-testid="connection-card"]');
		const lastCard = cards.last();
		await lastCard.locator('button').filter({ has: page.locator('svg.lucide-trash-2') }).click();

		// Should be removed
		await expect(page.getByText('Temp Connection')).not.toBeVisible();
	});

	test('toggle connection enable/disable', async ({ page }) => {
		const firstCard = page.locator('[data-testid="connection-card"]').first();

		// Click power toggle on first connection
		const powerButton = firstCard.locator('button').filter({ has: page.locator('svg.lucide-power') });
		await powerButton.click();

		// Card should show dimmed
		await expect(firstCard).toHaveClass(/opacity-50/);

		// Click again to re-enable
		await powerButton.click();
		await expect(firstCard).not.toHaveClass(/opacity-50/);
	});

	test('engine selection changes URL default', async ({ page }) => {
		await page.getByRole('button', { name: /add connection/i }).click();

		const urlInput = page.getByLabel('URL');

		// Default engine is Ollama, URL should be Ollama default
		await expect(urlInput).toHaveValue('http://localhost:11434');

		// Change engine to vLLM
		await page.getByLabel('Engine').selectOption('vllm');
		await expect(urlInput).toHaveValue('http://localhost:8000');

		// Change to Triton
		await page.getByLabel('Engine').selectOption('triton');
		await expect(urlInput).toHaveValue('http://localhost:8001');
	});

	test('existing settings sections are preserved', async ({ page }) => {
		await expect(page.getByText('Current Model')).toBeVisible();
		await expect(page.getByText('Translation Settings')).toBeVisible();
		await expect(page.getByText('Prompt Template')).toBeVisible();
	});
});
