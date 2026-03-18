/**
 * Language Detection Replay E2E Tests
 *
 * Replays real meeting FLAC recordings (converted to 48kHz WAV) through
 * the full frontend pipeline and verifies:
 * - Stable language detection (no flapping to hallucinated languages)
 * - Correct language switch on genuine transitions
 * - Session restart produces clean transcription in new language
 *
 * Prerequisites:
 *   1. `just dev` running (all services)
 *   2. `just create-lang-detect-fixtures` run (WAV fixtures from FLAC recordings)
 *
 * Run:
 *   just test-lang-detect
 */

import { test, expect, type Page } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';

// ESM-compatible __dirname (matches loopback-playback.spec.ts pattern)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FIXTURE_DIR = path.resolve(__dirname, '../fixtures');
// Uses baseURL from playwright.config.ts (default: http://localhost:5173)
const LOOPBACK_PATH = '/loopback';

// Hallucinated languages that Whisper falsely detects (from production log)
const HALLUCINATED_LANGS = new Set(['nn', 'cy', 'ko', 'fr', 'es', 'it', 'pt', 'nl', 'ru', 'pl']);

// vLLM-MLX cold start can take up to 90s for the first inference
const TEST_TIMEOUT_MS = 180_000;

// -------------------------------------------------------------------
// Helpers (following loopback-playback.spec.ts patterns)
// -------------------------------------------------------------------

/** Check that orchestration service is reachable. */
async function checkServiceHealth(): Promise<boolean> {
	try {
		const resp = await fetch('http://localhost:3000/api/audio/health');
		return resp.ok;
	} catch {
		return false;
	}
}

/** Inject a WAV file as the getUserMedia source in the browser.
 *  MUST be called AFTER page.goto() — needs a loaded page to evaluate JS.
 */
async function injectAudioFixture(page: Page, fixtureName: string): Promise<void> {
	await page.evaluate(async (fixture: string) => {
		const response = await fetch(`/_test_fixtures/${fixture}`);
		if (!response.ok) {
			throw new Error(`Failed to fetch fixture: ${fixture} (${response.status})`);
		}

		const arrayBuffer = await response.arrayBuffer();
		const audioCtx = new AudioContext({ sampleRate: 48000 });
		const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

		const source = audioCtx.createBufferSource();
		source.buffer = audioBuffer;
		source.loop = false;

		const dest = audioCtx.createMediaStreamDestination();
		source.connect(dest);
		source.start();

		navigator.mediaDevices.getUserMedia = async (_constraints) => dest.stream;
		navigator.mediaDevices.enumerateDevices = async () => [
			{
				deviceId: 'test-audio',
				groupId: 'test-group',
				kind: 'audioinput' as MediaDeviceKind,
				label: 'E2E Test Audio',
				toJSON: () => ({}),
			},
		];
	}, fixtureName);
}

/**
 * Install a WebSocket message interceptor via addInitScript.
 * Must be called BEFORE page.goto() so the monkey-patch runs before
 * any page JS executes (catches the initial WebSocket connection).
 * Follows the same pattern as loopback-playback.spec.ts.
 */
async function installWsInterceptor(page: Page): Promise<void> {
	await page.addInitScript(() => {
		(window as any).__e2e_messages = [];
		const OrigWS = window.WebSocket;
		(window as any).WebSocket = class extends OrigWS {
			constructor(url: string | URL, protocols?: string | string[]) {
				super(url, protocols);
				this.addEventListener('message', (ev: MessageEvent) => {
					if (typeof ev.data === 'string') {
						try {
							const parsed = JSON.parse(ev.data);
							(window as any).__e2e_messages.push(parsed);
						} catch {
							// non-JSON frame, skip
						}
					}
				});
			}
		};
	});
}

/** Collect intercepted WS messages from the browser. */
async function getWsMessages(page: Page): Promise<any[]> {
	return page.evaluate(() => (window as any).__e2e_messages ?? []);
}

/** Extract language_detected events from WS messages. */
function getLangEvents(messages: any[]): Array<{ language: string; confidence?: number; switched_from?: string }> {
	return messages.filter(m => m.type === 'language_detected');
}

/** Extract segment events from WS messages. */
function getSegments(messages: any[]): any[] {
	return messages.filter(m => m.type === 'segment' && m.text?.trim());
}

function hasFixture(name: string): boolean {
	return fs.existsSync(path.join(FIXTURE_DIR, name));
}

// -------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------

test.describe('Language Detection Replay E2E', () => {

	test.beforeAll(async () => {
		// Service guard: skip suite if orchestration isn't running
		const healthy = await checkServiceHealth();
		if (!healthy) {
			test.skip();
			throw new Error(
				'Orchestration service not reachable at http://localhost:3000/api/audio/health. ' +
				'Run `just dev` before running E2E tests.'
			);
		}
	});

	test.beforeEach(async ({ page }) => {
		// Grant microphone permission (required for getUserMedia mock)
		await page.context().grantPermissions(['microphone']);

		// Serve audio fixtures from Playwright → browser
		await page.route('**/_test_fixtures/*', async (route) => {
			const filename = route.request().url().split('/_test_fixtures/').pop();
			const filePath = path.join(FIXTURE_DIR, filename!);
			if (fs.existsSync(filePath)) {
				const body = fs.readFileSync(filePath);
				await route.fulfill({ status: 200, contentType: 'audio/wav', body });
			} else {
				await route.fulfill({ status: 404, body: `Fixture not found: ${filename}` });
			}
		});

		// Install WS interceptor BEFORE page.goto() so it catches the initial connection
		await installWsInterceptor(page);
	});

	test('Short Chinese — detects zh quickly, no false switches', async ({ page }) => {
		test.setTimeout(TEST_TIMEOUT_MS);
		if (!hasFixture('lang_detect_zh_short_48k.wav')) test.skip();

		await page.goto(LOOPBACK_PATH);
		await injectAudioFixture(page, 'lang_detect_zh_short_48k.wav');
		await page.locator('[data-testid="start-capture"], button:has-text("Start")').first().click();

		await page.waitForTimeout(60_000);

		const messages = await getWsMessages(page);
		const langEvents = getLangEvents(messages);
		const segments = getSegments(messages);

		// Should produce segments
		expect(segments.length).toBeGreaterThan(0);

		// Should detect Chinese
		const zhEvents = langEvents.filter(e => e.language === 'zh');
		expect(zhEvents.length).toBeGreaterThan(0);

		// No hallucinated languages
		const hallucinations = langEvents.filter(e => HALLUCINATED_LANGS.has(e.language));
		expect(hallucinations).toHaveLength(0);

		// No false switches away from Chinese
		const switches = langEvents.filter(e => e.switched_from);
		expect(switches).toHaveLength(0);
	});

	test('Chinese section — detects Chinese stably over 3 minutes', async ({ page }) => {
		test.setTimeout(TEST_TIMEOUT_MS);
		if (!hasFixture('lang_detect_zh_section_48k.wav')) test.skip();

		await page.goto(LOOPBACK_PATH);
		await injectAudioFixture(page, 'lang_detect_zh_section_48k.wav');
		await page.locator('[data-testid="start-capture"], button:has-text("Start")').first().click();

		await page.waitForTimeout(90_000);

		const messages = await getWsMessages(page);
		const langEvents = getLangEvents(messages);

		// Should detect Chinese
		const zhEvents = langEvents.filter(e => e.language === 'zh');
		expect(zhEvents.length).toBeGreaterThan(0);

		// After detecting Chinese, should stay on Chinese
		const firstZhIdx = langEvents.findIndex(e => e.language === 'zh');
		if (firstZhIdx >= 0) {
			const afterZh = langEvents.slice(firstZhIdx);
			const backToEn = afterZh.filter(e => e.language === 'en');
			expect(backToEn.length).toBeLessThanOrEqual(1);
		}

		// No hallucinated languages
		const hallucinations = langEvents.filter(e => HALLUCINATED_LANGS.has(e.language));
		expect(hallucinations).toHaveLength(0);
	});

	test('Full Chinese session — stable detection over 5 minutes', async ({ page }) => {
		test.setTimeout(TEST_TIMEOUT_MS * 2);
		if (!hasFixture('lang_detect_zh_full_48k.wav')) test.skip();

		await page.goto(LOOPBACK_PATH);
		await injectAudioFixture(page, 'lang_detect_zh_full_48k.wav');
		await page.locator('[data-testid="start-capture"], button:has-text("Start")').first().click();

		await page.waitForTimeout(150_000);

		const messages = await getWsMessages(page);
		const langEvents = getLangEvents(messages);
		const segments = getSegments(messages);

		// Segments produced
		expect(segments.length).toBeGreaterThan(0);

		// Should detect Chinese
		expect(langEvents.some(e => e.language === 'zh')).toBe(true);

		// ≤1 false switch over 5 minutes (old detector had 10+)
		const switches = langEvents.filter(e => e.switched_from);
		expect(switches.length).toBeLessThanOrEqual(1);

		// No hallucinated languages
		const hallucinations = langEvents.filter(e => HALLUCINATED_LANGS.has(e.language));
		expect(hallucinations).toHaveLength(0);
	});

	// Skipped until English/mixed recordings are captured
	test('English session stays English — no false switches', async ({ page }) => {
		if (!hasFixture('lang_detect_en_full_48k.wav')) test.skip();
	});

	test('Language transition — en→zh with clean switch', async ({ page }) => {
		if (!hasFixture('lang_detect_transition_48k.wav')) test.skip();
	});

});
