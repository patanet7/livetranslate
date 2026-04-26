/**
 * Test that runner.ts consumes selectors from the canonical lib/selectors.ts
 * rather than inlining literal strings. Phase 9.3 of PLAN_7.
 *
 * Strategy: read runner.ts source as text and assert the absence of
 * specific Meet selector strings (e.g. 'button[aria-label="Leave call"]')
 * in the file body — they should appear in selectors.ts only.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const RUNNER_SRC = readFileSync(resolve(__dirname, '../src/runner.ts'), 'utf-8');

describe('runner uses canonical selectors', () => {
  it('imports JOIN_BUTTON_SELECTORS from lib/selectors', () => {
    expect(RUNNER_SRC).toMatch(/from ['"]\.\/lib\/selectors['"]/);
    expect(RUNNER_SRC).toMatch(/JOIN_BUTTON_SELECTORS/);
  });

  it('imports LEAVE_BUTTON_SELECTORS', () => {
    expect(RUNNER_SRC).toMatch(/LEAVE_BUTTON_SELECTORS/);
  });

  it('imports PEOPLE_BUTTON_SELECTORS for in-call signal', () => {
    expect(RUNNER_SRC).toMatch(/PEOPLE_BUTTON_SELECTORS/);
  });

  it('does not inline the Leave call literal', () => {
    // Allow the literal in selectors.ts but not in runner.ts.
    expect(RUNNER_SRC).not.toMatch(/'button\[aria-label="Leave call"\]'/);
  });

  it('does not inline the Join now literal', () => {
    expect(RUNNER_SRC).not.toMatch(/'button:has-text\("Join now"\)'/);
  });

  it('uses clickFirst / findVisible / waitForAny helpers', () => {
    // At least one of these utility functions should be imported.
    expect(RUNNER_SRC).toMatch(/clickFirst|findVisible|waitForAny/);
  });
});
