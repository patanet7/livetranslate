import { test, expect, Page } from "@playwright/test";

test.describe("Bot Management E2E", () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;

    // Mock API responses
    await page.route("/api/bot/active", async (route) => {
      await route.fulfill({
        json: { bots: [], activeBotIds: [] },
      });
    });

    await page.route("/api/bot/stats", async (route) => {
      await route.fulfill({
        json: {
          totalBotsSpawned: 0,
          activeBots: 0,
          completedSessions: 0,
          errorRate: 0,
          averageSessionDuration: 0,
        },
      });
    });

    await page.goto("/");

    // Navigate to Bot Management page
    await page.click('[data-testid="nav-bot-management"]');
    await expect(page.locator("h1")).toContainText("Bot Management Dashboard");
  });

  test("should display the bot management dashboard", async () => {
    // Check main dashboard elements
    await expect(page.locator("h1")).toContainText("Bot Management Dashboard");
    await expect(page.getByText("Active Bots")).toBeVisible();
    await expect(page.getByText("Virtual Webcam")).toBeVisible();
    await expect(page.getByText("Session Database")).toBeVisible();
    await expect(page.getByText("Analytics")).toBeVisible();
    await expect(page.getByText("Settings")).toBeVisible();

    // Check status chips
    await expect(page.getByText("0 Active")).toBeVisible();
    await expect(page.getByText("0 Total")).toBeVisible();
    await expect(page.getByText("0 Completed")).toBeVisible();
  });

  test("should open and navigate through bot creation modal", async () => {
    // Click create bot FAB
    await page.click('[aria-label="create bot"]');

    // Should open create bot modal
    await expect(page.getByText("Create New Bot")).toBeVisible();

    // Step 1: Meeting Information
    await expect(page.getByText("Meeting Information")).toBeVisible();
    await page.fill('[label="Google Meet ID"]', "abc-defg-hij");
    await page.fill('[label="Meeting Title (Optional)"]', "E2E Test Meeting");
    await page.fill('[label="Organizer Email (Optional)"]', "test@example.com");

    // Navigate to next step
    await page.click('button:has-text("Next")');

    // Step 2: Translation Settings
    await expect(page.getByText("Translation Settings")).toBeVisible();
    await expect(page.getByText("Enable Auto-Translation")).toBeVisible();

    // Toggle French language
    await page.click("text=French");
    await page.click('button:has-text("Next")');

    // Step 3: Advanced Settings
    await expect(page.getByText("Advanced Settings")).toBeVisible();

    // Change priority to high
    await page.click('[aria-labelledby="bot-priority-label"]');
    await page.click("text=High Priority");
    await page.click('button:has-text("Next")');

    // Step 4: Review & Create
    await expect(page.getByText("Ready to Create Bot")).toBeVisible();
    await expect(page.getByText("abc-defg-hij")).toBeVisible();
    await expect(page.getByText("E2E Test Meeting")).toBeVisible();
    await expect(page.getByText("test@example.com")).toBeVisible();

    // Mock successful bot creation
    await page.route("/api/bot/spawn", async (route) => {
      await route.fulfill({
        json: {
          botId: "e2e-test-bot-123",
          bot: {
            botId: "e2e-test-bot-123",
            status: "spawning",
            meetingInfo: {
              meetingId: "abc-defg-hij",
              meetingTitle: "E2E Test Meeting",
              organizerEmail: "test@example.com",
              participantCount: 0,
            },
            audioCapture: {
              isCapturing: false,
              totalChunksCaptured: 0,
              averageQualityScore: 0,
              lastCaptureTimestamp: 0,
              deviceInfo: "",
            },
            captionProcessor: {
              totalCaptionsProcessed: 0,
              totalSpeakers: 0,
              speakerTimeline: [],
              lastCaptionTimestamp: 0,
            },
            virtualWebcam: {
              isStreaming: false,
              framesGenerated: 0,
              currentTranslations: [],
              webcamConfig: {
                width: 1280,
                height: 720,
                fps: 30,
                displayMode: "overlay",
                theme: "dark",
                maxTranslationsDisplayed: 5,
              },
            },
            timeCorrelation: {
              totalCorrelations: 0,
              successRate: 0,
              averageTimingOffset: 0,
              lastCorrelationTimestamp: 0,
            },
            performance: {
              sessionDuration: 0,
              totalProcessingTime: 0,
              averageLatency: 0,
              errorCount: 0,
            },
            createdAt: Date.now(),
            lastActiveAt: Date.now(),
          },
        },
      });
    });

    // Create the bot
    await page.click('button:has-text("Create Bot")');

    // Should show success notification
    await expect(page.getByText("created successfully")).toBeVisible();

    // Modal should close
    await expect(page.getByText("Create New Bot")).not.toBeVisible();
  });

  test("should navigate between tabs and show appropriate content", async () => {
    // Test Active Bots tab (default)
    await expect(page.getByText("Spawn New Bot")).toBeVisible();
    await expect(page.getByText("No Active Bots")).toBeVisible();

    // Test Virtual Webcam tab
    await page.click("text=Virtual Webcam");
    await expect(page.getByText("Virtual Webcam Manager")).toBeVisible();
    await expect(page.getByText("No Active Bots")).toBeVisible();

    // Test Session Database tab
    await page.click("text=Session Database");
    await expect(page.getByText("Session Database")).toBeVisible();
    await expect(page.getByText("Export All Data")).toBeVisible();

    // Test filters accordion
    await page.click("text=Filters & Search");
    await expect(page.getByPlaceholder("Search")).toBeVisible();

    // Test Analytics tab
    await page.click("text=Analytics");
    await expect(page.getByText("Bot Performance Analytics")).toBeVisible();
    await expect(page.getByText("Active Bots")).toBeVisible();
    await expect(page.getByText("Avg Latency")).toBeVisible();

    // Test Settings tab
    await page.click("text=Settings");
    await expect(page.getByText("Bot Configuration Settings")).toBeVisible();
    await expect(page.getByText("Audio Processing")).toBeVisible();
    await expect(page.getByText("Translation")).toBeVisible();
  });

  test("should handle form validation in bot creation", async () => {
    // Open create bot modal
    await page.click('[aria-label="create bot"]');

    // Try to submit without meeting ID
    await page.click('button:has-text("Next")');

    // Should show validation error
    await expect(page.getByText("Meeting ID is required")).toBeVisible();

    // Fill meeting ID and proceed
    await page.fill('[label="Google Meet ID"]', "test-meeting");
    await page.click('button:has-text("Next")');

    // Should proceed to language selection
    await expect(page.getByText("Translation Settings")).toBeVisible();

    // Deselect all languages
    await page.click("text=English");
    await page.click("text=Spanish");
    await page.click('button:has-text("Next")');

    // Should show language validation error
    await expect(
      page.getByText("At least one target language must be selected"),
    ).toBeVisible();
  });

  test("should display session database with filtering", async () => {
    // Navigate to Session Database tab
    await page.click("text=Session Database");

    // Mock session data
    await page.route("/api/bot/sessions", async (route) => {
      await route.fulfill({
        json: [
          {
            id: "session-1",
            botId: "bot-1",
            meetingId: "meeting-123",
            meetingTitle: "Test Meeting 1",
            startTime: new Date().toISOString(),
            status: "completed",
            participantCount: 3,
            totalTranslations: 50,
            qualityScore: 0.85,
          },
          {
            id: "session-2",
            botId: "bot-2",
            meetingId: "meeting-456",
            meetingTitle: "Test Meeting 2",
            startTime: new Date().toISOString(),
            status: "active",
            participantCount: 5,
            totalTranslations: 75,
            qualityScore: 0.92,
          },
        ],
      });
    });

    await page.route("/api/bot/translations", async (route) => {
      await route.fulfill({ json: [] });
    });

    await page.route("/api/bot/speaker-activity", async (route) => {
      await route.fulfill({ json: [] });
    });

    // Refresh the page to load mock data
    await page.reload();
    await page.click("text=Session Database");

    // Open filters
    await page.click("text=Filters & Search");

    // Test search functionality
    await page.fill('[placeholder="Search"]', "Test Meeting 1");

    // Test status filter
    await page.click('[aria-labelledby="status-label"]');
    await page.click("text=Completed");

    // Test date filtering
    const today = new Date().toISOString().split("T")[0];
    await page.fill('[aria-label="Date From"]', today);
  });

  test("should display analytics charts and metrics", async () => {
    // Navigate to Analytics tab
    await page.click("text=Analytics");

    // Mock analytics data
    await page.route("/api/bot/analytics*", async (route) => {
      await route.fulfill({
        json: {
          performance: [
            {
              timestamp: new Date().toISOString(),
              averageLatency: 150,
              qualityScore: 0.85,
              activeBots: 2,
              totalTranslations: 100,
              errorRate: 0.02,
            },
          ],
          languages: [
            { language: "en", count: 50, percentage: 50 },
            { language: "es", count: 30, percentage: 30 },
            { language: "fr", count: 20, percentage: 20 },
          ],
        },
      });
    });

    // Wait for charts to load
    await page.waitForTimeout(1000);

    // Check key metrics cards
    await expect(page.getByText("Active Bots")).toBeVisible();
    await expect(page.getByText("Avg Latency")).toBeVisible();
    await expect(page.getByText("Quality Score")).toBeVisible();
    await expect(page.getByText("Error Rate")).toBeVisible();

    // Test time range selector
    await page.click('[aria-labelledby="time-range-label"]');
    await page.click("text=Last 7 Days");

    // Check for chart elements
    await expect(page.getByText("Performance Trends")).toBeVisible();
    await expect(page.getByText("Language Distribution")).toBeVisible();
    await expect(page.getByText("Translation Volume")).toBeVisible();
  });

  test("should handle settings configuration", async () => {
    // Navigate to Settings tab
    await page.click("text=Settings");

    // Test Audio Processing settings
    await expect(page.getByText("Audio Processing")).toBeVisible();

    // Toggle VAD
    await page.click("text=Voice Activity Detection");

    // Test Translation settings
    await page.click("text=Translation");

    // Change translation quality
    await page.click('[aria-labelledby="translation-quality-label"]');
    await page.click("text=Accurate");

    // Test language selection
    await page.click("text=Arabic"); // Add Arabic language

    // Test Performance settings
    await page.click("text=Performance");

    // Change max concurrent bots
    await page.fill('[label="Max Concurrent Bots"]', "15");

    // Test Security settings
    await page.click("text=Security");

    // Toggle encryption
    await page.click("text=Enable Encryption");

    // Test Storage settings
    await page.click("text=Storage");

    // Change retention period
    await page.fill('[label="Retention Period (days)"]', "60");

    // Mock settings save
    await page.route("/api/bot/settings", async (route) => {
      await route.fulfill({
        json: { success: true },
      });
    });

    // Save settings
    await page.click('button:has-text("Save Settings")');

    // Should show success message
    await expect(page.getByText("Settings saved successfully")).toBeVisible();
  });

  test("should be responsive on mobile", async () => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    // Navigate to bot management
    await page.goto("/");
    await page.click('[data-testid="nav-bot-management"]');

    // Should still show main elements
    await expect(page.getByText("Bot Management Dashboard")).toBeVisible();
    await expect(page.getByText("Active Bots")).toBeVisible();

    // FAB should be visible and clickable
    await expect(page.locator('[aria-label="create bot"]')).toBeVisible();

    // Tabs should be scrollable/collapsible on mobile
    await expect(page.getByText("Virtual Webcam")).toBeVisible();
  });

  test("should handle WebSocket connection status", async () => {
    // Mock WebSocket connection
    await page.addInitScript(() => {
      class MockWebSocket {
        constructor(_url: string) {
          setTimeout(() => {
            if (this.onopen) this.onopen(new Event("open"));
          }, 100);
        }

        send() {}
        close() {}
        onopen: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent) => void) | null = null;
        onclose: ((event: CloseEvent) => void) | null = null;
        onerror: ((event: Event) => void) | null = null;
        readyState = 1;
      }

      (window as any).WebSocket = MockWebSocket;
    });

    await page.goto("/");
    await page.click('[data-testid="nav-bot-management"]');

    // Should show connection status indicator
    await expect(
      page.locator('[data-testid="connection-status"]'),
    ).toBeVisible();
  });
});
