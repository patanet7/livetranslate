import { describe, it, expect, beforeEach } from "vitest";
import botSlice, {
  spawnBot,
  spawnBotSuccess,
  spawnBotFailure,
  updateBotStatus,
  terminateBot,
  setBots,
  setActiveBotIds,
  setSystemStats,
  addBot,
  updateBot,
  removeBot,
  addTranslation,
  updateAudioCapture,
  updateWebcamStatus,
  setError,
  clearBotError,
  resetBotState,
} from "../botSlice";
import {
  createMockBotInstance,
  createMockTranslation,
  createMockSystemStats,
  MOCK_MEETING_REQUEST,
} from "@/test/utils";
import { AudioQualityMetrics } from "@/types";

describe("botSlice", () => {
  const initialState = botSlice.getInitialState();

  beforeEach(() => {
    // Reset to initial state before each test
  });

  describe("initial state", () => {
    it("should have correct initial state", () => {
      expect(initialState).toEqual({
        bots: {},
        activeBotIds: [],
        spawnerConfig: {
          maxConcurrentBots: 10,
          defaultTargetLanguages: ["en", "es", "fr"],
          autoTranslationEnabled: true,
          virtualWebcamEnabled: true,
          defaultBotConfig: expect.any(Object),
        },
        meetingRequests: {},
        systemStats: {
          totalBotsSpawned: 0,
          activeBots: 0,
          completedSessions: 0,
          errorRate: 0,
          averageSessionDuration: 0,
        },
        healthMetrics: {},
        realtimeData: {
          audioCapture: {},
          captions: {},
          translations: {},
          webcamFrames: {},
        },
        selectedBotId: null,
        dashboardView: "overview",
        error: null,
        loading: false,
      });
    });
  });

  describe("bot lifecycle management", () => {
    it("should spawn bot request", () => {
      const state = botSlice.reducer(
        initialState,
        spawnBot(MOCK_MEETING_REQUEST),
      );

      expect(state.loading).toBe(true);
      expect(Object.keys(state.meetingRequests)).toHaveLength(1);

      const requestId = Object.keys(state.meetingRequests)[0];
      const request = state.meetingRequests[requestId];

      expect(request.meetingId).toBe(MOCK_MEETING_REQUEST.meetingId);
      expect(request.status).toBe("pending");
      expect(request.createdAt).toBeTypeOf("number");
    });

    it("should handle successful bot spawn", () => {
      // First create a spawn request
      const stateWithRequest = botSlice.reducer(
        initialState,
        spawnBot(MOCK_MEETING_REQUEST),
      );
      const requestId = Object.keys(stateWithRequest.meetingRequests)[0];

      const botData = createMockBotInstance();
      const state = botSlice.reducer(
        stateWithRequest,
        spawnBotSuccess({ requestId, botId: botData.botId, botData }),
      );

      expect(state.loading).toBe(false);
      expect(state.bots[botData.botId]).toBeDefined();
      expect(state.activeBotIds).toContain(botData.botId);
      expect(state.systemStats.totalBotsSpawned).toBe(1);
      expect(state.systemStats.activeBots).toBe(1);
      expect(state.meetingRequests[requestId].status).toBe("completed");
    });

    it("should handle failed bot spawn", () => {
      const stateWithRequest = botSlice.reducer(
        initialState,
        spawnBot(MOCK_MEETING_REQUEST),
      );
      const requestId = Object.keys(stateWithRequest.meetingRequests)[0];

      const error = "Failed to connect to meeting";
      const state = botSlice.reducer(
        stateWithRequest,
        spawnBotFailure({ requestId, error }),
      );

      expect(state.loading).toBe(false);
      expect(state.error).toBe(error);
      expect(state.meetingRequests[requestId].status).toBe("failed");
    });

    it("should update bot status", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
      };

      const state = botSlice.reducer(
        stateWithBot,
        updateBotStatus({
          botId: bot.botId,
          status: "error",
          data: { performance: { errorCount: 5 } },
        }),
      );

      expect(state.bots[bot.botId].status).toBe("error");
      expect(state.bots[bot.botId].performance.errorCount).toBe(5);
      expect(
        new Date(state.bots[bot.botId].lastActiveAt).getTime(),
      ).toBeGreaterThan(new Date(bot.lastActiveAt).getTime());
    });

    it("should terminate bot", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
        activeBotIds: [bot.botId],
        systemStats: { ...initialState.systemStats, activeBots: 1 },
        realtimeData: {
          ...initialState.realtimeData,
          audioCapture: { [bot.botId]: {} as AudioQualityMetrics },
          translations: { [bot.botId]: [] },
        },
      };

      const state = botSlice.reducer(stateWithBot, terminateBot(bot.botId));

      expect(state.bots[bot.botId].status).toBe("terminated");
      expect(state.activeBotIds).not.toContain(bot.botId);
      expect(state.systemStats.activeBots).toBe(0);
      expect(state.systemStats.completedSessions).toBe(1);
      expect(state.realtimeData.audioCapture[bot.botId]).toBeUndefined();
      expect(state.realtimeData.translations[bot.botId]).toBeUndefined();
    });

    it("should set bots", () => {
      const bots = {
        "bot-1": createMockBotInstance({ botId: "bot-1" }),
        "bot-2": createMockBotInstance({ botId: "bot-2" }),
      };

      const state = botSlice.reducer(initialState, setBots(bots));

      expect(state.bots).toEqual(bots);
    });

    it("should set active bot IDs", () => {
      const activeBotIds = ["bot-1", "bot-2", "bot-3"];
      const state = botSlice.reducer(
        initialState,
        setActiveBotIds(activeBotIds),
      );

      expect(state.activeBotIds).toEqual(activeBotIds);
    });

    it("should add bot", () => {
      const bot = createMockBotInstance();
      const state = botSlice.reducer(initialState, addBot(bot));

      expect(state.bots[bot.botId]).toEqual(bot);
      expect(state.activeBotIds).toContain(bot.botId);
    });

    it("should not duplicate bot ID in activeBotIds when adding existing bot", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
        activeBotIds: [bot.botId],
      };

      const updatedBot = { ...bot, status: "active" as const };
      const state = botSlice.reducer(stateWithBot, addBot(updatedBot));

      expect(state.activeBotIds).toEqual([bot.botId]); // Should not duplicate
      expect(state.bots[bot.botId]).toEqual(updatedBot);
    });

    it("should update bot", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
      };

      const updates = {
        status: "error" as const,
        performance: { ...bot.performance, errorCount: 10 },
      };

      const state = botSlice.reducer(
        stateWithBot,
        updateBot({ botId: bot.botId, updates }),
      );

      expect(state.bots[bot.botId].status).toBe("error");
      expect(state.bots[bot.botId].performance.errorCount).toBe(10);
    });

    it("should remove bot", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
        activeBotIds: [bot.botId],
      };

      const state = botSlice.reducer(stateWithBot, removeBot(bot.botId));

      expect(state.bots[bot.botId]).toBeUndefined();
      expect(state.activeBotIds).not.toContain(bot.botId);
    });
  });

  describe("audio capture updates", () => {
    it("should update audio capture metrics", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
      };

      const metrics: AudioQualityMetrics = {
        rmsLevel: -15,
        peakLevel: -5,
        qualityScore: 0.9,
      };

      const state = botSlice.reducer(
        stateWithBot,
        updateAudioCapture({ botId: bot.botId, metrics }),
      );

      expect(state.bots[bot.botId].audioCapture.totalChunksCaptured).toBe(
        bot.audioCapture.totalChunksCaptured + 1,
      );
      expect(
        state.bots[bot.botId].audioCapture.averageQualityScore,
      ).toBeCloseTo(
        (bot.audioCapture.averageQualityScore + metrics.qualityScore!) / 2,
      );
      expect(state.realtimeData.audioCapture[bot.botId]).toEqual(metrics);
    });

    it("should set audio capture status", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
      };

      const state = botSlice.reducer(
        stateWithBot,
        botSlice.actions.setAudioCaptureStatus({
          botId: bot.botId,
          isCapturing: false,
          deviceInfo: "New Device",
        }),
      );

      expect(state.bots[bot.botId].audioCapture.isCapturing).toBe(false);
      expect(state.bots[bot.botId].audioCapture.deviceInfo).toBe("New Device");
    });
  });

  describe("translation management", () => {
    it("should add translation", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
      };

      const translation = createMockTranslation();
      const state = botSlice.reducer(
        stateWithBot,
        addTranslation({ botId: bot.botId, translation }),
      );

      // virtualWebcam stores simplified { language, text } objects
      expect(
        state.bots[bot.botId].virtualWebcam.currentTranslations,
      ).toContainEqual({
        language: translation.targetLanguage,
        text: translation.translatedText,
      });
      // realtimeData stores full Translation objects
      expect(state.realtimeData.translations[bot.botId]).toContainEqual(
        translation,
      );
    });

    it("should limit current translations to 3 for display", () => {
      const bot = createMockBotInstance({
        virtualWebcam: {
          ...createMockBotInstance().virtualWebcam,
          currentTranslations: [
            { language: "es", text: "Translation 1" },
            { language: "fr", text: "Translation 2" },
            { language: "de", text: "Translation 3" },
          ],
        },
      });

      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
      };

      const newTranslation = createMockTranslation();
      const state = botSlice.reducer(
        stateWithBot,
        addTranslation({ botId: bot.botId, translation: newTranslation }),
      );

      expect(
        state.bots[bot.botId].virtualWebcam.currentTranslations,
      ).toHaveLength(3);
      expect(
        state.bots[bot.botId].virtualWebcam.currentTranslations[2].text,
      ).toContain("mock translation");
      expect(
        state.bots[bot.botId].virtualWebcam.currentTranslations.find(
          (t) => t.text === "Translation 1",
        ),
      ).toBeUndefined(); // First translation should be removed
    });

    it("should limit realtime translations to 100", () => {
      const bot = createMockBotInstance();
      const stateWithTranslations = {
        ...initialState,
        bots: { [bot.botId]: bot },
        realtimeData: {
          ...initialState.realtimeData,
          translations: {
            [bot.botId]: Array.from({ length: 100 }, (_, i) =>
              createMockTranslation({ translationId: i.toString() }),
            ),
          },
        },
      };

      const newTranslation = createMockTranslation({ translationId: "100" });
      const state = botSlice.reducer(
        stateWithTranslations,
        addTranslation({ botId: bot.botId, translation: newTranslation }),
      );

      expect(state.realtimeData.translations[bot.botId]).toHaveLength(100);
      expect(state.realtimeData.translations[bot.botId][99]).toEqual(
        newTranslation,
      );
      expect(
        state.realtimeData.translations[bot.botId].find(
          (t) => t.translationId === "0",
        ),
      ).toBeUndefined(); // First translation should be removed
    });
  });

  describe("virtual webcam management", () => {
    it("should update webcam status", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
      };

      const state = botSlice.reducer(
        stateWithBot,
        updateWebcamStatus({
          botId: bot.botId,
          isStreaming: false,
          framesGenerated: 2000,
        }),
      );

      expect(state.bots[bot.botId].virtualWebcam.isStreaming).toBe(false);
      expect(state.bots[bot.botId].virtualWebcam.framesGenerated).toBe(2000);
    });

    it("should update webcam frame", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
      };

      const frameBase64 = "base64-encoded-frame-data";
      const state = botSlice.reducer(
        stateWithBot,
        botSlice.actions.updateWebcamFrame({ botId: bot.botId, frameBase64 }),
      );

      expect(state.bots[bot.botId].virtualWebcam.framesGenerated).toBe(
        bot.virtualWebcam.framesGenerated + 1,
      );
      expect(state.realtimeData.webcamFrames[bot.botId]).toBe(frameBase64);
    });

    it("should update webcam config", () => {
      const bot = createMockBotInstance();
      const stateWithBot = {
        ...initialState,
        bots: { [bot.botId]: bot },
      };

      const configUpdate = { width: 1920, height: 1080, fps: 60 };
      const state = botSlice.reducer(
        stateWithBot,
        botSlice.actions.updateWebcamConfig({
          botId: bot.botId,
          config: configUpdate,
        }),
      );

      expect(state.bots[bot.botId].virtualWebcam.webcamConfig.width).toBe(1920);
      expect(state.bots[bot.botId].virtualWebcam.webcamConfig.height).toBe(
        1080,
      );
      expect(state.bots[bot.botId].virtualWebcam.webcamConfig.fps).toBe(60);
      // Should preserve other config values
      expect(state.bots[bot.botId].virtualWebcam.webcamConfig.displayMode).toBe(
        "overlay",
      );
    });
  });

  describe("system statistics", () => {
    it("should set system stats", () => {
      const stats = createMockSystemStats({
        totalBotsSpawned: 25,
        activeBots: 5,
        completedSessions: 20,
        errorRate: 0.02,
        averageSessionDuration: 4500,
      });

      const state = botSlice.reducer(initialState, setSystemStats(stats));

      expect(state.systemStats).toEqual(stats);
    });

    it("should update system stats", () => {
      const initialStats = createMockSystemStats();
      const stateWithStats = {
        ...initialState,
        systemStats: initialStats,
      };

      const updates = { errorRate: 0.1, averageSessionDuration: 5000 };
      const state = botSlice.reducer(
        stateWithStats,
        botSlice.actions.updateSystemStats(updates),
      );

      expect(state.systemStats.errorRate).toBe(0.1);
      expect(state.systemStats.averageSessionDuration).toBe(5000);
      // Should preserve other stats
      expect(state.systemStats.totalBotsSpawned).toBe(
        initialStats.totalBotsSpawned,
      );
      expect(state.systemStats.activeBots).toBe(initialStats.activeBots);
    });
  });

  describe("error handling", () => {
    it("should set error", () => {
      const errorMessage = "Bot management error";
      const state = botSlice.reducer(initialState, setError(errorMessage));

      expect(state.error).toBe(errorMessage);
    });

    it("should clear error", () => {
      const errorState = { ...initialState, error: "Some error" };
      const state = botSlice.reducer(errorState, clearBotError());

      expect(state.error).toBe(null);
    });

    it("should set loading state", () => {
      const state = botSlice.reducer(
        initialState,
        botSlice.actions.setLoading(true),
      );

      expect(state.loading).toBe(true);
    });
  });

  describe("UI state management", () => {
    it("should set selected bot", () => {
      const botId = "selected-bot-id";
      const state = botSlice.reducer(
        initialState,
        botSlice.actions.setSelectedBot(botId),
      );

      expect(state.selectedBotId).toBe(botId);
    });

    it("should set dashboard view", () => {
      const view = "detailed";
      const state = botSlice.reducer(
        initialState,
        botSlice.actions.setDashboardView(view),
      );

      expect(state.dashboardView).toBe(view);
    });
  });

  describe("state reset", () => {
    it("should reset to initial state", () => {
      const modifiedState = {
        ...initialState,
        bots: { "bot-1": createMockBotInstance() },
        activeBotIds: ["bot-1"],
        error: "Some error",
        loading: true,
      };

      const state = botSlice.reducer(modifiedState, resetBotState());

      expect(state).toEqual(initialState);
    });
  });
});
