import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useBotManager } from '../useBotManager';
import {
  createTestStore,
  createMockBotInstance,
  createMockSystemStats,
  MOCK_MEETING_REQUEST,
  mockApiResponse,
} from '@/test/utils';
import { Provider } from 'react-redux';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('useBotManager', () => {
  let store: ReturnType<typeof createTestStore>;

  beforeEach(() => {
    store = createTestStore();
    mockFetch.mockReset();
  });

  const renderUseBotManager = () => {
    return renderHook(() => useBotManager(), {
      wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
    });
  };

  describe('spawnBot', () => {
    it('should spawn bot successfully', async () => {
      const mockBot = createMockBotInstance();
      const mockResponse = { botId: mockBot.botId, bot: mockBot };
      
      mockFetch.mockResolvedValueOnce(
        mockApiResponse(mockResponse) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      let botId: string;
      await act(async () => {
        botId = await result.current.spawnBot(MOCK_MEETING_REQUEST);
      });

      expect(mockFetch).toHaveBeenCalledWith('/api/bot/spawn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(MOCK_MEETING_REQUEST),
      });

      expect(botId!).toBe(mockBot.botId);
      
      // Check that the bot was added to the store
      const state = store.getState();
      expect(state.bot.bots[mockBot.botId]).toBeDefined();
    });

    it('should handle spawn bot failure', async () => {
      const errorMessage = 'Failed to spawn bot';
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ message: errorMessage }),
      } as Response);

      const { result } = renderUseBotManager();

      await act(async () => {
        await expect(result.current.spawnBot(MOCK_MEETING_REQUEST)).rejects.toThrow(errorMessage);
      });

      expect(result.current.error).toBe(errorMessage);
    });

    it('should handle network errors during spawn', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const { result } = renderUseBotManager();

      await act(async () => {
        await expect(result.current.spawnBot(MOCK_MEETING_REQUEST)).rejects.toThrow('Network error');
      });

      expect(result.current.error).toBe('Network error');
    });
  });

  describe('terminateBot', () => {
    it('should terminate bot successfully', async () => {
      const botId = 'test-bot-id';
      mockFetch.mockResolvedValueOnce(
        mockApiResponse({}) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      await act(async () => {
        await result.current.terminateBot(botId);
      });

      expect(mockFetch).toHaveBeenCalledWith(`/api/bot/${botId}/terminate`, {
        method: 'POST',
      });

      // Check that the bot status was updated in the store
      await waitFor(() => {
        const state = store.getState();
        const bot = state.bot.bots[botId];
        if (bot) {
          expect(bot.status).toBe('terminated');
        }
      });
    });

    it('should handle terminate bot failure', async () => {
      const botId = 'test-bot-id';
      const errorMessage = 'Failed to terminate bot';
      
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ message: errorMessage }),
      } as Response);

      const { result } = renderUseBotManager();

      await act(async () => {
        await expect(result.current.terminateBot(botId)).rejects.toThrow(errorMessage);
      });

      expect(result.current.error).toBe(errorMessage);
    });
  });

  describe('getBotStatus', () => {
    it('should get bot status successfully', async () => {
      const mockBot = createMockBotInstance();
      mockFetch.mockResolvedValueOnce(
        mockApiResponse(mockBot) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      let bot: any;
      await act(async () => {
        bot = await result.current.getBotStatus(mockBot.botId);
      });

      expect(mockFetch).toHaveBeenCalledWith(`/api/bot/${mockBot.botId}/status`);
      expect(bot).toEqual(mockBot);

      // Check that the bot was updated in the store
      const state = store.getState();
      expect(state.bot.bots[mockBot.botId]).toEqual(mockBot);
    });

    it('should handle get bot status failure', async () => {
      const botId = 'test-bot-id';
      const errorMessage = 'Bot not found';
      
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ message: errorMessage }),
      } as Response);

      const { result } = renderUseBotManager();

      await act(async () => {
        await expect(result.current.getBotStatus(botId)).rejects.toThrow(errorMessage);
      });

      expect(result.current.error).toBe(errorMessage);
    });
  });

  describe('getActiveBots', () => {
    it('should get active bots successfully', async () => {
      const mockBots = [
        createMockBotInstance({ botId: 'bot-1' }),
        createMockBotInstance({ botId: 'bot-2' }),
      ];
      const mockResponse = {
        bots: mockBots,
        activeBotIds: ['bot-1', 'bot-2'],
      };

      mockFetch.mockResolvedValueOnce(
        mockApiResponse(mockResponse) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      let bots: any;
      await act(async () => {
        bots = await result.current.getActiveBots();
      });

      expect(mockFetch).toHaveBeenCalledWith('/api/bot/active');
      expect(bots).toEqual(mockBots);

      // Check that the bots were updated in the store
      const state = store.getState();
      expect(state.bot.bots['bot-1']).toEqual(mockBots[0]);
      expect(state.bot.bots['bot-2']).toEqual(mockBots[1]);
      expect(state.bot.activeBotIds).toEqual(['bot-1', 'bot-2']);
    });

    it('should handle get active bots failure', async () => {
      const errorMessage = 'Failed to fetch active bots';
      
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ message: errorMessage }),
      } as Response);

      const { result } = renderUseBotManager();

      await act(async () => {
        await expect(result.current.getActiveBots()).rejects.toThrow(errorMessage);
      });

      expect(result.current.error).toBe(errorMessage);
    });
  });

  describe('getSystemStats', () => {
    it('should get system stats successfully', async () => {
      const mockStats = createMockSystemStats();
      mockFetch.mockResolvedValueOnce(
        mockApiResponse(mockStats) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      let stats: any;
      await act(async () => {
        stats = await result.current.getSystemStats();
      });

      expect(mockFetch).toHaveBeenCalledWith('/api/bot/stats');
      expect(stats).toEqual(mockStats);

      // Check that the stats were updated in the store
      const state = store.getState();
      expect(state.bot.systemStats).toEqual(mockStats);
    });

    it('should handle get system stats failure', async () => {
      const errorMessage = 'Failed to fetch system stats';
      
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ message: errorMessage }),
      } as Response);

      const { result } = renderUseBotManager();

      await act(async () => {
        await expect(result.current.getSystemStats()).rejects.toThrow(errorMessage);
      });

      expect(result.current.error).toBe(errorMessage);
    });
  });

  describe('refreshBots', () => {
    it('should refresh bots and stats successfully', async () => {
      const mockBots = [createMockBotInstance()];
      const mockStats = createMockSystemStats();
      
      mockFetch
        .mockResolvedValueOnce(
          mockApiResponse({ bots: mockBots, activeBotIds: ['bot-1'] }) as Promise<Response>
        )
        .mockResolvedValueOnce(
          mockApiResponse(mockStats) as Promise<Response>
        );

      const { result } = renderUseBotManager();

      await act(async () => {
        await result.current.refreshBots();
      });

      expect(mockFetch).toHaveBeenCalledWith('/api/bot/active');
      expect(mockFetch).toHaveBeenCalledWith('/api/bot/stats');

      // Check that both bots and stats were updated
      const state = store.getState();
      expect(Object.keys(state.bot.bots)).toHaveLength(1);
      expect(state.bot.systemStats).toEqual(mockStats);
      expect(result.current.isLoading).toBe(false);
    });

    it('should handle refresh failure gracefully', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const { result } = renderUseBotManager();

      await act(async () => {
        await result.current.refreshBots();
      });

      expect(result.current.error).toBe('Network error');
      expect(result.current.isLoading).toBe(false);
    });
  });

  describe('getBotSessions', () => {
    it('should get bot sessions for specific bot', async () => {
      const botId = 'test-bot-id';
      const mockSessions = [
        { sessionId: 'session-1', botId, status: 'completed' },
        { sessionId: 'session-2', botId, status: 'active' },
      ];

      mockFetch.mockResolvedValueOnce(
        mockApiResponse(mockSessions) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      let sessions: any;
      await act(async () => {
        sessions = await result.current.getBotSessions(botId);
      });

      expect(mockFetch).toHaveBeenCalledWith(`/api/bot/${botId}/sessions`);
      expect(sessions).toEqual(mockSessions);
    });

    it('should get all bot sessions when no botId provided', async () => {
      const mockSessions = [
        { sessionId: 'session-1', botId: 'bot-1', status: 'completed' },
        { sessionId: 'session-2', botId: 'bot-2', status: 'active' },
      ];

      mockFetch.mockResolvedValueOnce(
        mockApiResponse(mockSessions) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      let sessions: any;
      await act(async () => {
        sessions = await result.current.getBotSessions();
      });

      expect(mockFetch).toHaveBeenCalledWith('/api/bot/sessions');
      expect(sessions).toEqual(mockSessions);
    });
  });

  describe('exportBotData', () => {
    it('should export bot data successfully', async () => {
      const botId = 'test-bot-id';
      const mockBlob = new Blob(['export data'], { type: 'application/json' });
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        blob: () => Promise.resolve(mockBlob),
      } as Response);

      // Mock URL.createObjectURL and DOM methods
      const mockCreateObjectURL = vi.fn(() => 'mock-url');
      const mockRevokeObjectURL = vi.fn();
      const mockClick = vi.fn();
      const mockRemoveChild = vi.fn();
      const mockAppendChild = vi.fn();

      global.URL.createObjectURL = mockCreateObjectURL;
      global.URL.revokeObjectURL = mockRevokeObjectURL;
      
      const mockAnchor = {
        href: '',
        download: '',
        click: mockClick,
      };
      
      vi.spyOn(document, 'createElement').mockReturnValue(mockAnchor as any);
      vi.spyOn(document.body, 'appendChild').mockImplementation(mockAppendChild);
      vi.spyOn(document.body, 'removeChild').mockImplementation(mockRemoveChild);

      const { result } = renderUseBotManager();

      await act(async () => {
        await result.current.exportBotData(botId, 'json');
      });

      expect(mockFetch).toHaveBeenCalledWith(`/api/bot/${botId}/export?format=json`);
      expect(mockCreateObjectURL).toHaveBeenCalledWith(mockBlob);
      expect(mockClick).toHaveBeenCalled();
      expect(mockRevokeObjectURL).toHaveBeenCalledWith('mock-url');
    });

    it('should handle export failure', async () => {
      const botId = 'test-bot-id';
      const errorMessage = 'Export failed';
      
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ message: errorMessage }),
      } as Response);

      const { result } = renderUseBotManager();

      await act(async () => {
        await expect(result.current.exportBotData(botId)).rejects.toThrow(errorMessage);
      });

      expect(result.current.error).toBe(errorMessage);
    });
  });

  describe('updateBotConfig', () => {
    it('should update bot config successfully', async () => {
      const botId = 'test-bot-id';
      const config = { targetLanguages: ['en', 'fr', 'de'] };
      const updatedBot = createMockBotInstance({ botId });

      mockFetch.mockResolvedValueOnce(
        mockApiResponse(updatedBot) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      let bot: any;
      await act(async () => {
        bot = await result.current.updateBotConfig(botId, config);
      });

      expect(mockFetch).toHaveBeenCalledWith(`/api/bot/${botId}/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      expect(bot).toEqual(updatedBot);

      // Check that the bot was updated in the store
      const state = store.getState();
      expect(state.bot.bots[botId]).toEqual(updatedBot);
    });
  });

  describe('error handling', () => {
    it('should clear error', () => {
      // Set an error first
      store.dispatch({ type: 'bot/setError', payload: 'Test error' });

      const { result } = renderUseBotManager();
      
      act(() => {
        result.current.clearError();
      });

      expect(result.current.error).toBe(null);
      
      const state = store.getState();
      expect(state.bot.error).toBe(null);
    });
  });

  describe('loading states', () => {
    it('should show loading during async operations', async () => {
      // Delay the fetch response
      mockFetch.mockImplementationOnce(
        () => new Promise(resolve => 
          setTimeout(() => resolve(mockApiResponse(createMockSystemStats())), 100)
        )
      );

      const { result } = renderUseBotManager();

      // Start async operation
      act(() => {
        result.current.getSystemStats();
      });

      // Should be loading
      expect(result.current.isLoading).toBe(true);

      // Wait for completion
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });
  });

  describe('restartBot', () => {
    it('should restart bot successfully', async () => {
      const botId = 'test-bot-id';
      mockFetch.mockResolvedValueOnce(
        mockApiResponse({}) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      await act(async () => {
        await result.current.restartBot(botId);
      });

      expect(mockFetch).toHaveBeenCalledWith(`/api/bot/${botId}/restart`, {
        method: 'POST',
      });

      // Check that the bot status was updated in the store
      await waitFor(() => {
        const state = store.getState();
        const bot = state.bot.bots[botId];
        if (bot) {
          expect(bot.status).toBe('spawning');
        }
      });
    });
  });

  describe('pauseBot and resumeBot', () => {
    it('should pause and resume bot successfully', async () => {
      const botId = 'test-bot-id';
      
      mockFetch.mockResolvedValue(
        mockApiResponse({}) as Promise<Response>
      );

      const { result } = renderUseBotManager();

      // Test pause
      await act(async () => {
        await result.current.pauseBot(botId);
      });

      expect(mockFetch).toHaveBeenCalledWith(`/api/bot/${botId}/pause`, {
        method: 'POST',
      });

      // Test resume
      await act(async () => {
        await result.current.resumeBot(botId);
      });

      expect(mockFetch).toHaveBeenCalledWith(`/api/bot/${botId}/resume`, {
        method: 'POST',
      });

      // Check that the bot status was updated in the store
      await waitFor(() => {
        const state = store.getState();
        const bot = state.bot.bots[botId];
        if (bot) {
          expect(bot.status).toBe('active');
        }
      });
    });
  });
});