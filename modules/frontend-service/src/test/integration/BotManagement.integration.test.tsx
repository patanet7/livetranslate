import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import BotManagement from '@/pages/BotManagement';
import {
  render,
  createTestStore,
  createMockBotInstance,
  createMockSystemStats,
  mockApiResponse,
} from '@/test/utils';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock WebSocket
const mockWebSocket = {
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  readyState: 1,
};

global.WebSocket = vi.fn(() => mockWebSocket) as any;

describe('BotManagement Integration', () => {
  let store: ReturnType<typeof createTestStore>;

  beforeEach(() => {
    store = createTestStore();
    mockFetch.mockReset();
    vi.clearAllMocks();
  });

  const renderBotManagement = () => {
    return render(<BotManagement />, { store });
  };

  describe('full bot lifecycle workflow', () => {
    it('should complete the full bot management workflow', async () => {
      const user = userEvent.setup();

      // Mock API responses
      const mockBot = createMockBotInstance();
      const mockStats = createMockSystemStats();

      mockFetch
        // Initial data load
        .mockResolvedValueOnce(
          mockApiResponse({ bots: [], activeBotIds: [] }) as Promise<Response>
        )
        .mockResolvedValueOnce(mockApiResponse(mockStats) as Promise<Response>)
        // Bot spawn
        .mockResolvedValueOnce(
          mockApiResponse({ botId: mockBot.botId, bot: mockBot }) as Promise<Response>
        )
        // Refresh after spawn
        .mockResolvedValueOnce(
          mockApiResponse({
            bots: [mockBot],
            activeBotIds: [mockBot.botId],
          }) as Promise<Response>
        )
        .mockResolvedValueOnce(mockApiResponse(mockStats) as Promise<Response>)
        // Bot termination
        .mockResolvedValueOnce(mockApiResponse({}) as Promise<Response>);

      renderBotManagement();

      // Wait for initial load
      await waitFor(() => {
        expect(screen.getByText('Bot Management Dashboard')).toBeInTheDocument();
      });

      // Should show no active bots initially
      expect(screen.getByText('0 Active')).toBeInTheDocument();

      // Open create bot modal
      const createBotFab = screen.getByLabelText('create bot');
      await user.click(createBotFab);

      // Should open create bot modal
      await waitFor(() => {
        expect(screen.getByText('Create New Bot')).toBeInTheDocument();
      });

      // Fill out the bot creation form
      const meetingIdInput = screen.getByLabelText(/Google Meet ID/i);
      await user.type(meetingIdInput, 'abc-defg-hij');

      const meetingTitleInput = screen.getByLabelText(/Meeting Title/i);
      await user.type(meetingTitleInput, 'Integration Test Meeting');

      // Navigate through the stepper
      const nextButton = screen.getByText('Next');
      await user.click(nextButton);

      // Language selection step
      expect(screen.getByText('Translation Settings')).toBeInTheDocument();
      await user.click(nextButton);

      // Advanced settings step
      expect(screen.getByText('Advanced Settings')).toBeInTheDocument();
      await user.click(nextButton);

      // Review step
      expect(screen.getByText('Ready to Create Bot')).toBeInTheDocument();

      // Create the bot
      const createButton = screen.getByText('Create Bot');
      await user.click(createButton);

      // Wait for bot creation success
      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith('/api/bot/spawn', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: expect.stringContaining('abc-defg-hij'),
        });
      });

      // Modal should close and show success notification
      await waitFor(() => {
        expect(screen.queryByText('Create New Bot')).not.toBeInTheDocument();
      });

      // Should now show 1 active bot
      await waitFor(() => {
        expect(screen.getByText('1 Active')).toBeInTheDocument();
      });

      // Should show the bot in the active bots list
      expect(screen.getByText(mockBot.meetingInfo.meetingTitle!)).toBeInTheDocument();
      expect(screen.getByText(`Bot ID: ${mockBot.botId}`)).toBeInTheDocument();

      // Test bot termination
      const moreOptionsButton = screen.getByLabelText('more');
      await user.click(moreOptionsButton);

      const terminateOption = screen.getByText('Terminate Bot');
      await user.click(terminateOption);

      // Wait for termination API call
      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(`/api/bot/${mockBot.botId}/terminate`, {
          method: 'POST',
        });
      });
    });
  });

  describe('tab navigation and functionality', () => {
    it('should navigate between tabs and show appropriate content', async () => {
      const user = userEvent.setup();

      // Mock initial API responses
      mockFetch
        .mockResolvedValueOnce(
          mockApiResponse({ bots: [], activeBotIds: [] }) as Promise<Response>
        )
        .mockResolvedValueOnce(mockApiResponse(createMockSystemStats()) as Promise<Response>);

      renderBotManagement();

      await waitFor(() => {
        expect(screen.getByText('Bot Management Dashboard')).toBeInTheDocument();
      });

      // Test Virtual Webcam tab
      const webcamTab = screen.getByText('Virtual Webcam');
      await user.click(webcamTab);

      expect(screen.getByText('Virtual Webcam Manager')).toBeInTheDocument();
      expect(screen.getByText('No Active Bots')).toBeInTheDocument();

      // Test Session Database tab
      const databaseTab = screen.getByText('Session Database');
      await user.click(databaseTab);

      expect(screen.getByText('Session Database')).toBeInTheDocument();
      expect(screen.getByText('Export All Data')).toBeInTheDocument();

      // Test Analytics tab
      const analyticsTab = screen.getByText('Analytics');
      await user.click(analyticsTab);

      expect(screen.getByText('Bot Performance Analytics')).toBeInTheDocument();

      // Test Settings tab
      const settingsTab = screen.getByText('Settings');
      await user.click(settingsTab);

      expect(screen.getByText('Bot Configuration Settings')).toBeInTheDocument();
    });
  });

  describe('real-time updates', () => {
    it('should handle real-time bot status updates', async () => {
      const mockBot = createMockBotInstance();
      const mockStats = createMockSystemStats();

      // Mock initial state with active bot
      mockFetch
        .mockResolvedValueOnce(
          mockApiResponse({
            bots: [mockBot],
            activeBotIds: [mockBot.botId],
          }) as Promise<Response>
        )
        .mockResolvedValueOnce(mockApiResponse(mockStats) as Promise<Response>);

      renderBotManagement();

      await waitFor(() => {
        expect(screen.getByText(mockBot.meetingInfo.meetingTitle!)).toBeInTheDocument();
      });

      // Simulate WebSocket message for bot status update
      const websocketMessage = {
        type: 'bot:status_change',
        data: {
          botId: mockBot.botId,
          status: 'error',
          errorCount: 5,
        },
      };

      // Trigger WebSocket message handler
      const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
        call => call[0] === 'message'
      )?.[1];

      if (messageHandler) {
        messageHandler({ data: JSON.stringify(websocketMessage) });
      }

      // Should update the bot status in the UI
      await waitFor(() => {
        expect(screen.getByText('error')).toBeInTheDocument();
      });
    });
  });

  describe('error handling', () => {
    it('should handle API errors gracefully', async () => {
      // Mock API failure
      mockFetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce(mockApiResponse(createMockSystemStats()) as Promise<Response>);

      renderBotManagement();

      // Should show error state but not crash
      await waitFor(() => {
        expect(screen.getByText('Bot Management Dashboard')).toBeInTheDocument();
      });

      // Error notification should be shown
      await waitFor(() => {
        expect(screen.getByText(/Network error/i)).toBeInTheDocument();
      });
    });

    it('should handle bot spawn failures', async () => {
      const user = userEvent.setup();

      // Mock successful initial load
      mockFetch
        .mockResolvedValueOnce(
          mockApiResponse({ bots: [], activeBotIds: [] }) as Promise<Response>
        )
        .mockResolvedValueOnce(mockApiResponse(createMockSystemStats()) as Promise<Response>)
        // Mock bot spawn failure
        .mockResolvedValueOnce({
          ok: false,
          json: () => Promise.resolve({ message: 'Meeting not found' }),
        } as Response);

      renderBotManagement();

      await waitFor(() => {
        expect(screen.getByText('Bot Management Dashboard')).toBeInTheDocument();
      });

      // Try to create a bot
      const createBotFab = screen.getByLabelText('create bot');
      await user.click(createBotFab);

      // Fill minimal form and submit
      const meetingIdInput = screen.getByLabelText(/Google Meet ID/i);
      await user.type(meetingIdInput, 'invalid-meeting-id');

      // Navigate to final step
      const nextButton = screen.getByText('Next');
      await user.click(nextButton); // Language step
      await user.click(screen.getByText('Next')); // Settings step
      await user.click(screen.getByText('Next')); // Review step

      const createButton = screen.getByText('Create Bot');
      await user.click(createButton);

      // Should show error notification
      await waitFor(() => {
        expect(screen.getByText(/Meeting not found/i)).toBeInTheDocument();
      });
    });
  });

  describe('data export functionality', () => {
    it('should export bot data successfully', async () => {
      const user = userEvent.setup();

      // Mock blob response
      const mockBlob = new Blob(['export data'], { type: 'application/json' });
      
      mockFetch
        .mockResolvedValueOnce(
          mockApiResponse({ bots: [], activeBotIds: [] }) as Promise<Response>
        )
        .mockResolvedValueOnce(mockApiResponse(createMockSystemStats()) as Promise<Response>)
        .mockResolvedValueOnce({
          ok: true,
          blob: () => Promise.resolve(mockBlob),
        } as Response);

      // Mock DOM methods for download
      const mockCreateObjectURL = vi.fn(() => 'mock-url');
      const mockRevokeObjectURL = vi.fn();
      const mockClick = vi.fn();

      global.URL.createObjectURL = mockCreateObjectURL;
      global.URL.revokeObjectURL = mockRevokeObjectURL;
      
      const mockAnchor = {
        href: '',
        download: '',
        click: mockClick,
      };
      
      vi.spyOn(document, 'createElement').mockReturnValue(mockAnchor as any);
      vi.spyOn(document.body, 'appendChild').mockImplementation(vi.fn());
      vi.spyOn(document.body, 'removeChild').mockImplementation(vi.fn());

      renderBotManagement();

      await waitFor(() => {
        expect(screen.getByText('Bot Management Dashboard')).toBeInTheDocument();
      });

      // Navigate to Session Database tab
      const databaseTab = screen.getByText('Session Database');
      await user.click(databaseTab);

      // Click export button
      const exportButton = screen.getByText('Export All Data');
      await user.click(exportButton);

      // Should trigger download
      expect(mockCreateObjectURL).toHaveBeenCalledWith(mockBlob);
      expect(mockClick).toHaveBeenCalled();
    });
  });

  describe('responsive design', () => {
    it('should adapt to mobile viewport', async () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      // Mock matchMedia for mobile
      window.matchMedia = vi.fn().mockImplementation(query => ({
        matches: query.includes('max-width'),
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }));

      mockFetch
        .mockResolvedValueOnce(
          mockApiResponse({ bots: [], activeBotIds: [] }) as Promise<Response>
        )
        .mockResolvedValueOnce(mockApiResponse(createMockSystemStats()) as Promise<Response>);

      renderBotManagement();

      await waitFor(() => {
        expect(screen.getByText('Bot Management Dashboard')).toBeInTheDocument();
      });

      // Should still render all essential elements
      expect(screen.getByText('Active Bots')).toBeInTheDocument();
      expect(screen.getByText('Virtual Webcam')).toBeInTheDocument();
      expect(screen.getByLabelText('create bot')).toBeInTheDocument();
    });
  });
});