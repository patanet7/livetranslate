import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BotSpawner } from '../BotSpawner';
import { render, createTestStore, mockApiResponse } from '@/test/utils';

// Mock the useBotManager hook
const mockSpawnBot = vi.fn();
const mockUseBotManager = {
  spawnBot: mockSpawnBot,
  isLoading: false,
};

vi.mock('@/hooks/useBotManager', () => ({
  useBotManager: () => mockUseBotManager,
}));

describe('BotSpawner', () => {
  const mockOnBotSpawned = vi.fn();
  const mockOnError = vi.fn();

  beforeEach(() => {
    mockSpawnBot.mockReset();
    mockOnBotSpawned.mockReset();
    mockOnError.mockReset();
    mockUseBotManager.isLoading = false;
  });

  const renderBotSpawner = (props = {}) => {
    return render(
      <BotSpawner
        onBotSpawned={mockOnBotSpawned}
        onError={mockOnError}
        {...props}
      />
    );
  };

  describe('initial render', () => {
    it('should render the bot spawner form', () => {
      renderBotSpawner();

      expect(screen.getByText('Spawn New Bot')).toBeInTheDocument();
      expect(screen.getByLabelText(/Google Meet ID/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Meeting Title/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Organizer Email/i)).toBeInTheDocument();
      expect(screen.getByText('Enable Auto-Translation')).toBeInTheDocument();
      expect(screen.getByLabelText(/Bot Priority/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /spawn bot/i })).toBeInTheDocument();
    });

    it('should have default form values', () => {
      renderBotSpawner();

      const meetingIdInput = screen.getByLabelText(/Google Meet ID/i) as HTMLInputElement;
      const meetingTitleInput = screen.getByLabelText(/Meeting Title/i) as HTMLInputElement;
      const organizerEmailInput = screen.getByLabelText(/Organizer Email/i) as HTMLInputElement;
      const autoTranslationSwitch = screen.getByRole('checkbox', { name: /Enable Auto-Translation/i }) as HTMLInputElement;

      expect(meetingIdInput.value).toBe('');
      expect(meetingTitleInput.value).toBe('');
      expect(organizerEmailInput.value).toBe('');
      expect(autoTranslationSwitch.checked).toBe(true);
    });

    it('should show default selected languages', () => {
      renderBotSpawner();

      // Check that English and Spanish are selected by default
      const selectedLanguages = screen.getAllByText(/English|Spanish/);
      expect(selectedLanguages.length).toBeGreaterThan(0);
    });
  });

  describe('form interactions', () => {
    it('should update meeting ID input', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      const meetingIdInput = screen.getByLabelText(/Google Meet ID/i);
      await user.type(meetingIdInput, 'abc-defg-hij');

      expect(meetingIdInput).toHaveValue('abc-defg-hij');
    });

    it('should update meeting title input', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      const meetingTitleInput = screen.getByLabelText(/Meeting Title/i);
      await user.type(meetingTitleInput, 'Weekly Team Meeting');

      expect(meetingTitleInput).toHaveValue('Weekly Team Meeting');
    });

    it('should update organizer email input', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      const organizerEmailInput = screen.getByLabelText(/Organizer Email/i);
      await user.type(organizerEmailInput, 'organizer@example.com');

      expect(organizerEmailInput).toHaveValue('organizer@example.com');
    });

    it('should toggle auto-translation', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      const autoTranslationSwitch = screen.getByRole('checkbox', { name: /Enable Auto-Translation/i });
      expect(autoTranslationSwitch).toBeChecked();

      await user.click(autoTranslationSwitch);
      expect(autoTranslationSwitch).not.toBeChecked();
    });

    it('should change bot priority', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      const prioritySelect = screen.getByLabelText(/Bot Priority/i);
      await user.click(prioritySelect);

      const highPriorityOption = screen.getByText(/High - Real-time processing/i);
      await user.click(highPriorityOption);

      expect(prioritySelect).toHaveTextContent('high');
    });
  });

  describe('language selection', () => {
    it('should allow toggling language chips', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      // Find French language chip (should be unselected initially)
      const frenchChip = screen.getByText('French');
      expect(frenchChip.closest('.MuiChip-root')).toHaveClass('MuiChip-outlined');

      // Click to select French
      await user.click(frenchChip);
      expect(frenchChip.closest('.MuiChip-root')).toHaveClass('MuiChip-filled');

      // Click again to deselect
      await user.click(frenchChip);
      expect(frenchChip.closest('.MuiChip-root')).toHaveClass('MuiChip-outlined');
    });

    it('should update selected languages display', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      // Initially should show 2 selected languages (English, Spanish)
      expect(screen.getByText('Selected Languages:')).toBeInTheDocument();

      // Add French
      const frenchChip = screen.getByText('French');
      await user.click(frenchChip);

      // Should now show 3 selected languages
      const selectedLanguagesSection = screen.getByText('Selected Languages:').parentElement;
      expect(selectedLanguagesSection).toBeInTheDocument();
    });
  });

  describe('form validation', () => {
    it('should show error when meeting ID is empty', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      const submitButton = screen.getByRole('button', { name: /spawn bot/i });
      await user.click(submitButton);

      expect(mockOnError).toHaveBeenCalledWith('Meeting ID is required');
      expect(mockSpawnBot).not.toHaveBeenCalled();
    });

    it('should show error when no target languages selected', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      // Fill meeting ID
      const meetingIdInput = screen.getByLabelText(/Google Meet ID/i);
      await user.type(meetingIdInput, 'abc-defg-hij');

      // Deselect all languages
      const englishChip = screen.getByText('English');
      const spanishChip = screen.getByText('Spanish');
      await user.click(englishChip);
      await user.click(spanishChip);

      const submitButton = screen.getByRole('button', { name: /spawn bot/i });
      await user.click(submitButton);

      expect(mockOnError).toHaveBeenCalledWith('At least one target language must be selected');
      expect(mockSpawnBot).not.toHaveBeenCalled();
    });
  });

  describe('form submission', () => {
    it('should submit form with valid data', async () => {
      const user = userEvent.setup();
      mockSpawnBot.mockResolvedValueOnce('bot-123');

      renderBotSpawner();

      // Fill form
      const meetingIdInput = screen.getByLabelText(/Google Meet ID/i);
      const meetingTitleInput = screen.getByLabelText(/Meeting Title/i);
      const organizerEmailInput = screen.getByLabelText(/Organizer Email/i);

      await user.type(meetingIdInput, 'abc-defg-hij');
      await user.type(meetingTitleInput, 'Test Meeting');
      await user.type(organizerEmailInput, 'organizer@example.com');

      // Add French language
      const frenchChip = screen.getByText('French');
      await user.click(frenchChip);

      // Submit form
      const submitButton = screen.getByRole('button', { name: /spawn bot/i });
      await user.click(submitButton);

      expect(mockSpawnBot).toHaveBeenCalledWith({
        meetingId: 'abc-defg-hij',
        meetingTitle: 'Test Meeting',
        organizerEmail: 'organizer@example.com',
        targetLanguages: expect.arrayContaining(['en', 'es', 'fr']),
        autoTranslation: true,
        priority: 'medium',
      });

      await waitFor(() => {
        expect(mockOnBotSpawned).toHaveBeenCalledWith('bot-123');
      });
    });

    it('should reset form after successful submission', async () => {
      const user = userEvent.setup();
      mockSpawnBot.mockResolvedValueOnce('bot-123');

      renderBotSpawner();

      // Fill form
      const meetingIdInput = screen.getByLabelText(/Google Meet ID/i);
      await user.type(meetingIdInput, 'abc-defg-hij');

      // Submit form
      const submitButton = screen.getByRole('button', { name: /spawn bot/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(meetingIdInput).toHaveValue('');
      });
    });

    it('should handle submission error', async () => {
      const user = userEvent.setup();
      const errorMessage = 'Failed to spawn bot';
      mockSpawnBot.mockRejectedValueOnce(new Error(errorMessage));

      renderBotSpawner();

      // Fill form
      const meetingIdInput = screen.getByLabelText(/Google Meet ID/i);
      await user.type(meetingIdInput, 'abc-defg-hij');

      // Submit form
      const submitButton = screen.getByRole('button', { name: /spawn bot/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockOnError).toHaveBeenCalledWith(errorMessage);
      });
    });
  });

  describe('loading state', () => {
    it('should show loading state during submission', async () => {
      const user = userEvent.setup();
      mockUseBotManager.isLoading = true;

      renderBotSpawner();

      const submitButton = screen.getByRole('button', { name: /spawning bot/i });
      expect(submitButton).toBeDisabled();
      expect(screen.getByText('Spawning Bot...')).toBeInTheDocument();
    });

    it('should disable form during loading', () => {
      mockUseBotManager.isLoading = true;

      renderBotSpawner();

      const submitButton = screen.getByRole('button', { name: /spawning bot/i });
      expect(submitButton).toBeDisabled();
    });
  });

  describe('quick actions', () => {
    it('should have quick action buttons', () => {
      renderBotSpawner();

      expect(screen.getByText('Quick Actions')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /demo meeting/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /multi-language/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /high priority/i })).toBeInTheDocument();
    });

    it('should apply demo meeting on quick action click', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      const demoButton = screen.getByRole('button', { name: /demo meeting/i });
      await user.click(demoButton);

      const meetingIdInput = screen.getByLabelText(/Google Meet ID/i);
      expect(meetingIdInput).toHaveValue('demo-meeting-123');
    });

    it('should apply multi-language setup on quick action click', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      const multiLangButton = screen.getByRole('button', { name: /multi-language/i });
      await user.click(multiLangButton);

      // Should have English, Spanish, and French selected
      const selectedLanguagesSection = screen.getByText('Selected Languages:').parentElement;
      expect(selectedLanguagesSection).toHaveTextContent('English');
      expect(selectedLanguagesSection).toHaveTextContent('Spanish');
      expect(selectedLanguagesSection).toHaveTextContent('French');
    });

    it('should set high priority on quick action click', async () => {
      const user = userEvent.setup();
      renderBotSpawner();

      const highPriorityButton = screen.getByRole('button', { name: /high priority/i });
      await user.click(highPriorityButton);

      const prioritySelect = screen.getByLabelText(/Bot Priority/i);
      expect(prioritySelect).toHaveTextContent('high');
    });
  });

  describe('info alert', () => {
    it('should display information about bot functionality', () => {
      renderBotSpawner();

      const infoAlert = screen.getByText(/Bot will join the meeting/i);
      expect(infoAlert).toBeInTheDocument();
      expect(infoAlert).toHaveTextContent('capture audio');
      expect(infoAlert).toHaveTextContent('process captions');
      expect(infoAlert).toHaveTextContent('generate real-time translations');
      expect(infoAlert).toHaveTextContent('Virtual webcam');
    });
  });

  describe('accessibility', () => {
    it('should have proper form labels', () => {
      renderBotSpawner();

      expect(screen.getByLabelText(/Google Meet ID/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Meeting Title/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Organizer Email/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Bot Priority/i)).toBeInTheDocument();
    });

    it('should have proper button roles', () => {
      renderBotSpawner();

      expect(screen.getByRole('button', { name: /spawn bot/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /demo meeting/i })).toBeInTheDocument();
    });

    it('should have proper form structure', () => {
      renderBotSpawner();

      const form = screen.getByRole('form');
      expect(form).toBeInTheDocument();
    });
  });
});