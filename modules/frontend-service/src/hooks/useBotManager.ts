import { useState, useCallback } from 'react';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import {
  setBots,
  setActiveBotIds,
  setSystemStats,
  addBot,
  updateBot,
  setError,
} from '@/store/slices/botSlice';
import { MeetingRequest, BotInstance, SystemStats } from '@/types';

export const useBotManager = () => {
  const dispatch = useAppDispatch();
  const { bots, activeBotIds, systemStats, loading, error } = useAppSelector(state => state.bot);
  
  const [localLoading, setLocalLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleError = useCallback((error: any) => {
    const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred';
    setLocalError(errorMessage);
    dispatch(setError(errorMessage));
  }, [dispatch]);

  const spawnBot = useCallback(async (request: MeetingRequest): Promise<string> => {
    setLocalLoading(true);
    setLocalError(null);
    
    try {
      // Convert MeetingRequest to nested BotSpawnRequest format expected by backend
      const botSpawnRequest = {
        config: {
          meeting_info: {
            meeting_id: request.meetingId,
            meeting_title: request.meetingTitle || `Meeting ${request.meetingId}`,
            meeting_url: `https://meet.google.com/${request.meetingId}`,
            platform: 'google_meet',
            organizer_email: request.organizerEmail || null,
            participant_count: 0
          },
          audio_capture: {
            sample_rate: 16000,
            channels: 1,
            buffer_duration: 4.0,
            enable_noise_reduction: true,
            enable_echo_cancellation: false,
            enable_auto_gain_control: false,
            quality_threshold: 0.7
          },
          translation: {
            target_languages: request.targetLanguages || ['en'],
            enable_auto_translation: request.autoTranslation ?? true,
            translation_quality: 'balanced',
            real_time_translation: true,
            confidence_threshold: 0.6
          },
          webcam: {
            display_mode: 'overlay',
            theme: 'dark',
            max_translations_displayed: 5,
            translation_duration_seconds: 10.0,
            show_speaker_names: true,
            show_confidence: true,
            show_timestamps: false
          },
          priority: request.priority || 'medium',
          auto_terminate_minutes: 180,
          enable_recording: true,
          enable_transcription: true,
          enable_speaker_diarization: true,
          enable_virtual_webcam: true
        },
        user_id: null,
        session_id: null,
        metadata: {
          organizer_email: request.organizerEmail,
          frontend_version: '2.0',
          spawn_source: 'bot_management_ui',
          audio_storage_enabled: true,
          cleanup_on_exit: true
        }
      };

      const response = await fetch('/api/bot/spawn', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(botSpawnRequest),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || errorData.message || 'Failed to spawn bot');
      }

      const data = await response.json();

      // Create bot instance for the store (partial update - full data will come from status updates)
      const botInstance: Partial<BotInstance> = {
        id: data.bot_id,
        botId: data.bot_id,
        status: (data.status as BotInstance['status']) || 'spawning',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        lastActiveAt: new Date().toISOString(),
        errorMessages: [],
      };

      // Add the new bot to the store (will be merged with existing state)
      dispatch(addBot(botInstance as BotInstance));

      return data.bot_id;
    } catch (error) {
      handleError(error);
      throw error;
    } finally {
      setLocalLoading(false);
    }
  }, [dispatch, handleError]);

  const terminateBot = useCallback(async (botId: string): Promise<void> => {
    setLocalLoading(true);
    setLocalError(null);
    
    try {
      const response = await fetch(`/api/bot/${botId}/terminate`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to terminate bot');
      }

      // Update bot status to terminated
      dispatch(updateBot({
        botId,
        updates: {
          status: 'terminated',
          lastActiveAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      }));
    } catch (error) {
      handleError(error);
      throw error;
    } finally {
      setLocalLoading(false);
    }
  }, [dispatch, handleError]);

  const getBotStatus = useCallback(async (botId: string): Promise<BotInstance> => {
    try {
      const response = await fetch(`/api/bot/${botId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || errorData.message || 'Failed to get bot status');
      }

      const rawBot = await response.json();

      // Convert backend format to frontend format (partial update)
      const botUpdates: Partial<BotInstance> = {
        id: rawBot.bot_id || rawBot.botId || botId,
        botId: rawBot.bot_id || rawBot.botId || botId,
        status: rawBot.status || 'active',
        createdAt: rawBot.created_at || rawBot.createdAt || new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        lastActiveAt: rawBot.last_activity || rawBot.lastActiveAt || new Date().toISOString(),
        errorMessages: rawBot.error_message ? [rawBot.error_message] : [],
      };

      // Update the bot in the store
      dispatch(updateBot({
        botId,
        updates: botUpdates,
      }));

      return botUpdates as BotInstance;
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [dispatch, handleError]);

  const getActiveBots = useCallback(async (): Promise<BotInstance[]> => {
    try {
      const response = await fetch('/api/bot/active');
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || errorData.message || 'Failed to get active bots');
      }

      const activeBots = await response.json();

      // Convert backend format to frontend format (partial updates)
      const processedBots = Array.isArray(activeBots) ? activeBots : [];
      const botsObject = processedBots.reduce((acc: Record<string, Partial<BotInstance>>, bot: any) => {
        const processedBot: Partial<BotInstance> = {
          id: bot.bot_id || bot.botId,
          botId: bot.bot_id || bot.botId,
          status: bot.status || 'active',
          createdAt: bot.created_at || bot.createdAt || new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          lastActiveAt: bot.last_activity || bot.lastActiveAt || new Date().toISOString(),
          errorMessages: bot.error_message ? [bot.error_message] : [],
        };
        acc[processedBot.botId!] = processedBot;
        return acc;
      }, {});

      const activeBotIds = processedBots.map((bot: any) => bot.bot_id || bot.botId);

      dispatch(setBots(botsObject as Record<string, BotInstance>));
      dispatch(setActiveBotIds(activeBotIds));

      return processedBots;
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [dispatch, handleError]);

  const getSystemStats = useCallback(async (): Promise<SystemStats> => {
    try {
      const response = await fetch('/api/bot/stats');
      
      if (!response.ok) {
        // Return default stats instead of throwing error to prevent crashes
        console.warn(`Bot stats API failed with status ${response.status}, using default values`);
        return {
          totalBotsSpawned: 0,
          activeBots: 0,
          completedSessions: 0,
          errorRate: 0,
          averageSessionDuration: 0,
        };
      }

      const rawStats = await response.json();

      // Convert backend format to frontend format
      const stats: SystemStats = {
        totalBotsSpawned: rawStats.total_bots_created || rawStats.totalBotsSpawned || 0,
        activeBots: rawStats.active_bots || rawStats.activeBots || 0,
        completedSessions: rawStats.total_bots_completed || rawStats.completedSessions || 0,
        errorRate: rawStats.recovery_rate ? (100 - rawStats.recovery_rate) / 100 : rawStats.errorRate || 0,
        averageSessionDuration: rawStats.capacity_utilization || rawStats.averageSessionDuration || 0
      };

      // Update the store with system stats
      dispatch(setSystemStats(stats));

      return stats;
    } catch (error) {
      console.error('Error getting system stats:', error);
      handleError(error);
      // Return default stats instead of throwing to prevent crashes
      return {
        totalBotsSpawned: 0,
        activeBots: 0,
        completedSessions: 0,
        errorRate: 0,
        averageSessionDuration: 0,
      };
    }
  }, [dispatch, handleError]);

  const refreshBots = useCallback(async (): Promise<void> => {
    setLocalLoading(true);
    setLocalError(null);
    
    try {
      await Promise.all([
        getActiveBots(),
        getSystemStats(),
      ]);
    } catch (error) {
      handleError(error);
    } finally {
      setLocalLoading(false);
    }
  }, [getActiveBots, getSystemStats, handleError]);

  const getBotSessions = useCallback(async (botId?: string) => {
    try {
      const url = botId ? `/api/bot/${botId}/sessions` : '/api/bot/sessions';
      const response = await fetch(url);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to get bot sessions');
      }

      return await response.json();
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [handleError]);

  const exportBotData = useCallback(async (botId: string, format: 'json' | 'csv' = 'json') => {
    try {
      const response = await fetch(`/api/bot/${botId}/export?format=${format}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to export bot data');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `bot_${botId}_export.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [handleError]);

  const updateBotConfig = useCallback(async (botId: string, config: any) => {
    try {
      const response = await fetch(`/api/bot/${botId}/config`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to update bot config');
      }

      const updatedBot = await response.json();
      
      // Update the bot in the store
      dispatch(updateBot({
        botId,
        updates: updatedBot,
      }));

      return updatedBot;
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [dispatch, handleError]);

  const getBotLogs = useCallback(async (botId: string, limit: number = 100) => {
    try {
      const response = await fetch(`/api/bot/${botId}/logs?limit=${limit}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to get bot logs');
      }

      return await response.json();
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [handleError]);

  const getBotTranslations = useCallback(async (botId: string, limit: number = 50) => {
    try {
      const response = await fetch(`/api/bot/${botId}/translations?limit=${limit}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to get bot translations');
      }

      return await response.json();
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [handleError]);

  const restartBot = useCallback(async (botId: string): Promise<void> => {
    setLocalLoading(true);
    setLocalError(null);
    
    try {
      const response = await fetch(`/api/bot/${botId}/restart`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to restart bot');
      }

      // Update bot status to spawning
      dispatch(updateBot({
        botId,
        updates: {
          status: 'spawning',
          lastActiveAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      }));
    } catch (error) {
      handleError(error);
      throw error;
    } finally {
      setLocalLoading(false);
    }
  }, [dispatch, handleError]);

  const pauseBot = useCallback(async (botId: string): Promise<void> => {
    try {
      const response = await fetch(`/api/bot/${botId}/pause`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to pause bot');
      }

      // Update bot status (you might want to add a 'paused' status)
      dispatch(updateBot({
        botId,
        updates: {
          lastActiveAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      }));
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [dispatch, handleError]);

  const resumeBot = useCallback(async (botId: string): Promise<void> => {
    try {
      const response = await fetch(`/api/bot/${botId}/resume`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to resume bot');
      }

      // Update bot status back to active
      dispatch(updateBot({
        botId,
        updates: {
          status: 'active',
          lastActiveAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      }));
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [dispatch, handleError]);

  return {
    // State
    bots,
    activeBotIds,
    systemStats,
    isLoading: loading || localLoading,
    error: error || localError,

    // Bot management
    spawnBot,
    terminateBot,
    restartBot,
    pauseBot,
    resumeBot,
    getBotStatus,
    getActiveBots,
    refreshBots,
    updateBotConfig,

    // Data retrieval
    getSystemStats,
    getBotSessions,
    getBotLogs,
    getBotTranslations,
    exportBotData,

    // Utility
    clearError: () => {
      setLocalError(null);
      dispatch(setError(null));
    },
  };
};