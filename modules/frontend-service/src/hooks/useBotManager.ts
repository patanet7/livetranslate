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
      const response = await fetch('/api/bot/spawn', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to spawn bot');
      }

      const data = await response.json();
      
      // Add the new bot to the store
      dispatch(addBot(data.bot));
      
      return data.botId;
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
          lastActiveAt: Date.now(),
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
      const response = await fetch(`/api/bot/${botId}/status`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to get bot status');
      }

      const bot = await response.json();
      
      // Update the bot in the store
      dispatch(updateBot({
        botId,
        updates: bot,
      }));

      return bot;
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
        throw new Error(errorData.message || 'Failed to get active bots');
      }

      const data = await response.json();
      
      // Update the store with active bots
      const botsObject = data.bots.reduce((acc: Record<string, BotInstance>, bot: BotInstance) => {
        acc[bot.botId] = bot;
        return acc;
      }, {});
      
      dispatch(setBots(botsObject));
      dispatch(setActiveBotIds(data.activeBotIds));

      return data.bots;
    } catch (error) {
      handleError(error);
      throw error;
    }
  }, [dispatch, handleError]);

  const getSystemStats = useCallback(async (): Promise<SystemStats> => {
    try {
      const response = await fetch('/api/bot/stats');
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to get system stats');
      }

      const stats = await response.json();
      
      // Update the store with system stats
      dispatch(setSystemStats(stats));

      return stats;
    } catch (error) {
      handleError(error);
      throw error;
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
          lastActiveAt: Date.now(),
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
          lastActiveAt: Date.now(),
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
          lastActiveAt: Date.now(),
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