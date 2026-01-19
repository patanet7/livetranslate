/**
 * Offline Detection and Queue Management Hook
 *
 * Provides offline detection, request queuing, and automatic retry
 * when connection is restored
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useSnackbar } from "notistack";
import { useNotifications } from "./useNotifications";

interface QueuedRequest {
  id: string;
  url: string;
  options: RequestInit;
  timestamp: Date;
  retryCount: number;
  maxRetries: number;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}

interface OfflineOptions {
  enableQueue?: boolean;
  maxQueueSize?: number;
  maxRetries?: number;
  retryDelay?: number;
  showNotifications?: boolean;
}

export const useOfflineHandler = (options: OfflineOptions = {}) => {
  const {
    enableQueue = true,
    maxQueueSize = 50,
    maxRetries = 3,
    retryDelay = 1000,
    showNotifications = true,
  } = options;

  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const { notifyWarning, notifySuccess } = useNotifications();

  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [queueSize, setQueueSize] = useState(0);
  const [isProcessingQueue, setIsProcessingQueue] = useState(false);

  const requestQueueRef = useRef<QueuedRequest[]>([]);
  const offlineNotificationRef = useRef<string | null>(null);

  // Update online status
  const updateOnlineStatus = useCallback(() => {
    setIsOnline(navigator.onLine);
  }, []);

  // Show/hide offline notification
  const showOfflineNotification = useCallback(() => {
    if (!showNotifications || offlineNotificationRef.current) return;

    offlineNotificationRef.current = enqueueSnackbar(
      "You are currently offline. Changes will be queued and synced when connection is restored.",
      {
        variant: "warning",
        persist: true,
        action: (key) => (
          <button
            onClick={() => closeSnackbar(key)}
            style={{
              color: "white",
              background: "none",
              border: "1px solid white",
              padding: "4px 8px",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Dismiss
          </button>
        ),
      },
    ) as string;

    notifyWarning(
      "Offline Mode",
      "You are currently offline. Changes will be queued and synced when connection is restored.",
      false,
    );
  }, [showNotifications, enqueueSnackbar, closeSnackbar, notifyWarning]);

  const hideOfflineNotification = useCallback(() => {
    if (offlineNotificationRef.current) {
      closeSnackbar(offlineNotificationRef.current);
      offlineNotificationRef.current = null;
    }

    if (showNotifications) {
      enqueueSnackbar("Connection restored! Syncing queued changes...", {
        variant: "success",
        autoHideDuration: 3000,
      });

      notifySuccess("Connection Restored", "Syncing queued changes...");
    }
  }, [showNotifications, enqueueSnackbar, closeSnackbar, notifySuccess]);

  // Add request to queue
  const queueRequest = useCallback(
    (url: string, options: RequestInit = {}): Promise<any> => {
      return new Promise((resolve, reject) => {
        if (requestQueueRef.current.length >= maxQueueSize) {
          reject(new Error("Request queue is full"));
          return;
        }

        const queuedRequest: QueuedRequest = {
          id: `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          url,
          options,
          timestamp: new Date(),
          retryCount: 0,
          maxRetries,
          resolve,
          reject,
        };

        requestQueueRef.current.push(queuedRequest);
        setQueueSize(requestQueueRef.current.length);
      });
    },
    [maxQueueSize, maxRetries],
  );

  // Process queued requests
  const processQueue = useCallback(async () => {
    if (
      !isOnline ||
      requestQueueRef.current.length === 0 ||
      isProcessingQueue
    ) {
      return;
    }

    setIsProcessingQueue(true);

    const queue = [...requestQueueRef.current];
    requestQueueRef.current = [];

    for (const request of queue) {
      try {
        const response = await fetch(request.url, request.options);

        if (response.ok) {
          const data = await response.json();
          request.resolve(data);
        } else {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
      } catch (error) {
        request.retryCount++;

        if (request.retryCount < request.maxRetries) {
          // Re-queue for retry
          requestQueueRef.current.push({
            ...request,
            timestamp: new Date(),
          });
        } else {
          request.reject(error);
        }
      }

      // Small delay between requests to avoid overwhelming the server
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
    }

    setQueueSize(requestQueueRef.current.length);
    setIsProcessingQueue(false);

    // Process any remaining requests
    if (requestQueueRef.current.length > 0) {
      setTimeout(processQueue, retryDelay * 2);
    }
  }, [isOnline, isProcessingQueue, retryDelay]);

  // Enhanced fetch function that handles offline scenarios
  const fetchWithOfflineSupport = useCallback(
    async (url: string, options: RequestInit = {}): Promise<any> => {
      if (isOnline) {
        try {
          const response = await fetch(url, options);
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
          return await response.json();
        } catch (error) {
          // If fetch fails and we're supposedly online, check if it's a network error
          if (!navigator.onLine) {
            setIsOnline(false);
            if (enableQueue) {
              return queueRequest(url, options);
            }
          }
          throw error;
        }
      } else {
        if (enableQueue) {
          return queueRequest(url, options);
        } else {
          throw new Error("No internet connection and queuing is disabled");
        }
      }
    },
    [isOnline, enableQueue, queueRequest],
  );

  // Clear queue
  const clearQueue = useCallback(() => {
    requestQueueRef.current.forEach((request) => {
      request.reject(new Error("Request queue cleared"));
    });
    requestQueueRef.current = [];
    setQueueSize(0);
  }, []);

  // Get queue statistics
  const getQueueStats = useCallback(() => {
    const now = new Date();
    const queue = requestQueueRef.current;

    return {
      totalRequests: queue.length,
      oldestRequest: queue.length > 0 ? queue[0].timestamp : null,
      newestRequest:
        queue.length > 0 ? queue[queue.length - 1].timestamp : null,
      averageAge:
        queue.length > 0
          ? queue.reduce(
              (sum, req) => sum + (now.getTime() - req.timestamp.getTime()),
              0,
            ) / queue.length
          : 0,
      retryingRequests: queue.filter((req) => req.retryCount > 0).length,
    };
  }, []);

  // Set up event listeners
  useEffect(() => {
    const handleOnline = () => {
      updateOnlineStatus();
      hideOfflineNotification();
      if (enableQueue) {
        processQueue();
      }
    };

    const handleOffline = () => {
      updateOnlineStatus();
      showOfflineNotification();
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    // Initial status check
    updateOnlineStatus();

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [
    updateOnlineStatus,
    showOfflineNotification,
    hideOfflineNotification,
    processQueue,
    enableQueue,
  ]);

  // Process queue when coming back online
  useEffect(() => {
    if (isOnline && enableQueue && requestQueueRef.current.length > 0) {
      processQueue();
    }
  }, [isOnline, enableQueue, processQueue]);

  // Show offline notification when going offline
  useEffect(() => {
    if (!isOnline) {
      showOfflineNotification();
    } else {
      hideOfflineNotification();
    }
  }, [isOnline, showOfflineNotification, hideOfflineNotification]);

  return {
    isOnline,
    queueSize,
    isProcessingQueue,
    fetchWithOfflineSupport,
    queueRequest,
    processQueue,
    clearQueue,
    getQueueStats,
  };
};

export default useOfflineHandler;
