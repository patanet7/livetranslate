/**
 * Analytics Hook
 * 
 * RTK Query-based replacement for unifiedAnalyticsService.ts
 * Provides real-time analytics data with proper caching and error handling
 */

import { useCallback, useMemo } from 'react';
import {
  useGetSystemMetricsQuery,
  useGetSystemHealthQuery,
  useGetAnalyticsOverviewQuery,
} from '@/store/slices/apiSlice';
import type { ServiceHealth as BaseServiceHealth } from '@/types';

// Types
export interface SystemMetrics {
  timestamp: Date;
  cpu: {
    percentage: number;
    cores: number;
    trend: 'up' | 'down' | 'stable';
  };
  memory: {
    percentage: number;
    used: number;
    total: number;
    trend: 'up' | 'down' | 'stable';
  };
  disk: {
    percentage: number;
    used: number;
    total: number;
    trend: 'up' | 'down' | 'stable';
  };
  network: {
    utilization: number;
    latency: number;
    trend: 'up' | 'down' | 'stable';
  };
  performance: {
    activeConnections: number;
    requestsPerSecond: number;
    errorRate: number;
    averageResponseTime: number;
    queueLength: number;
  };
}

// Extended ServiceHealth with analytics-specific fields
export interface ServiceHealth extends Omit<BaseServiceHealth, 'serviceName' | 'lastCheck' | 'status'> {
  id: string;
  name: string;
  status: 'healthy' | 'degraded' | 'critical' | 'unknown';
  responseTime: number;
  lastCheck: Date;
  trend: 'improving' | 'degrading' | 'stable';
  details: string[];
  dependencies: string[];
}

export interface AnalyticsOverview {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageProcessingTime: number;
  peakProcessingTime: number;
  activeStreams: number;
  systemHealthScore: number;
  uptimePercentage: number;
}

export interface ConnectionStatus {
  isConnected: boolean;
  lastUpdate: Date;
  errorCount: number;
  retryAttempts: number;
}

// Configuration
export const ANALYTICS_CONFIG = {
  REFRESH_INTERVAL: 2000, // 2 seconds for real-time updates
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
  CACHE_DURATION: 30000, // 30 seconds cache
  METRIC_THRESHOLDS: {
    cpu: { warning: 70, critical: 90 },
    memory: { warning: 80, critical: 95 },
    disk: { warning: 85, critical: 95 },
    latency: { warning: 500, critical: 1000 },
    responseTime: { warning: 500, critical: 1000 },
    queueLength: { warning: 50, critical: 100 },
  },
};

export const useAnalytics = () => {
  // RTK Query hooks with polling for real-time updates
  const {
    data: systemMetricsData,
    error: systemMetricsError,
    isLoading: systemMetricsLoading,
    isFetching: systemMetricsFetching,
    refetch: refetchSystemMetrics,
  } = useGetSystemMetricsQuery(undefined, {
    pollingInterval: ANALYTICS_CONFIG.REFRESH_INTERVAL,
    refetchOnMountOrArgChange: true,
    refetchOnFocus: true,
    refetchOnReconnect: true,
  });

  const {
    data: systemHealthData,
    error: systemHealthError,
    isLoading: systemHealthLoading,
    isFetching: systemHealthFetching,
    refetch: refetchSystemHealth,
  } = useGetSystemHealthQuery(undefined, {
    pollingInterval: ANALYTICS_CONFIG.REFRESH_INTERVAL,
    refetchOnMountOrArgChange: true,
    refetchOnFocus: true,
    refetchOnReconnect: true,
  });

  const {
    data: analyticsOverviewData,
    error: analyticsOverviewError,
    isLoading: analyticsOverviewLoading,
    isFetching: analyticsOverviewFetching,
    refetch: refetchAnalyticsOverview,
  } = useGetAnalyticsOverviewQuery(undefined, {
    pollingInterval: ANALYTICS_CONFIG.REFRESH_INTERVAL,
    refetchOnMountOrArgChange: true,
    refetchOnFocus: true,
    refetchOnReconnect: true,
  });

  // Transform data from API responses to expected format
  const systemMetrics = useMemo((): SystemMetrics | null => {
    if (!systemMetricsData?.data) return null;

    const data = systemMetricsData.data;
    const now = new Date();

    return {
      timestamp: now,
      cpu: {
        percentage: data.performance?.cpu?.percentage || 0,
        cores: data.performance?.cpu?.cores || 1,
        trend: data.performance?.cpu?.trend || 'stable',
      },
      memory: {
        percentage: data.performance?.memory?.percentage || 0,
        used: data.performance?.memory?.used || 0,
        total: data.performance?.memory?.total || 0,
        trend: data.performance?.memory?.trend || 'stable',
      },
      disk: {
        percentage: data.performance?.disk?.percentage || 0,
        used: data.performance?.disk?.used || 0,
        total: data.performance?.disk?.total || 0,
        trend: data.performance?.disk?.trend || 'stable',
      },
      network: {
        utilization: data.performance?.network?.utilization || 0,
        latency: data.performance?.network?.latency || 0,
        trend: data.performance?.network?.trend || 'stable',
      },
      performance: {
        activeConnections: data.websocket?.active_connections || 0,
        requestsPerSecond: data.performance?.requests_per_second || 0,
        errorRate: data.performance?.error_rate || 0,
        averageResponseTime: data.performance?.average_response_time || 0,
        queueLength: data.performance?.queue_length || 0,
      },
    };
  }, [systemMetricsData]);

  const serviceHealth = useMemo((): ServiceHealth[] => {
    if (!systemHealthData?.data?.services) return [];

    const data = systemHealthData.data;
    const now = new Date();

    return Object.entries(data.services).map(([serviceId, service]: [string, any]) => ({
      id: serviceId,
      name: service.name || serviceId,
      status: service.status === 'healthy' ? 'healthy' : 
             service.status === 'degraded' ? 'degraded' : 'critical',
      uptime: service.uptime || 0,
      responseTime: service.response_time || 0,
      version: service.version || 'unknown',
      lastCheck: now,
      trend: service.trend || 'stable',
      details: service.details || [],
      dependencies: service.dependencies || [],
    }));
  }, [systemHealthData]);

  const analyticsOverview = useMemo((): AnalyticsOverview | null => {
    if (!analyticsOverviewData?.data) return null;

    const data = analyticsOverviewData.data;

    return {
      totalRequests: data.overview?.total_requests || 0,
      successfulRequests: data.overview?.successful_requests || 0,
      failedRequests: data.overview?.failed_requests || 0,
      averageProcessingTime: data.overview?.average_response_time || 0,
      peakProcessingTime: data.overview?.peak_response_time || 0,
      activeStreams: data.audio?.active_streams || 0,
      systemHealthScore: data.overview?.system_health_score || 0,
      uptimePercentage: data.overview?.uptime_percentage || 0,
    };
  }, [analyticsOverviewData]);

  // Connection status based on API states
  const connectionStatus = useMemo((): ConnectionStatus => {
    const hasErrors = systemMetricsError || systemHealthError || analyticsOverviewError;
    const isLoading = systemMetricsLoading || systemHealthLoading || analyticsOverviewLoading;

    return {
      isConnected: !hasErrors && !isLoading,
      lastUpdate: new Date(),
      errorCount: hasErrors ? 1 : 0,
      retryAttempts: 0, // RTK Query handles retries automatically
    };
  }, [
    systemMetricsError,
    systemHealthError,
    analyticsOverviewError,
    systemMetricsLoading,
    systemHealthLoading,
    analyticsOverviewLoading,
    systemMetricsFetching,
    systemHealthFetching,
    analyticsOverviewFetching,
  ]);

  // Loading states
  const isLoading = systemMetricsLoading || systemHealthLoading || analyticsOverviewLoading;
  const isFetching = systemMetricsFetching || systemHealthFetching || analyticsOverviewFetching;

  // Error handling
  const errors = {
    systemMetrics: systemMetricsError,
    systemHealth: systemHealthError,
    analyticsOverview: analyticsOverviewError,
  };

  const hasError = Object.values(errors).some(error => error !== undefined);

  // Manual refresh function
  const refreshData = useCallback(async () => {
    await Promise.all([
      refetchSystemMetrics(),
      refetchSystemHealth(),
      refetchAnalyticsOverview(),
    ]);
  }, [refetchSystemMetrics, refetchSystemHealth, refetchAnalyticsOverview]);

  // Utility functions
  const getMetricStatus = useCallback((
    value: number,
    metric: keyof typeof ANALYTICS_CONFIG.METRIC_THRESHOLDS
  ): 'healthy' | 'warning' | 'critical' => {
    const thresholds = ANALYTICS_CONFIG.METRIC_THRESHOLDS[metric];
    if (value >= thresholds.critical) return 'critical';
    if (value >= thresholds.warning) return 'warning';
    return 'healthy';
  }, []);

  const formatMetricValue = useCallback((value: number, unit: string): string => {
    if (unit === 'ms') {
      return `${Math.round(value)}ms`;
    }
    if (unit === '%') {
      return `${Math.round(value)}%`;
    }
    if (unit === 'MB' || unit === 'GB') {
      return `${(value / 1024 / 1024).toFixed(1)}MB`;
    }
    return `${Math.round(value)}${unit}`;
  }, []);

  // Combined data object for components that need all data
  const allData = useMemo(() => ({
    systemMetrics,
    serviceHealth,
    analyticsOverview,
    connectionStatus,
  }), [systemMetrics, serviceHealth, analyticsOverview, connectionStatus]);

  return {
    // Data
    systemMetrics,
    serviceHealth,
    analyticsOverview,
    connectionStatus,
    allData,

    // States
    isLoading,
    isFetching,
    hasError,
    errors,

    // Actions
    refreshData,

    // Utilities
    getMetricStatus,
    formatMetricValue,

    // Configuration
    config: ANALYTICS_CONFIG,
  };
};

export type AnalyticsHook = ReturnType<typeof useAnalytics>;
export default useAnalytics;