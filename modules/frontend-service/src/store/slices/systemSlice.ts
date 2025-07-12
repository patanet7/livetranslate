import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { ServiceHealth, SystemHealth } from '@/types';

interface SystemState {
  // Service health monitoring
  serviceHealth: {
    orchestration: ServiceHealth | null;
    whisper: ServiceHealth | null;
    translation: ServiceHealth | null;
    monitoring: ServiceHealth | null;
  };
  
  // Overall system health
  systemHealth: SystemHealth | null;
  
  // Performance metrics
  performance: {
    cpu: {
      usage: number;
      cores: number;
      temperature?: number;
    };
    memory: {
      used: number;
      total: number;
      percentage: number;
    };
    disk: {
      used: number;
      total: number;
      percentage: number;
    };
    network: {
      bytesIn: number;
      bytesOut: number;
      latency: number;
    };
  };
  
  // Application metrics
  applicationMetrics: {
    activeConnections: number;
    requestsPerSecond: number;
    averageResponseTime: number;
    errorRate: number;
    uptime: number;
    version: string;
  };
  
  // System configuration
  configuration: {
    environment: 'development' | 'staging' | 'production';
    debug: boolean;
    logLevel: 'debug' | 'info' | 'warn' | 'error';
    features: Record<string, boolean>;
    limits: {
      maxBots: number;
      maxConnections: number;
      maxFileSize: number;
      sessionTimeout: number;
    };
  };
  
  // Service versions and build info
  buildInfo: {
    version: string;
    buildDate: string;
    gitCommit: string;
    environment: string;
    dependencies: Record<string, string>;
  };
  
  // Alerts and notifications
  alerts: Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'critical';
    title: string;
    message: string;
    timestamp: number;
    acknowledged: boolean;
    source: string;
  }>;
  
  // System logs (recent entries)
  logs: Array<{
    id: string;
    level: 'debug' | 'info' | 'warn' | 'error';
    message: string;
    timestamp: number;
    source: string;
    metadata?: Record<string, any>;
  }>;
  
  // Feature flags
  featureFlags: Record<string, boolean>;
  
  // Maintenance mode
  maintenance: {
    enabled: boolean;
    message: string;
    scheduledStart?: number;
    scheduledEnd?: number;
  };
  
  // Statistics
  statistics: {
    totalRequests: number;
    totalErrors: number;
    totalSessions: number;
    totalDataProcessed: number; // in bytes
    averageSessionDuration: number;
    peakConcurrentUsers: number;
  };
  
  // Loading and error states
  loading: boolean;
  error: string | null;
  lastUpdated: number;
}

const initialState: SystemState = {
  serviceHealth: {
    orchestration: null,
    whisper: null,
    translation: null,
    monitoring: null,
  },
  
  systemHealth: null,
  
  performance: {
    cpu: {
      usage: 0,
      cores: 1,
    },
    memory: {
      used: 0,
      total: 0,
      percentage: 0,
    },
    disk: {
      used: 0,
      total: 0,
      percentage: 0,
    },
    network: {
      bytesIn: 0,
      bytesOut: 0,
      latency: 0,
    },
  },
  
  applicationMetrics: {
    activeConnections: 0,
    requestsPerSecond: 0,
    averageResponseTime: 0,
    errorRate: 0,
    uptime: 0,
    version: '1.0.0',
  },
  
  configuration: {
    environment: 'development',
    debug: true,
    logLevel: 'info',
    features: {},
    limits: {
      maxBots: 50,
      maxConnections: 1000,
      maxFileSize: 100 * 1024 * 1024, // 100MB
      sessionTimeout: 1800000, // 30 minutes
    },
  },
  
  buildInfo: {
    version: '1.0.0',
    buildDate: '',
    gitCommit: '',
    environment: 'development',
    dependencies: {},
  },
  
  alerts: [],
  logs: [],
  featureFlags: {},
  
  maintenance: {
    enabled: false,
    message: '',
  },
  
  statistics: {
    totalRequests: 0,
    totalErrors: 0,
    totalSessions: 0,
    totalDataProcessed: 0,
    averageSessionDuration: 0,
    peakConcurrentUsers: 0,
  },
  
  loading: false,
  error: null,
  lastUpdated: 0,
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    // Service health updates
    updateServiceHealth: (state, action: PayloadAction<{ 
      service: keyof SystemState['serviceHealth']; 
      health: ServiceHealth 
    }>) => {
      state.serviceHealth[action.payload.service] = action.payload.health;
      state.lastUpdated = Date.now();
    },
    
    updateSystemHealth: (state, action: PayloadAction<SystemHealth>) => {
      state.systemHealth = action.payload;
      state.lastUpdated = Date.now();
    },
    
    // Update from WebSocket health data
    updateSystemMetrics: (state, action: PayloadAction<any>) => {
      const data = action.payload;
      
      // Update service health from services data
      if (data.services) {
        Object.entries(data.services).forEach(([serviceName, serviceData]: [string, any]) => {
          if (serviceName in state.serviceHealth) {
            state.serviceHealth[serviceName as keyof SystemState['serviceHealth']] = {
              serviceName,
              status: serviceData.status,
              lastCheck: serviceData.last_check || Date.now(),
              responseTime: serviceData.response_time || 0,
              errorCount: serviceData.error_count || 0,
              version: '1.0.0',
              dependencies: [],
              healthChecks: [],
              alerts: [],
              endpoints: [],
              lastError: serviceData.last_error || null,
            };
          }
        });
      }
      
      // Update performance metrics
      if (data.performance) {
        const perf = data.performance;
        if (perf.cpu) {
          state.performance.cpu = {
            usage: perf.cpu.usage || 0,
            cores: perf.cpu.cores || 1,
          };
        }
        if (perf.memory) {
          state.performance.memory = {
            used: perf.memory.used || 0,
            total: perf.memory.total || 1,
            percentage: perf.memory.percentage || 0,
          };
        }
        if (perf.disk) {
          state.performance.disk = {
            used: perf.disk.used || 0,
            total: perf.disk.total || 1,
            percentage: perf.disk.percentage || 0,
          };
        }
        if (perf.network) {
          state.performance.network = {
            bytesIn: perf.network.bytes_recv || 0,
            bytesOut: perf.network.bytes_sent || 0,
            latency: 0,
          };
        }
      }
      
      // Update uptime
      if (data.uptime) {
        state.applicationMetrics.uptime = data.uptime;
      }
      
      state.lastUpdated = Date.now();
    },
    
    // Performance metrics
    updatePerformanceMetrics: (state, action: PayloadAction<Partial<SystemState['performance']>>) => {
      Object.assign(state.performance, action.payload);
      state.lastUpdated = Date.now();
    },
    
    updateCPUMetrics: (state, action: PayloadAction<SystemState['performance']['cpu']>) => {
      state.performance.cpu = action.payload;
    },
    
    updateMemoryMetrics: (state, action: PayloadAction<SystemState['performance']['memory']>) => {
      state.performance.memory = action.payload;
    },
    
    updateNetworkMetrics: (state, action: PayloadAction<SystemState['performance']['network']>) => {
      state.performance.network = action.payload;
    },
    
    // Application metrics
    updateApplicationMetrics: (state, action: PayloadAction<Partial<SystemState['applicationMetrics']>>) => {
      Object.assign(state.applicationMetrics, action.payload);
      state.lastUpdated = Date.now();
    },
    
    incrementConnections: (state) => {
      state.applicationMetrics.activeConnections += 1;
    },
    
    decrementConnections: (state) => {
      state.applicationMetrics.activeConnections = Math.max(0, state.applicationMetrics.activeConnections - 1);
    },
    
    updateRequestsPerSecond: (state, action: PayloadAction<number>) => {
      state.applicationMetrics.requestsPerSecond = action.payload;
    },
    
    // Configuration updates
    updateConfiguration: (state, action: PayloadAction<Partial<SystemState['configuration']>>) => {
      Object.assign(state.configuration, action.payload);
    },
    
    updateFeatureFlag: (state, action: PayloadAction<{ feature: string; enabled: boolean }>) => {
      state.configuration.features[action.payload.feature] = action.payload.enabled;
    },
    
    updateLimits: (state, action: PayloadAction<Partial<SystemState['configuration']['limits']>>) => {
      Object.assign(state.configuration.limits, action.payload);
    },
    
    // Build info
    setBuildInfo: (state, action: PayloadAction<SystemState['buildInfo']>) => {
      state.buildInfo = action.payload;
    },
    
    // Alerts management
    addAlert: (state, action: PayloadAction<Omit<SystemState['alerts'][0], 'id' | 'timestamp' | 'acknowledged'>>) => {
      const alert = {
        id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: Date.now(),
        acknowledged: false,
        ...action.payload,
      };
      
      state.alerts.unshift(alert);
      
      // Keep only the last 100 alerts
      if (state.alerts.length > 100) {
        state.alerts.pop();
      }
    },
    
    acknowledgeAlert: (state, action: PayloadAction<string>) => {
      const alert = state.alerts.find(a => a.id === action.payload);
      if (alert) {
        alert.acknowledged = true;
      }
    },
    
    removeAlert: (state, action: PayloadAction<string>) => {
      state.alerts = state.alerts.filter(a => a.id !== action.payload);
    },
    
    clearAlerts: (state) => {
      state.alerts = [];
    },
    
    // Logs management
    addLog: (state, action: PayloadAction<Omit<SystemState['logs'][0], 'id' | 'timestamp'>>) => {
      const log = {
        id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: Date.now(),
        ...action.payload,
      };
      
      state.logs.unshift(log);
      
      // Keep only the last 500 logs
      if (state.logs.length > 500) {
        state.logs.pop();
      }
    },
    
    clearLogs: (state) => {
      state.logs = [];
    },
    
    // Feature flags
    setFeatureFlags: (state, action: PayloadAction<Record<string, boolean>>) => {
      state.featureFlags = action.payload;
    },
    
    toggleFeatureFlag: (state, action: PayloadAction<string>) => {
      state.featureFlags[action.payload] = !state.featureFlags[action.payload];
    },
    
    // Maintenance mode
    setMaintenanceMode: (state, action: PayloadAction<{
      enabled: boolean;
      message?: string;
      scheduledStart?: number;
      scheduledEnd?: number;
    }>) => {
      Object.assign(state.maintenance, action.payload);
    },
    
    enableMaintenance: (state, action: PayloadAction<{ message: string }>) => {
      state.maintenance.enabled = true;
      state.maintenance.message = action.payload.message;
    },
    
    disableMaintenance: (state) => {
      state.maintenance.enabled = false;
      state.maintenance.message = '';
      state.maintenance.scheduledStart = undefined;
      state.maintenance.scheduledEnd = undefined;
    },
    
    // Statistics updates
    updateStatistics: (state, action: PayloadAction<Partial<SystemState['statistics']>>) => {
      Object.assign(state.statistics, action.payload);
    },
    
    incrementRequests: (state) => {
      state.statistics.totalRequests += 1;
    },
    
    incrementErrors: (state) => {
      state.statistics.totalErrors += 1;
    },
    
    incrementSessions: (state) => {
      state.statistics.totalSessions += 1;
    },
    
    addDataProcessed: (state, action: PayloadAction<number>) => {
      state.statistics.totalDataProcessed += action.payload;
    },
    
    updatePeakUsers: (state, action: PayloadAction<number>) => {
      if (action.payload > state.statistics.peakConcurrentUsers) {
        state.statistics.peakConcurrentUsers = action.payload;
      }
    },
    
    // Loading and error states
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
      state.loading = false;
    },
    
    clearError: (state) => {
      state.error = null;
    },
    
    // Reset state
    resetSystemState: () => initialState,
    
    // Bulk updates for real-time data
    updateSystemMetricsBulk: (state, action: PayloadAction<{
      performance?: Partial<SystemState['performance']>;
      applicationMetrics?: Partial<SystemState['applicationMetrics']>;
      serviceHealth?: Partial<SystemState['serviceHealth']>;
    }>) => {
      if (action.payload.performance) {
        Object.assign(state.performance, action.payload.performance);
      }
      if (action.payload.applicationMetrics) {
        Object.assign(state.applicationMetrics, action.payload.applicationMetrics);
      }
      if (action.payload.serviceHealth) {
        Object.assign(state.serviceHealth, action.payload.serviceHealth);
      }
      state.lastUpdated = Date.now();
    },
  },
});

export const {
  updateServiceHealth,
  updateSystemHealth,
  updatePerformanceMetrics,
  updateCPUMetrics,
  updateMemoryMetrics,
  updateNetworkMetrics,
  updateApplicationMetrics,
  incrementConnections,
  decrementConnections,
  updateRequestsPerSecond,
  updateConfiguration,
  updateFeatureFlag,
  updateLimits,
  setBuildInfo,
  addAlert,
  acknowledgeAlert,
  removeAlert,
  clearAlerts,
  addLog,
  clearLogs,
  setFeatureFlags,
  toggleFeatureFlag,
  setMaintenanceMode,
  enableMaintenance,
  disableMaintenance,
  updateStatistics,
  incrementRequests,
  incrementErrors,
  incrementSessions,
  addDataProcessed,
  updatePeakUsers,
  setLoading,
  setError,
  clearError,
  resetSystemState,
  updateSystemMetrics,
  updateSystemMetricsBulk,
} = systemSlice.actions;

export default systemSlice;