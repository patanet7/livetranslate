import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { ThemeMode } from '@/styles/theme';
import { Notification } from '@/types';

interface UIState {
  // Theme and appearance
  theme: ThemeMode;
  
  // Layout and navigation
  sidebarOpen: boolean;
  sidebarCollapsed: boolean;
  activeTab: string;
  activePage: string;
  
  // Notifications
  notifications: Notification[];
  
  // Modals and dialogs
  modals: {
    botSpawner: boolean;
    audioSettings: boolean;
    systemSettings: boolean;
    helpDialog: boolean;
    aboutDialog: boolean;
  };
  
  // Loading states
  loading: {
    global: boolean;
    audio: boolean;
    bot: boolean;
    system: boolean;
  };
  
  // Dashboard preferences
  dashboard: {
    autoRefresh: boolean;
    refreshInterval: number;
    compactMode: boolean;
    showAdvancedMetrics: boolean;
    selectedMetrics: string[];
  };
  
  // Audio testing preferences
  audioTesting: {
    showAdvancedControls: boolean;
    autoStartRecording: boolean;
    showVisualization: boolean;
    visualizationType: 'waveform' | 'frequency' | 'level';
  };
  
  // Bot management preferences
  botManagement: {
    autoSelectNewBots: boolean;
    showPerformanceMetrics: boolean;
    groupByStatus: boolean;
    sortBy: 'name' | 'created' | 'status' | 'performance';
    sortOrder: 'asc' | 'desc';
  };
  
  // Responsive breakpoints
  breakpoint: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  isMobile: boolean;
  
  // Error states
  errors: {
    global: string | null;
    audio: string | null;
    bot: string | null;
    websocket: string | null;
  };
  
  // User preferences
  preferences: {
    language: string;
    dateFormat: string;
    timeFormat: '12h' | '24h';
    numberFormat: string;
    animations: boolean;
    soundEnabled: boolean;
  };
}

const initialState: UIState = {
  theme: 'light',
  
  sidebarOpen: true,
  sidebarCollapsed: false,
  activeTab: 'dashboard',
  activePage: '/',
  
  notifications: [],
  
  modals: {
    botSpawner: false,
    audioSettings: false,
    systemSettings: false,
    helpDialog: false,
    aboutDialog: false,
  },
  
  loading: {
    global: false,
    audio: false,
    bot: false,
    system: false,
  },
  
  dashboard: {
    autoRefresh: true,
    refreshInterval: 5000,
    compactMode: false,
    showAdvancedMetrics: false,
    selectedMetrics: ['audio', 'bots', 'system'],
  },
  
  audioTesting: {
    showAdvancedControls: false,
    autoStartRecording: false,
    showVisualization: true,
    visualizationType: 'waveform',
  },
  
  botManagement: {
    autoSelectNewBots: true,
    showPerformanceMetrics: true,
    groupByStatus: false,
    sortBy: 'created',
    sortOrder: 'desc',
  },
  
  breakpoint: 'lg',
  isMobile: false,
  
  errors: {
    global: null,
    audio: null,
    bot: null,
    websocket: null,
  },
  
  preferences: {
    language: 'en',
    dateFormat: 'MM/dd/yyyy',
    timeFormat: '12h',
    numberFormat: 'en-US',
    animations: true,
    soundEnabled: true,
  },
};

let notificationId = 0;

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    // Theme management
    setTheme: (state, action: PayloadAction<ThemeMode>) => {
      state.theme = action.payload;
      localStorage.setItem('theme', action.payload);
    },
    
    toggleTheme: (state) => {
      state.theme = state.theme === 'light' ? 'dark' : 'light';
      localStorage.setItem('theme', state.theme);
    },
    
    // Layout and navigation
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
    },
    
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload;
    },
    
    toggleSidebarCollapsed: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
    },
    
    setActiveTab: (state, action: PayloadAction<string>) => {
      state.activeTab = action.payload;
    },
    
    setActivePage: (state, action: PayloadAction<string>) => {
      state.activePage = action.payload;
    },
    
    // Notifications
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp'>>) => {
      const notification: Notification = {
        id: `notification-${notificationId++}`,
        timestamp: Date.now(),
        ...action.payload,
      };
      
      state.notifications.push(notification);
      
      // Auto-remove notifications if autoHide is true
      if (notification.autoHide !== false) {
        // This would be handled by middleware to remove after delay
      }
    },
    
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    
    clearNotifications: (state) => {
      state.notifications = [];
    },
    
    markNotificationAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        // Could add a 'read' property to notifications
      }
    },
    
    // Modals and dialogs
    openModal: (state, action: PayloadAction<keyof UIState['modals']>) => {
      state.modals[action.payload] = true;
    },
    
    closeModal: (state, action: PayloadAction<keyof UIState['modals']>) => {
      state.modals[action.payload] = false;
    },
    
    closeAllModals: (state) => {
      Object.keys(state.modals).forEach(key => {
        state.modals[key as keyof UIState['modals']] = false;
      });
    },
    
    // Loading states
    setLoading: (state, action: PayloadAction<{ scope: keyof UIState['loading']; loading: boolean }>) => {
      state.loading[action.payload.scope] = action.payload.loading;
    },
    
    setGlobalLoading: (state, action: PayloadAction<boolean>) => {
      state.loading.global = action.payload;
    },
    
    // Dashboard preferences
    updateDashboardPreferences: (state, action: PayloadAction<Partial<UIState['dashboard']>>) => {
      Object.assign(state.dashboard, action.payload);
    },
    
    toggleDashboardAutoRefresh: (state) => {
      state.dashboard.autoRefresh = !state.dashboard.autoRefresh;
    },
    
    setDashboardRefreshInterval: (state, action: PayloadAction<number>) => {
      state.dashboard.refreshInterval = action.payload;
    },
    
    toggleDashboardCompactMode: (state) => {
      state.dashboard.compactMode = !state.dashboard.compactMode;
    },
    
    // Audio testing preferences
    updateAudioTestingPreferences: (state, action: PayloadAction<Partial<UIState['audioTesting']>>) => {
      Object.assign(state.audioTesting, action.payload);
    },
    
    toggleAudioAdvancedControls: (state) => {
      state.audioTesting.showAdvancedControls = !state.audioTesting.showAdvancedControls;
    },
    
    setAudioVisualizationType: (state, action: PayloadAction<'waveform' | 'frequency' | 'level'>) => {
      state.audioTesting.visualizationType = action.payload;
    },
    
    // Bot management preferences
    updateBotManagementPreferences: (state, action: PayloadAction<Partial<UIState['botManagement']>>) => {
      Object.assign(state.botManagement, action.payload);
    },
    
    setBotSortBy: (state, action: PayloadAction<UIState['botManagement']['sortBy']>) => {
      state.botManagement.sortBy = action.payload;
    },
    
    setBotSortOrder: (state, action: PayloadAction<'asc' | 'desc'>) => {
      state.botManagement.sortOrder = action.payload;
    },
    
    toggleBotGroupByStatus: (state) => {
      state.botManagement.groupByStatus = !state.botManagement.groupByStatus;
    },
    
    // Responsive breakpoints
    setBreakpoint: (state, action: PayloadAction<UIState['breakpoint']>) => {
      state.breakpoint = action.payload;
      state.isMobile = action.payload === 'xs' || action.payload === 'sm';
      
      // Auto-collapse sidebar on mobile
      if (state.isMobile && state.sidebarOpen) {
        state.sidebarCollapsed = true;
      }
    },
    
    // Error handling
    setError: (state, action: PayloadAction<{ scope: keyof UIState['errors']; error: string | null }>) => {
      state.errors[action.payload.scope] = action.payload.error;
    },
    
    clearError: (state, action: PayloadAction<keyof UIState['errors']>) => {
      state.errors[action.payload] = null;
    },
    
    clearAllErrors: (state) => {
      Object.keys(state.errors).forEach(key => {
        state.errors[key as keyof UIState['errors']] = null;
      });
    },
    
    // User preferences
    updatePreferences: (state, action: PayloadAction<Partial<UIState['preferences']>>) => {
      Object.assign(state.preferences, action.payload);
      
      // Save to localStorage
      localStorage.setItem('userPreferences', JSON.stringify(state.preferences));
    },
    
    setLanguage: (state, action: PayloadAction<string>) => {
      state.preferences.language = action.payload;
      localStorage.setItem('language', action.payload);
    },
    
    toggleAnimations: (state) => {
      state.preferences.animations = !state.preferences.animations;
      localStorage.setItem('userPreferences', JSON.stringify(state.preferences));
    },
    
    toggleSound: (state) => {
      state.preferences.soundEnabled = !state.preferences.soundEnabled;
      localStorage.setItem('userPreferences', JSON.stringify(state.preferences));
    },
    
    // Initialize UI from localStorage
    initializeUI: (state) => {
      // Load theme
      const savedTheme = localStorage.getItem('theme') as ThemeMode;
      if (savedTheme) {
        state.theme = savedTheme;
      }
      
      // Load preferences
      const savedPreferences = localStorage.getItem('userPreferences');
      if (savedPreferences) {
        try {
          const preferences = JSON.parse(savedPreferences);
          Object.assign(state.preferences, preferences);
        } catch (error) {
          console.warn('Failed to parse saved preferences:', error);
        }
      }
      
      // Load language
      const savedLanguage = localStorage.getItem('language');
      if (savedLanguage) {
        state.preferences.language = savedLanguage;
      }
    },
    
    // Reset UI state
    resetUIState: () => initialState,
  },
});

export const {
  setTheme,
  toggleTheme,
  setSidebarOpen,
  toggleSidebar,
  setSidebarCollapsed,
  toggleSidebarCollapsed,
  setActiveTab,
  setActivePage,
  addNotification,
  removeNotification,
  clearNotifications,
  markNotificationAsRead,
  openModal,
  closeModal,
  closeAllModals,
  setLoading,
  setGlobalLoading,
  updateDashboardPreferences,
  toggleDashboardAutoRefresh,
  setDashboardRefreshInterval,
  toggleDashboardCompactMode,
  updateAudioTestingPreferences,
  toggleAudioAdvancedControls,
  setAudioVisualizationType,
  updateBotManagementPreferences,
  setBotSortBy,
  setBotSortOrder,
  toggleBotGroupByStatus,
  setBreakpoint,
  setError,
  clearError,
  clearAllErrors,
  updatePreferences,
  setLanguage,
  toggleAnimations,
  toggleSound,
  initializeUI,
  resetUIState,
} = uiSlice.actions;

export default uiSlice;