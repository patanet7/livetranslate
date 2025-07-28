/**
 * App Error Provider
 * 
 * Provides global error handling context and integrates all error handling
 * components together for comprehensive error management
 */

import React, { createContext, useContext, ReactNode } from 'react';
import { SnackbarProvider } from 'notistack';
import { CriticalErrorBoundary } from './ErrorBoundary';
import useErrorHandler from '@/hooks/useErrorHandler';
import useOfflineHandler from '@/hooks/useOfflineHandler';

interface AppErrorContextType {
  handleError: ReturnType<typeof useErrorHandler>['handleError'];
  handleNetworkError: ReturnType<typeof useErrorHandler>['handleNetworkError'];
  handleApiError: ReturnType<typeof useErrorHandler>['handleApiError'];
  handleValidationError: ReturnType<typeof useErrorHandler>['handleValidationError'];
  handleCriticalError: ReturnType<typeof useErrorHandler>['handleCriticalError'];
  isOnline: boolean;
  queueSize: number;
  fetchWithOfflineSupport: ReturnType<typeof useOfflineHandler>['fetchWithOfflineSupport'];
}

const AppErrorContext = createContext<AppErrorContextType | null>(null);

interface AppErrorProviderProps {
  children: ReactNode;
}

const AppErrorProviderInner: React.FC<AppErrorProviderProps> = ({ children }) => {
  const errorHandler = useErrorHandler();
  const offlineHandler = useOfflineHandler({
    enableQueue: true,
    maxQueueSize: 100,
    maxRetries: 3,
    showNotifications: true,
  });

  const contextValue: AppErrorContextType = {
    ...errorHandler,
    isOnline: offlineHandler.isOnline,
    queueSize: offlineHandler.queueSize,
    fetchWithOfflineSupport: offlineHandler.fetchWithOfflineSupport,
  };

  return (
    <AppErrorContext.Provider value={contextValue}>
      {children}
    </AppErrorContext.Provider>
  );
};

export const AppErrorProvider: React.FC<AppErrorProviderProps> = ({ children }) => {
  return (
    <CriticalErrorBoundary>
      <SnackbarProvider
        maxSnack={5}
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        autoHideDuration={5000}
        preventDuplicate
        dense
      >
        <AppErrorProviderInner>
          {children}
        </AppErrorProviderInner>
      </SnackbarProvider>
    </CriticalErrorBoundary>
  );
};

export const useAppError = (): AppErrorContextType => {
  const context = useContext(AppErrorContext);
  if (!context) {
    throw new Error('useAppError must be used within an AppErrorProvider');
  }
  return context;
};

export default AppErrorProvider;