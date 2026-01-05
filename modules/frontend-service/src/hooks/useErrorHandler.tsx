/**
 * Global Error Handler Hook
 * 
 * Provides comprehensive error handling with user-friendly notifications,
 * automatic retry mechanisms, and error reporting
 */

import { useCallback, useEffect } from 'react';
import { useSnackbar } from 'notistack';
import { useNotifications } from './useNotifications';

// Error types and classifications
export interface AppError {
  type: 'network' | 'api' | 'validation' | 'auth' | 'unknown';
  code?: string | number;
  message: string;
  details?: any;
  timestamp: Date;
  source?: string;
  recoverable?: boolean;
  userMessage?: string;
}

export interface ErrorHandlerOptions {
  showNotification?: boolean;
  logError?: boolean;
  reportError?: boolean;
  showRetry?: boolean;
  source?: string;
}

export const useErrorHandler = () => {
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const { notifyError, notifyWarning } = useNotifications();

  // Error classification
  const classifyError = useCallback((error: any): AppError => {
    const timestamp = new Date();
    
    // RTK Query errors
    if (error.status !== undefined) {
      if (error.status === 'FETCH_ERROR') {
        return {
          type: 'network',
          code: 'NETWORK_ERROR',
          message: 'Network connection failed',
          userMessage: 'Unable to connect to the server. Please check your internet connection.',
          details: error,
          timestamp,
          recoverable: true,
        };
      }
      
      if (error.status === 'TIMEOUT_ERROR') {
        return {
          type: 'network',
          code: 'TIMEOUT',
          message: 'Request timed out',
          userMessage: 'The request took too long to complete. Please try again.',
          details: error,
          timestamp,
          recoverable: true,
        };
      }
      
      if (error.status === 401) {
        return {
          type: 'auth',
          code: 401,
          message: 'Unauthorized access',
          userMessage: 'Your session has expired. Please refresh the page.',
          details: error,
          timestamp,
          recoverable: false,
        };
      }
      
      if (error.status === 403) {
        return {
          type: 'auth',
          code: 403,
          message: 'Access forbidden',
          userMessage: 'You do not have permission to perform this action.',
          details: error,
          timestamp,
          recoverable: false,
        };
      }
      
      if (error.status === 404) {
        return {
          type: 'api',
          code: 404,
          message: 'Resource not found',
          userMessage: 'The requested resource was not found.',
          details: error,
          timestamp,
          recoverable: false,
        };
      }
      
      if (error.status >= 400 && error.status < 500) {
        return {
          type: 'validation',
          code: error.status,
          message: error.data?.message || 'Client error occurred',
          userMessage: error.data?.message || 'There was an issue with your request.',
          details: error,
          timestamp,
          recoverable: true,
        };
      }
      
      if (error.status >= 500) {
        return {
          type: 'api',
          code: error.status,
          message: error.data?.message || 'Server error occurred',
          userMessage: 'A server error occurred. Our team has been notified.',
          details: error,
          timestamp,
          recoverable: true,
        };
      }
    }
    
    // JavaScript errors
    if (error instanceof Error) {
      return {
        type: 'unknown',
        code: error.name,
        message: error.message,
        userMessage: 'An unexpected error occurred. Please try again.',
        details: { stack: error.stack },
        timestamp,
        recoverable: true,
      };
    }
    
    // Generic errors
    return {
      type: 'unknown',
      code: 'UNKNOWN',
      message: typeof error === 'string' ? error : 'Unknown error occurred',
      userMessage: 'An unexpected error occurred. Please try again.',
      details: error,
      timestamp,
      recoverable: true,
    };
  }, []);

  // Error reporting
  const reportError = useCallback((appError: AppError, options: ErrorHandlerOptions) => {
    if (!options.reportError) return;

    try {
      const errorReport = {
        ...appError,
        source: options.source,
        userAgent: navigator.userAgent,
        url: window.location.href,
        sessionId: sessionStorage.getItem('sessionId'),
      };

      // In a real app, send to error reporting service
      console.group('🚨 Error Report');
      console.error('Error:', errorReport);
      console.groupEnd();

      // Example: Send to external service
      // errorReportingService.captureException(appError, errorReport);
    } catch (reportingError) {
      console.error('Failed to report error:', reportingError);
    }
  }, []);

  // Handle errors with notifications and recovery options
  const handleError = useCallback((
    error: any,
    options: ErrorHandlerOptions = {}
  ) => {
    const {
      showNotification = true,
      logError = true,
      reportError: shouldReport = true,
      showRetry = false,
      source = 'unknown',
    } = options;

    const appError = classifyError(error);
    appError.source = source;

    // Log error
    if (logError) {
      console.error(`[${source}] Error:`, appError);
    }

    // Report error
    if (shouldReport && (appError.type === 'api' || appError.type === 'unknown')) {
      reportError(appError, options);
    }

    // Show notification
    if (showNotification) {
      const severity = 
        appError.type === 'network' ? 'warning' :
        appError.type === 'auth' ? 'error' :
        appError.type === 'validation' ? 'warning' :
        'error';

      enqueueSnackbar(appError.userMessage || appError.message, {
        variant: severity,
        persist: severity === 'error',
        action: showRetry && appError.recoverable ? (key) => (
          <button
            onClick={() => {
              closeSnackbar(key);
              // Retry logic would be handled by the calling component
            }}
            style={{
              color: 'white',
              background: 'none',
              border: '1px solid white',
              padding: '4px 8px',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Retry
          </button>
        ) : undefined,
      });

      // Add to global notifications store
      const title = `${appError.type.charAt(0).toUpperCase() + appError.type.slice(1)} Error`;
      const message = appError.userMessage || appError.message;
      const autoHide = severity !== 'error';

      if (severity === 'error') {
        notifyError(title, message, autoHide);
      } else {
        notifyWarning(title, message, autoHide);
      }
    }

    return appError;
  }, [classifyError, reportError, enqueueSnackbar, closeSnackbar, notifyError, notifyWarning]);

  // Specialized error handlers
  const handleNetworkError = useCallback((error: any, source?: string) => {
    return handleError(error, {
      showNotification: true,
      logError: true,
      reportError: false, // Network errors are usually temporary
      showRetry: true,
      source: source || 'network',
    });
  }, [handleError]);

  const handleApiError = useCallback((error: any, source?: string) => {
    return handleError(error, {
      showNotification: true,
      logError: true,
      reportError: true,
      showRetry: error.status >= 500, // Only show retry for server errors
      source: source || 'api',
    });
  }, [handleError]);

  const handleValidationError = useCallback((error: any, source?: string) => {
    return handleError(error, {
      showNotification: true,
      logError: false, // Validation errors are expected
      reportError: false,
      showRetry: false,
      source: source || 'validation',
    });
  }, [handleError]);

  const handleCriticalError = useCallback((error: any, source?: string) => {
    return handleError(error, {
      showNotification: true,
      logError: true,
      reportError: true,
      showRetry: false,
      source: source || 'critical',
    });
  }, [handleError]);

  // Global error handler for unhandled promise rejections
  useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', event.reason);
      handleError(event.reason, {
        source: 'unhandledRejection',
        reportError: true,
      });
    };

    const handleGlobalError = (event: ErrorEvent) => {
      console.error('Global error:', event.error);
      handleError(event.error, {
        source: 'globalError',
        reportError: true,
      });
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    window.addEventListener('error', handleGlobalError);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      window.removeEventListener('error', handleGlobalError);
    };
  }, [handleError]);

  return {
    handleError,
    handleNetworkError,
    handleApiError,
    handleValidationError,
    handleCriticalError,
    classifyError,
  };
};

export default useErrorHandler;