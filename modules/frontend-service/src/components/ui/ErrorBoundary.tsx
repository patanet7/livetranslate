import { Component, ErrorInfo, ReactNode } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Divider,
  Alert,
  Stack,
  Chip,
  Collapse,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh,
  BugReport,
  ContentCopy,
  Home as HomeIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
} from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  level?: 'page' | 'component' | 'critical';
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  enableRetry?: boolean;
  enableReporting?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
  retryCount: number;
  showDetails: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  private maxRetries = 3;
  private retryTimeout?: NodeJS.Timeout;

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
      retryCount: 0,
      showDetails: false,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
      errorId: `error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo,
    });

    // Call custom error handler
    this.props.onError?.(error, errorInfo);

    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.group('🚨 Error Boundary Caught Error');
      console.error('Error:', error);
      console.error('Error Info:', errorInfo);
      console.error('Component Stack:', errorInfo.componentStack);
      console.groupEnd();
    }

    // Report error to monitoring service
    this.reportError(error, errorInfo);
  }

  private reportError = (error: Error, errorInfo: ErrorInfo) => {
    if (!this.props.enableReporting) return;

    try {
      // In a real app, you'd send this to your error reporting service
      const errorReport = {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        level: this.props.level || 'component',
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      };

      // Example: Send to error reporting service
      // errorReportingService.captureException(error, errorReport);
      
      console.warn('Error reported:', errorReport);
    } catch (reportingError) {
      console.error('Failed to report error:', reportingError);
    }
  };

  private handleRetry = () => {
    if (this.state.retryCount >= this.maxRetries) {
      return;
    }

    this.setState(prevState => ({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: prevState.retryCount + 1,
    }));

    // Add a small delay to prevent immediate re-errors
    this.retryTimeout = setTimeout(() => {
      // Force re-render
      this.forceUpdate();
    }, 1000);
  };

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  private toggleDetails = () => {
    this.setState(prevState => ({
      showDetails: !prevState.showDetails,
    }));
  };

  handleCopyError = async () => {
    const errorDetails = {
      errorId: this.state.errorId,
      error: this.state.error?.toString(),
      stack: this.state.error?.stack,
      componentStack: this.state.errorInfo?.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
    };

    try {
      await navigator.clipboard.writeText(JSON.stringify(errorDetails, null, 2));
      alert('Error details copied to clipboard');
    } catch (err) {
      console.error('Failed to copy error details:', err);
    }
  };

  componentWillUnmount() {
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
    }
  }

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { level = 'component', enableRetry = true } = this.props;
      const { error, retryCount, showDetails } = this.state;
      const canRetry = enableRetry && retryCount < this.maxRetries;

      // Different UI based on error level
      const severity = level === 'critical' ? 'error' : level === 'page' ? 'warning' : 'info';
      const title = level === 'critical' 
        ? 'Application Error' 
        : level === 'page' 
        ? 'Page Error' 
        : 'Component Error';

      const description = level === 'critical'
        ? 'A critical error has occurred. The application may not function properly.'
        : level === 'page'
        ? 'An error occurred while loading this page.'
        : 'A component failed to render properly.';

      return (
        <Box
          sx={{
            p: 3,
            textAlign: 'center',
            minHeight: level === 'critical' ? '100vh' : 'auto',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: level === 'critical' ? 'center' : 'flex-start',
            alignItems: 'center',
          }}
        >
          <Paper 
            elevation={level === 'critical' ? 8 : 2} 
            sx={{ 
              p: 4, 
              maxWidth: 600, 
              width: '100%',
              textAlign: 'left'
            }}
          >
            <Stack spacing={3}>
              {/* Header */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <ErrorIcon color={severity} sx={{ fontSize: 40 }} />
                <Box>
                  <Typography variant="h5" color={`${severity}.main`} gutterBottom>
                    {title}
                  </Typography>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Chip 
                      label={`Level: ${level}`} 
                      size="small" 
                      color={severity}
                      variant="outlined"
                    />
                    {retryCount > 0 && (
                      <Chip 
                        label={`Retry: ${retryCount}/${this.maxRetries}`} 
                        size="small" 
                        color="warning"
                        variant="outlined"
                      />
                    )}
                  </Stack>
                </Box>
              </Box>

              {/* Error Alert */}
              <Alert severity={severity}>
                <Typography variant="body1" gutterBottom>
                  {description}
                </Typography>
                {error && (
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', mt: 1 }}>
                    {error.message}
                  </Typography>
                )}
              </Alert>

              {/* Action Buttons */}
              <Stack direction="row" spacing={2} justifyContent="center">
                {canRetry && (
                  <Button
                    variant="contained"
                    startIcon={<Refresh />}
                    onClick={this.handleRetry}
                    color="primary"
                  >
                    Try Again ({this.maxRetries - retryCount} remaining)
                  </Button>
                )}
                
                {level !== 'critical' && (
                  <Button
                    variant="outlined"
                    startIcon={<HomeIcon />}
                    onClick={this.handleGoHome}
                  >
                    Go Home
                  </Button>
                )}
                
                <Button
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={this.handleReload}
                  color="warning"
                >
                  Reload Page
                </Button>
              </Stack>

              <Divider />

              {/* Error Details Toggle */}
              <Box>
                <Button
                  variant="text"
                  startIcon={showDetails ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  onClick={this.toggleDetails}
                  size="small"
                  color="inherit"
                >
                  {showDetails ? 'Hide' : 'Show'} Technical Details
                </Button>
                
                <Collapse in={showDetails}>
                  <Box sx={{ mt: 2 }}>
                    <Stack spacing={2}>
                      <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50' }}>
                        <Typography variant="subtitle2" color="error" gutterBottom>
                          Error ID: {this.state.errorId}
                        </Typography>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-word' }}>
                          {error?.message}
                        </Typography>
                      </Paper>

                      {error?.stack && (
                        <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50' }}>
                          <Typography variant="subtitle2" color="error" gutterBottom>
                            Stack Trace:
                          </Typography>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              fontFamily: 'monospace', 
                              fontSize: '0.75rem',
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-word',
                              maxHeight: 200,
                              overflow: 'auto'
                            }}
                          >
                            {error.stack}
                          </Typography>
                        </Paper>
                      )}

                      <Button
                        variant="outlined"
                        startIcon={<ContentCopy />}
                        onClick={this.handleCopyError}
                        size="small"
                        sx={{ alignSelf: 'flex-start' }}
                      >
                        Copy Error Details
                      </Button>
                    </Stack>
                  </Box>
                </Collapse>
              </Box>

              {/* Development Info */}
              {process.env.NODE_ENV === 'development' && (
                <Alert severity="info" icon={<BugReport />}>
                  <Typography variant="body2">
                    <strong>Development Mode:</strong> This error boundary is showing detailed 
                    information because you're in development mode. In production, users 
                    would see a more user-friendly error message.
                  </Typography>
                </Alert>
              )}
            </Stack>
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}

// Higher-order component for easy wrapping
export const withErrorBoundary = <P extends object>(
  WrappedComponent: React.ComponentType<P>,
  errorBoundaryProps?: Omit<Props, 'children'>
) => {
  const WithErrorBoundaryComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <WrappedComponent {...props} />
    </ErrorBoundary>
  );

  WithErrorBoundaryComponent.displayName = `withErrorBoundary(${WrappedComponent.displayName || WrappedComponent.name})`;
  
  return WithErrorBoundaryComponent;
};

// Specialized error boundaries
export const PageErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <ErrorBoundary level="page" enableRetry={true} enableReporting={true}>
    {children}
  </ErrorBoundary>
);

export const ComponentErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <ErrorBoundary level="component" enableRetry={true} enableReporting={false}>
    {children}
  </ErrorBoundary>
);

export const CriticalErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <ErrorBoundary level="critical" enableRetry={false} enableReporting={true}>
    {children}
  </ErrorBoundary>
);

export default ErrorBoundary;