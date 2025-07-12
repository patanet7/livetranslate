import { Component, ErrorInfo, ReactNode } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Stack,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh,
  BugReport,
  ExpandMore,
  ContentCopy,
} from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
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
    // Log the error to error reporting service
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo,
    });

    // Report to external error tracking service
    if (process.env.NODE_ENV === 'production') {
      // Example: Sentry.captureException(error, { extra: errorInfo });
    }
  }

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
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

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'background.default',
            padding: 3,
          }}
        >
          <Paper
            elevation={3}
            sx={{
              maxWidth: 600,
              width: '100%',
              padding: 4,
              textAlign: 'center',
            }}
          >
            {/* Error icon and title */}
            <Box sx={{ marginBottom: 3 }}>
              <ErrorIcon 
                sx={{ 
                  fontSize: 64, 
                  color: 'error.main',
                  marginBottom: 2,
                }} 
              />
              <Typography variant="h4" component="h1" gutterBottom>
                Oops! Something went wrong
              </Typography>
              <Typography variant="body1" color="text.secondary">
                We're sorry, but an unexpected error occurred. Please try refreshing the page or contact support if the problem persists.
              </Typography>
            </Box>

            {/* Error ID */}
            <Alert severity="error" sx={{ marginBottom: 3, textAlign: 'left' }}>
              <Typography variant="body2">
                <strong>Error ID:</strong> {this.state.errorId}
              </Typography>
              <Typography variant="body2">
                <strong>Time:</strong> {new Date().toLocaleString()}
              </Typography>
            </Alert>

            {/* Action buttons */}
            <Stack direction="row" spacing={2} justifyContent="center" sx={{ marginBottom: 3 }}>
              <Button
                variant="contained"
                startIcon={<Refresh />}
                onClick={this.handleReload}
                size="large"
              >
                Refresh Page
              </Button>
              <Button
                variant="outlined"
                onClick={this.handleGoHome}
                size="large"
              >
                Go to Home
              </Button>
            </Stack>

            <Divider sx={{ marginY: 3 }} />

            {/* Error details (expandable) */}
            <Accordion elevation={0}>
              <AccordionSummary
                expandIcon={<ExpandMore />}
                aria-controls="error-details-content"
                id="error-details-header"
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <BugReport fontSize="small" />
                  <Typography>Technical Details</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Stack spacing={2} sx={{ textAlign: 'left' }}>
                  {/* Error message */}
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Error Message:
                    </Typography>
                    <Paper 
                      variant="outlined" 
                      sx={{ 
                        padding: 2, 
                        backgroundColor: 'grey.50',
                        fontFamily: 'monospace',
                        fontSize: '0.875rem',
                      }}
                    >
                      {this.state.error?.toString()}
                    </Paper>
                  </Box>

                  {/* Stack trace */}
                  {this.state.error?.stack && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Stack Trace:
                      </Typography>
                      <Paper 
                        variant="outlined" 
                        sx={{ 
                          padding: 2, 
                          backgroundColor: 'grey.50',
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          maxHeight: 200,
                          overflow: 'auto',
                          whiteSpace: 'pre-wrap',
                        }}
                      >
                        {this.state.error.stack}
                      </Paper>
                    </Box>
                  )}

                  {/* Component stack */}
                  {this.state.errorInfo?.componentStack && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Component Stack:
                      </Typography>
                      <Paper 
                        variant="outlined" 
                        sx={{ 
                          padding: 2, 
                          backgroundColor: 'grey.50',
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          maxHeight: 200,
                          overflow: 'auto',
                          whiteSpace: 'pre-wrap',
                        }}
                      >
                        {this.state.errorInfo.componentStack}
                      </Paper>
                    </Box>
                  )}

                  {/* Copy button */}
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
              </AccordionDetails>
            </Accordion>

            {/* Help text */}
            <Typography variant="body2" color="text.secondary" sx={{ marginTop: 3 }}>
              If this error continues to occur, please share the error ID and details with our support team.
            </Typography>
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}