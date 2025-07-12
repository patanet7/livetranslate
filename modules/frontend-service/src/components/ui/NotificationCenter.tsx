import React from 'react';
import {
  Snackbar,
  Alert,
  AlertTitle,
  Slide,
  IconButton,
  Box,
  Typography,
  Button,
} from '@mui/material';
import { Close } from '@mui/icons-material';
import { TransitionProps } from '@mui/material/transitions';
import { useAppDispatch, useAppSelector } from '@/store';
import { removeNotification } from '@/store/slices/uiSlice';

function SlideTransition(props: TransitionProps & {
  children: React.ReactElement;
}) {
  return <Slide {...props} direction="left" />;
}

export const NotificationCenter: React.FC = () => {
  const dispatch = useAppDispatch();
  const { notifications } = useAppSelector(state => state.ui);

  const handleClose = (notificationId: string) => {
    dispatch(removeNotification(notificationId));
  };

  // Only show the most recent notification
  const currentNotification = notifications[notifications.length - 1];

  if (!currentNotification) {
    return null;
  }

  return (
    <Snackbar
      open={true}
      autoHideDuration={currentNotification.autoHide !== false ? 6000 : null}
      onClose={() => handleClose(currentNotification.id)}
      anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      TransitionComponent={SlideTransition}
      sx={{ 
        marginTop: 8, // Account for AppBar height
        zIndex: theme => theme.zIndex.snackbar,
      }}
    >
      <Alert
        severity={currentNotification.type}
        variant="filled"
        action={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {/* Custom actions */}
            {currentNotification.actions?.map((action, index) => (
              <Button
                key={index}
                color="inherit"
                size="small"
                onClick={action.action}
                sx={{ 
                  minWidth: 'auto',
                  fontSize: '0.75rem',
                  padding: '2px 8px',
                }}
              >
                {action.label}
              </Button>
            ))}
            
            {/* Close button */}
            <IconButton
              size="small"
              aria-label="close"
              color="inherit"
              onClick={() => handleClose(currentNotification.id)}
            >
              <Close fontSize="small" />
            </IconButton>
          </Box>
        }
        sx={{
          maxWidth: 400,
          '& .MuiAlert-message': {
            overflow: 'hidden',
          },
        }}
      >
        {currentNotification.title && (
          <AlertTitle sx={{ fontSize: '0.875rem', fontWeight: 600 }}>
            {currentNotification.title}
          </AlertTitle>
        )}
        <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
          {currentNotification.message}
        </Typography>
        
        {/* Show notification count if there are multiple */}
        {notifications.length > 1 && (
          <Typography 
            variant="caption" 
            sx={{ 
              display: 'block', 
              marginTop: 1, 
              opacity: 0.8,
              fontSize: '0.7rem',
            }}
          >
            {notifications.length - 1} more notification{notifications.length > 2 ? 's' : ''}
          </Typography>
        )}
      </Alert>
    </Snackbar>
  );
};