import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Alert,
} from '@mui/material';
import { ConnectionIndicator } from '@/components/ui/ConnectionIndicator';
import { useAppSelector } from '@/store';

const WebSocketTest: React.FC = () => {
  const { connection } = useAppSelector(state => state.websocket);

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <Typography variant="h4" component="h1">
          WebSocket Testing
        </Typography>
        <ConnectionIndicator 
          isConnected={connection.isConnected}
          reconnectAttempts={connection.reconnectAttempts}
          showLabel
        />
      </Box>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Real-time WebSocket connection testing and message monitoring
      </Typography>

      <Card>
        <CardContent>
          <Alert severity="info">
            WebSocket Testing interface is being migrated with enhanced real-time capabilities.
            This will include connection management, message testing, and comprehensive logging.
          </Alert>
        </CardContent>
      </Card>
    </Box>
  );
};

export default WebSocketTest;