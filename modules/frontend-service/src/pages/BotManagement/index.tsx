import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Tabs,
  Tab,
  Container,
  Alert,
  Snackbar,
  Fab,
  Badge,
  Chip,
  Stack,
} from '@mui/material';
import {
  Add as AddIcon,
  SmartToy as BotIcon,
  Videocam as VideocamIcon,
  Storage as StorageIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '@/store/hooks';
import { BotSpawner } from './components/BotSpawner';
import { ActiveBots } from './components/ActiveBots';
import { VirtualWebcam } from './components/VirtualWebcam';
import { SessionDatabase } from './components/SessionDatabase';
import { BotAnalytics } from './components/BotAnalytics';
import { BotSettings } from './components/BotSettings';
import { CreateBotModal } from './components/CreateBotModal';
import { useBotManager } from '@/hooks/useBotManager';
import { TabPanel } from '@/components/ui';

const BotManagement: React.FC = () => {
  const { bots, activeBotIds, systemStats } = useAppSelector(state => state.bot);
  const [tabValue, setTabValue] = useState(0);
  const [createBotModalOpen, setCreateBotModalOpen] = useState(false);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({ open: false, message: '', severity: 'info' });

  const { 
 
    terminateBot, 
    getSystemStats, 
    refreshBots,
    error 
  } = useBotManager();

  useEffect(() => {
    // Initial data load
    refreshBots();
    getSystemStats();

    // Set up real-time updates
    const interval = setInterval(() => {
      getSystemStats();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleCreateBot = () => {
    setCreateBotModalOpen(true);
  };

  const handleCloseNotification = () => {
    setNotification(prev => ({ ...prev, open: false }));
  };

  const showNotification = (message: string, severity: 'success' | 'error' | 'warning' | 'info') => {
    setNotification({ open: true, message, severity });
  };

  const activeBotCount = activeBotIds.length;
  const totalBots = Object.keys(bots).length;

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <div>
            <Typography variant="h4" component="h1" gutterBottom>
              🤖 Bot Management Dashboard
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Google Meet bot lifecycle management, virtual webcam control, and real-time monitoring
            </Typography>
          </div>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <Stack direction="row" spacing={1}>
              <Chip 
                icon={<BotIcon />} 
                label={`${activeBotCount} Active`} 
                color="primary" 
                variant="outlined"
              />
              <Chip 
                label={`${totalBots} Total`} 
                color="secondary" 
                variant="outlined"
              />
              <Chip 
                label={`${systemStats.completedSessions} Completed`} 
                color="success" 
                variant="outlined"
              />
            </Stack>
          </Box>
        </Box>

        {/* Status Alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* Main Navigation Tabs */}
        <Paper sx={{ width: '100%', mb: 3 }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            variant="fullWidth"
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab 
              icon={<Badge badgeContent={activeBotCount} color="primary"><BotIcon /></Badge>} 
              label="Active Bots" 
            />
            <Tab 
              icon={<VideocamIcon />} 
              label="Virtual Webcam" 
            />
            <Tab 
              icon={<StorageIcon />} 
              label="Session Database" 
            />
            <Tab 
              icon={<AnalyticsIcon />} 
              label="Analytics" 
            />
            <Tab 
              icon={<SettingsIcon />} 
              label="Settings" 
            />
          </Tabs>
        </Paper>
      </Box>

      {/* Tab Panels */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} lg={4}>
            <BotSpawner 
              onBotSpawned={(botId) => {
                showNotification(`Bot ${botId} spawned successfully`, 'success');
                refreshBots();
              }}
              onError={(error) => showNotification(error, 'error')}
            />
          </Grid>
          <Grid item xs={12} lg={8}>
            <ActiveBots 
              bots={bots}
              activeBotIds={activeBotIds}
              onTerminateBot={(botId) => {
                terminateBot(botId);
                showNotification(`Bot ${botId} terminated`, 'info');
              }}
              onBotError={(botId, error) => {
                showNotification(`Bot ${botId} error: ${error}`, 'error');
              }}
            />
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <VirtualWebcam 
          bots={bots}
          activeBotIds={activeBotIds}
          onWebcamUpdate={(botId, config) => {
            showNotification(`Webcam updated for bot ${botId}`, 'success');
          }}
        />
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <SessionDatabase 
          onError={(error) => showNotification(error, 'error')}
        />
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        <BotAnalytics 
          systemStats={systemStats}
          bots={bots}
          onRefresh={getSystemStats}
        />
      </TabPanel>

      <TabPanel value={tabValue} index={4}>
        <BotSettings 
          onSettingsUpdate={(settings) => {
            showNotification('Settings updated successfully', 'success');
          }}
        />
      </TabPanel>

      {/* Floating Action Button */}
      <Fab 
        color="primary" 
        aria-label="create bot"
        onClick={handleCreateBot}
        sx={{ position: 'fixed', bottom: 24, right: 24 }}
      >
        <AddIcon />
      </Fab>

      {/* Create Bot Modal */}
      <CreateBotModal 
        open={createBotModalOpen}
        onClose={() => setCreateBotModalOpen(false)}
        onBotCreated={(botId) => {
          showNotification(`Bot ${botId} created successfully`, 'success');
          refreshBots();
          setCreateBotModalOpen(false);
        }}
        onError={(error) => showNotification(error, 'error')}
      />

      {/* Notification Snackbar */}
      <Snackbar 
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default BotManagement;