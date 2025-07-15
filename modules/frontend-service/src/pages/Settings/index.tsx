import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Card,
  CardContent,
  Container,
  Alert,
  Snackbar,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '@/store';
import AudioProcessingSettings from './components/AudioProcessingSettings';
import ChunkingSettings from './components/ChunkingSettings';
import CorrelationSettings from './components/CorrelationSettings';
import TranslationSettings from './components/TranslationSettings';
import { PromptManagementSettings } from './components/PromptManagementSettings';
import SystemSettings from './components/SystemSettings';
import BotSettings from './components/BotSettings';
import ConfigSyncSettings from './components/ConfigSyncSettings';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `settings-tab-${index}`,
    'aria-controls': `settings-tabpanel-${index}`,
  };
}

const Settings: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [saveNotification, setSaveNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error';
  }>({ open: false, message: '', severity: 'success' });

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleSettingsSaved = (message: string, success: boolean = true) => {
    setSaveNotification({
      open: true,
      message,
      severity: success ? 'success' : 'error',
    });
  };

  const handleCloseNotification = () => {
    setSaveNotification(prev => ({ ...prev, open: false }));
  };

  return (
    <Container maxWidth="lg">
      <Box>
        <Typography variant="h4" component="h1" gutterBottom>
          Settings
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Comprehensive system configuration, service settings, and hyperparameter tuning
        </Typography>

        <Card>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs 
              value={activeTab} 
              onChange={handleTabChange} 
              aria-label="settings tabs"
              variant="scrollable"
              scrollButtons="auto"
            >
              <Tab label="Audio Processing" {...a11yProps(0)} />
              <Tab label="Chunking" {...a11yProps(1)} />
              <Tab label="Speaker Correlation" {...a11yProps(2)} />
              <Tab label="Translation" {...a11yProps(3)} />
              <Tab label="Prompt Management" {...a11yProps(4)} />
              <Tab label="Bot Management" {...a11yProps(5)} />
              <Tab label="Config Sync" {...a11yProps(6)} />
              <Tab label="System" {...a11yProps(7)} />
            </Tabs>
          </Box>

          <CardContent>
            <TabPanel value={activeTab} index={0}>
              <AudioProcessingSettings onSave={handleSettingsSaved} />
            </TabPanel>
            
            <TabPanel value={activeTab} index={1}>
              <ChunkingSettings onSave={handleSettingsSaved} />
            </TabPanel>
            
            <TabPanel value={activeTab} index={2}>
              <CorrelationSettings onSave={handleSettingsSaved} />
            </TabPanel>
            
            <TabPanel value={activeTab} index={3}>
              <TranslationSettings onSave={handleSettingsSaved} />
            </TabPanel>
            
            <TabPanel value={activeTab} index={4}>
              <PromptManagementSettings />
            </TabPanel>
            
            <TabPanel value={activeTab} index={5}>
              <BotSettings onSave={handleSettingsSaved} />
            </TabPanel>
            
            <TabPanel value={activeTab} index={6}>
              <ConfigSyncSettings onSave={handleSettingsSaved} />
            </TabPanel>
            
            <TabPanel value={activeTab} index={7}>
              <SystemSettings onSave={handleSettingsSaved} />
            </TabPanel>
          </CardContent>
        </Card>

        <Snackbar
          open={saveNotification.open}
          autoHideDuration={4000}
          onClose={handleCloseNotification}
          message={saveNotification.message}
        />
      </Box>
    </Container>
  );
};

export default Settings;