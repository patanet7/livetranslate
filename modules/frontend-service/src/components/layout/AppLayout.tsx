import React from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Tooltip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Brightness4,
  Brightness7,
  Notifications,
  Settings,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '@/store';
import { 
  toggleSidebar, 
  toggleTheme, 
  openModal 
} from '@/store/slices/uiSlice';
import { Sidebar } from './Sidebar';
import { NotificationCenter } from '../ui/NotificationCenter';
import { ConnectionIndicator } from '../ui/ConnectionIndicator';

interface AppLayoutProps {
  children: React.ReactNode;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const theme = useTheme();
  const dispatch = useAppDispatch();
  
  const { 
    sidebarOpen, 
    sidebarCollapsed, 
    theme: currentTheme,
    notifications,
    isMobile 
  } = useAppSelector(state => state.ui);
  
  const { 
    isConnected, 
    reconnectAttempts 
  } = useAppSelector(state => state.websocket.connection);
  
  const { 
    activeBots 
  } = useAppSelector(state => state.bot.systemStats);

  const handleToggleSidebar = () => {
    dispatch(toggleSidebar());
  };

  const handleToggleTheme = () => {
    dispatch(toggleTheme());
  };

  const handleOpenSettings = () => {
    dispatch(openModal('systemSettings'));
  };

  const handleOpenNotifications = () => {
    // Could open a notifications panel
  };

  const sidebarWidth = sidebarCollapsed ? 72 : 240;
  const appBarHeight = 64;

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          ...(sidebarOpen && !isMobile && {
            marginLeft: sidebarWidth,
            width: `calc(100% - ${sidebarWidth}px)`,
            transition: theme.transitions.create(['width', 'margin'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
          }),
        }}
      >
        <Toolbar sx={{ height: appBarHeight }}>
          {/* Menu button */}
          <IconButton
            color="inherit"
            aria-label="toggle sidebar"
            onClick={handleToggleSidebar}
            edge="start"
            sx={{ 
              marginRight: 2,
              ...(sidebarOpen && !isMobile && { display: 'none' }),
            }}
          >
            <MenuIcon />
          </IconButton>

          {/* App title and status */}
          <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h6" component="h1" sx={{ fontWeight: 600 }}>
              LiveTranslate Orchestration
            </Typography>
            
            {/* Active bots indicator */}
            {activeBots > 0 && (
              <Badge 
                badgeContent={activeBots} 
                color="primary" 
                sx={{
                  '& .MuiBadge-badge': {
                    backgroundColor: theme.palette.success.main,
                    color: theme.palette.success.contrastText,
                  }
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  Active Bots
                </Typography>
              </Badge>
            )}
          </Box>

          {/* Right side controls */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {/* Connection indicator */}
            <ConnectionIndicator 
              isConnected={isConnected}
              reconnectAttempts={reconnectAttempts}
            />

            {/* Notifications */}
            <Tooltip title="Notifications">
              <IconButton
                color="inherit"
                onClick={handleOpenNotifications}
                aria-label="notifications"
              >
                <Badge badgeContent={notifications.length} color="error">
                  <Notifications />
                </Badge>
              </IconButton>
            </Tooltip>

            {/* Theme toggle */}
            <Tooltip title={`Switch to ${currentTheme === 'light' ? 'dark' : 'light'} mode`}>
              <IconButton
                color="inherit"
                onClick={handleToggleTheme}
                aria-label="toggle theme"
              >
                {currentTheme === 'light' ? <Brightness4 /> : <Brightness7 />}
              </IconButton>
            </Tooltip>

            {/* Settings */}
            <Tooltip title="Settings">
              <IconButton
                color="inherit"
                onClick={handleOpenSettings}
                aria-label="settings"
              >
                <Settings />
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Sidebar */}
      <Sidebar />

      {/* Main content area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          transition: theme.transitions.create('margin', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          marginTop: `${appBarHeight}px`,
          marginLeft: 0, // Always 0 - let flexbox handle the positioning
          backgroundColor: theme.palette.background.default,
          minHeight: `calc(100vh - ${appBarHeight}px)`,
          position: 'relative',
        }}
      >
        {/* Content container with padding */}
        <Box
          sx={{
            paddingTop: theme.spacing(3),
            paddingRight: theme.spacing(3),
            paddingBottom: theme.spacing(3),
            paddingLeft: theme.spacing(2), // 16px breathing room from sidebar
            maxWidth: '100%',
            overflow: 'hidden',
          }}
        >
          {children}
        </Box>

        {/* Notification center */}
        <NotificationCenter />
      </Box>

      {/* Mobile overlay when sidebar is open */}
      {isMobile && sidebarOpen && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: alpha(theme.palette.common.black, 0.5),
            zIndex: theme.zIndex.drawer - 1,
          }}
          onClick={handleToggleSidebar}
        />
      )}
    </Box>
  );
};