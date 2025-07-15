import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  Typography,
  Chip,
  Collapse,
  IconButton,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Dashboard,
  AudioFile,
  SmartToy,
  Cable,
  Settings,
  ChevronLeft,
  ChevronRight,
  ExpandLess,
  ExpandMore,
  Mic,
  Equalizer,
  Analytics,
  VideoCall,
  Translate,
  Timeline,
  Videocam,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '@/store';
import { 
  setSidebarOpen, 
  toggleSidebarCollapsed,
  setActiveTab 
} from '@/store/slices/uiSlice';

interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon: React.ReactNode;
  badge?: string | number;
  badgeColor?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  children?: NavigationItem[];
}

export const Sidebar: React.FC = () => {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  const { 
    sidebarOpen, 
    sidebarCollapsed, 
    isMobile 
  } = useAppSelector(state => state.ui);
  
  const { 
    activeBots 
  } = useAppSelector(state => state.bot.systemStats);
  
  const { 
    isConnected 
  } = useAppSelector(state => state.websocket.connection);

  const [expandedSections, setExpandedSections] = React.useState<string[]>(['audio', 'transcription', 'translation', 'analytics']);

  // Navigation structure
  const navigationItems: NavigationItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      path: '/',
      icon: <Dashboard />,
    },
    {
      id: 'audio',
      label: 'Audio Testing',
      path: '/audio-testing',
      icon: <AudioFile />,
    },
    {
      id: 'transcription',
      label: 'Transcription Testing',
      path: '/transcription-testing',
      icon: <Mic />,
    },
    {
      id: 'translation',
      label: 'Translation Testing',
      path: '/translation-testing',
      icon: <Translate />,
    },
    {
      id: 'meeting-test',
      label: 'Meeting Test',
      path: '/meeting-test',
      icon: <Videocam />,
    },
    {
      id: 'bots',
      label: 'Bot Management',
      path: '/bot-management',
      icon: <SmartToy />,
      badge: activeBots > 0 ? activeBots.toString() : undefined,
      badgeColor: 'success',
    },
    {
      id: 'analytics',
      label: 'Analytics Dashboard',
      path: '/analytics',
      icon: <Analytics />,
    },
    {
      id: 'websocket',
      label: 'WebSocket Test',
      path: '/websocket-test',
      icon: <Cable />,
      badge: isConnected ? 'Connected' : 'Disconnected',
      badgeColor: isConnected ? 'success' : 'error',
    },
    {
      id: 'settings',
      label: 'Settings',
      path: '/settings',
      icon: <Settings />,
    },
  ];

  const sidebarWidth = sidebarCollapsed ? 72 : 240;

  const handleToggleCollapse = () => {
    dispatch(toggleSidebarCollapsed());
  };

  const handleNavigate = (path: string, id: string) => {
    navigate(path);
    dispatch(setActiveTab(id));
    
    // Close sidebar on mobile after navigation
    if (isMobile) {
      dispatch(setSidebarOpen(false));
    }
  };

  const handleToggleSection = (sectionId: string) => {
    setExpandedSections(prev => 
      prev.includes(sectionId) 
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  };

  const isActiveRoute = (path: string) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path.split('#')[0]);
  };

  const renderNavigationItem = (item: NavigationItem, depth = 0) => {
    const isActive = isActiveRoute(item.path);
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedSections.includes(item.id);

    return (
      <React.Fragment key={item.id}>
        <ListItem
          disablePadding
          sx={{
            display: 'block',
            paddingLeft: depth > 0 ? theme.spacing(2) : 0,
          }}
        >
          <ListItemButton
            onClick={() => {
              if (hasChildren && !sidebarCollapsed) {
                handleToggleSection(item.id);
              } else {
                handleNavigate(item.path, item.id);
              }
            }}
            selected={isActive}
            sx={{
              minHeight: 48,
              justifyContent: sidebarCollapsed ? 'center' : 'flex-start',
              borderRadius: 1,
              margin: '4px 8px',
              backgroundColor: isActive ? alpha(theme.palette.primary.main, 0.12) : 'transparent',
              '&:hover': {
                backgroundColor: isActive 
                  ? alpha(theme.palette.primary.main, 0.16)
                  : alpha(theme.palette.action.hover, 0.08),
              },
              '&.Mui-selected': {
                backgroundColor: alpha(theme.palette.primary.main, 0.12),
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.16),
                },
              },
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 0,
                mr: sidebarCollapsed ? 0 : 3,
                justifyContent: 'center',
                color: isActive ? theme.palette.primary.main : 'inherit',
              }}
            >
              {item.icon}
            </ListItemIcon>
            
            {!sidebarCollapsed && (
              <>
                <ListItemText 
                  primary={item.label}
                  sx={{
                    opacity: 1,
                    '& .MuiListItemText-primary': {
                      color: isActive ? theme.palette.primary.main : 'inherit',
                      fontWeight: isActive ? 600 : 400,
                    },
                  }}
                />
                
                {item.badge && (
                  <Chip
                    label={item.badge}
                    size="small"
                    color={item.badgeColor || 'default'}
                    sx={{ 
                      height: 20, 
                      fontSize: '0.75rem',
                      marginRight: hasChildren ? 1 : 0,
                    }}
                  />
                )}
                
                {hasChildren && (
                  <IconButton
                    size="small"
                    sx={{ padding: 0.5 }}
                  >
                    {isExpanded ? <ExpandLess /> : <ExpandMore />}
                  </IconButton>
                )}
              </>
            )}
          </ListItemButton>
        </ListItem>

        {/* Render children */}
        {hasChildren && !sidebarCollapsed && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children!.map(child => 
                renderNavigationItem(child, depth + 1)
              )}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  };

  return (
    <Drawer
      variant={isMobile ? 'temporary' : 'persistent'}
      anchor="left"
      open={sidebarOpen}
      onClose={() => dispatch(setSidebarOpen(false))}
      sx={{
        width: sidebarWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: sidebarWidth,
          boxSizing: 'border-box',
          backgroundColor: theme.palette.background.default,
          borderRight: `1px solid ${theme.palette.divider}`,
          transition: theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
          overflowX: 'hidden',
        },
      }}
    >
      {/* Sidebar header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: sidebarCollapsed ? 'center' : 'space-between',
          padding: theme.spacing(1, 2),
          minHeight: 64, // Match AppBar height
          borderBottom: `1px solid ${theme.palette.divider}`,
        }}
      >
        {!sidebarCollapsed && (
          <Typography variant="h6" sx={{ fontWeight: 600, color: theme.palette.primary.main }}>
            LiveTranslate
          </Typography>
        )}
        
        {!isMobile && (
          <IconButton
            onClick={handleToggleCollapse}
            size="small"
            sx={{
              border: `1px solid ${theme.palette.divider}`,
              backgroundColor: theme.palette.background.paper,
            }}
          >
            {sidebarCollapsed ? <ChevronRight /> : <ChevronLeft />}
          </IconButton>
        )}
      </Box>

      {/* Navigation list */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', paddingY: 1 }}>
        <List>
          {navigationItems.map(item => renderNavigationItem(item))}
        </List>
      </Box>

      {/* Sidebar footer */}
      {!sidebarCollapsed && (
        <Box
          sx={{
            padding: theme.spacing(2),
            borderTop: `1px solid ${theme.palette.divider}`,
            backgroundColor: alpha(theme.palette.primary.main, 0.04),
          }}
        >
          <Typography variant="caption" color="text.secondary" display="block">
            Version 1.0.0
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {isConnected ? 'Connected' : 'Disconnected'}
          </Typography>
        </Box>
      )}
    </Drawer>
  );
};