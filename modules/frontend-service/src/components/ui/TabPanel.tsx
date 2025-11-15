/**
 * TabPanel - Reusable Tab Panel Component
 *
 * Provides a consistent tab panel implementation across the application.
 * Supports optional fade animations and custom ID prefixes.
 *
 * Consolidates 10 duplicate implementations across:
 * - Settings, SystemAnalytics, BotManagement components
 * - AudioProcessingHub components
 * - PipelineEditor components
 */

import React from 'react';
import { Box, Fade } from '@mui/material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
  idPrefix?: string;
  disableFade?: boolean;
}

export const TabPanel: React.FC<TabPanelProps> = ({
  children,
  value,
  index,
  idPrefix = 'tabpanel',
  disableFade = false,
  ...other
}) => {
  const content = (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`${idPrefix}-${index}`}
      aria-labelledby={`${idPrefix}-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );

  return disableFade ? content : (
    <Fade in={value === index} timeout={300}>
      {content}
    </Fade>
  );
};

export default TabPanel;
