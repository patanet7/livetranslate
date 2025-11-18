/**
 * Stat Card Component
 *
 * Reusable statistics display card component.
 * Eliminates 22+ duplicate Grid+Paper+Typography patterns.
 */

import React from 'react';
import { Grid, Paper, Typography, GridProps } from '@mui/material';

export interface StatCardProps {
  /**
   * Value to display (number, string, or React node)
   */
  value: string | number | React.ReactNode;

  /**
   * Label text below the value
   */
  label: string;

  /**
   * Color theme for the value
   * @default 'primary'
   */
  color?: 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info' | 'default';

  /**
   * Grid item props for responsive layout
   * @default { xs: 12, md: 6, lg: 2.4 }
   */
  gridProps?: GridProps;

  /**
   * Optional icon to display above the value
   */
  icon?: React.ReactNode;

  /**
   * Optional subtitle below the label
   */
  subtitle?: string;

  /**
   * Paper elevation
   * @default 1
   */
  elevation?: number;

  /**
   * Custom value formatter
   */
  formatter?: (value: string | number) => string;
}

/**
 * Statistics Card Component
 *
 * Used for: Dashboard metrics, service statistics, analytics displays
 * Replaces: ~22 duplicate Grid+Paper+Typography blocks in Settings components
 *
 * @example
 * // Basic usage
 * <StatCard value={1234} label="Total Requests" />
 *
 * // With color
 * <StatCard value={42} label="Active Bots" color="success" />
 *
 * // With custom grid layout
 * <StatCard
 *   value="99.9%"
 *   label="Uptime"
 *   color="primary"
 *   gridProps={{ xs: 12, md: 4 }}
 * />
 *
 * // With icon
 * <StatCard
 *   value={150}
 *   label="Errors"
 *   color="error"
 *   icon={<ErrorIcon />}
 * />
 *
 * // With formatter
 * <StatCard
 *   value={1234567}
 *   label="Total Size"
 *   formatter={(v) => `${(Number(v) / 1024 / 1024).toFixed(2)} MB`}
 * />
 */
export const StatCard: React.FC<StatCardProps> = ({
  value,
  label,
  color = 'primary',
  gridProps = { xs: 12, md: 6, lg: 2.4 },
  icon,
  subtitle,
  elevation = 1,
  formatter,
}) => {
  // Format value if formatter provided
  const displayValue =
    formatter && (typeof value === 'string' || typeof value === 'number')
      ? formatter(value)
      : value;

  return (
    <Grid item {...gridProps}>
      <Paper
        elevation={elevation}
        sx={{
          p: 2,
          textAlign: 'center',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        }}
      >
        {icon && (
          <Typography
            variant="h4"
            sx={{
              mb: 1,
              color: `${color}.main`,
              display: 'flex',
              justifyContent: 'center',
            }}
          >
            {icon}
          </Typography>
        )}

        <Typography variant="h6" color={`${color}.main`} sx={{ fontWeight: 600 }}>
          {displayValue}
        </Typography>

        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>

        {subtitle && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
            {subtitle}
          </Typography>
        )}
      </Paper>
    </Grid>
  );
};

/**
 * Stat Card Group Component
 *
 * Container for multiple StatCards with consistent spacing
 *
 * @example
 * <StatCardGroup>
 *   <StatCard value={100} label="Total" />
 *   <StatCard value={95} label="Success" color="success" />
 *   <StatCard value={5} label="Errors" color="error" />
 * </StatCardGroup>
 */
export const StatCardGroup: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <Grid container spacing={3}>
      {children}
    </Grid>
  );
};

export default StatCard;
