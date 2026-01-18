import React from "react";
import {
  Box,
  Chip,
  Tooltip,
  IconButton,
  Popover,
  Typography,
  Paper,
  Stack,
  Divider,
} from "@mui/material";
import { Wifi, WifiOff, Warning, Circle, Refresh } from "@mui/icons-material";
import { useAppSelector } from "@/store";
import { useWebSocket } from "@/hooks/useWebSocket";

interface ConnectionIndicatorProps {
  isConnected: boolean;
  reconnectAttempts: number;
  size?: "small" | "medium";
  showLabel?: boolean;
}

export const ConnectionIndicator: React.FC<ConnectionIndicatorProps> = ({
  isConnected,
  reconnectAttempts,
  size = "small",
  showLabel = false,
}) => {
  const [anchorEl, setAnchorEl] = React.useState<HTMLElement | null>(null);
  const { connect, disconnect } = useWebSocket();

  const { connection, stats, config } = useAppSelector(
    (state) => state.websocket,
  );

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleReconnect = () => {
    disconnect();
    setTimeout(() => connect(), 100);
    handleClose();
  };

  const open = Boolean(anchorEl);

  // Determine status and appearance
  const getStatus = () => {
    if (isConnected) {
      return {
        status: "connected",
        color: "success" as const,
        icon: <Wifi />,
        label: "Connected",
        tooltip: "WebSocket connection is active",
      };
    } else if (reconnectAttempts > 0) {
      return {
        status: "reconnecting",
        color: "warning" as const,
        icon: <Warning />,
        label: "Reconnecting",
        tooltip: `Reconnecting... (${reconnectAttempts}/${config.maxReconnectAttempts})`,
      };
    } else {
      return {
        status: "disconnected",
        color: "error" as const,
        icon: <WifiOff />,
        label: "Disconnected",
        tooltip: "WebSocket connection is lost",
      };
    }
  };

  const statusInfo = getStatus();

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  const formatBytes = (bytes: number) => {
    const units = ["B", "KB", "MB", "GB"];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  if (showLabel) {
    return (
      <>
        <Chip
          icon={statusInfo.icon}
          label={statusInfo.label}
          color={statusInfo.color}
          size={size}
          onClick={handleClick}
          sx={{
            cursor: "pointer",
            "& .MuiChip-icon": {
              fontSize: size === "small" ? "1rem" : "1.25rem",
            },
          }}
        />

        <Popover
          open={open}
          anchorEl={anchorEl}
          onClose={handleClose}
          anchorOrigin={{
            vertical: "bottom",
            horizontal: "right",
          }}
          transformOrigin={{
            vertical: "top",
            horizontal: "right",
          }}
        >
          <Paper sx={{ padding: 2, minWidth: 280 }}>
            <Stack spacing={2}>
              {/* Connection status */}
              <Box>
                <Typography variant="h6" gutterBottom>
                  WebSocket Connection
                </Typography>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Circle
                    fontSize="small"
                    color={statusInfo.color}
                    sx={{ fontSize: "0.75rem" }}
                  />
                  <Typography variant="body2">{statusInfo.label}</Typography>
                  {reconnectAttempts > 0 && (
                    <Typography variant="caption" color="text.secondary">
                      ({reconnectAttempts}/{config.maxReconnectAttempts})
                    </Typography>
                  )}
                </Box>
              </Box>

              <Divider />

              {/* Connection details */}
              <Stack spacing={1}>
                <Typography variant="subtitle2">Connection Details</Typography>

                <Box
                  sx={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 1,
                  }}
                >
                  <Typography variant="caption" color="text.secondary">
                    Status:
                  </Typography>
                  <Typography variant="caption">{statusInfo.label}</Typography>

                  <Typography variant="caption" color="text.secondary">
                    URL:
                  </Typography>
                  <Typography variant="caption" sx={{ wordBreak: "break-all" }}>
                    {config.url}
                  </Typography>

                  {connection.connectionId && (
                    <>
                      <Typography variant="caption" color="text.secondary">
                        ID:
                      </Typography>
                      <Typography
                        variant="caption"
                        sx={{ fontFamily: "monospace" }}
                      >
                        {connection.connectionId.slice(-8)}
                      </Typography>
                    </>
                  )}
                </Box>
              </Stack>

              {/* Statistics */}
              {isConnected && (
                <>
                  <Divider />
                  <Stack spacing={1}>
                    <Typography variant="subtitle2">Statistics</Typography>

                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr",
                        gap: 1,
                      }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        Uptime:
                      </Typography>
                      <Typography variant="caption">
                        {stats.connectionDuration > 0
                          ? formatDuration(
                              Date.now() - stats.connectionDuration,
                            )
                          : "0s"}
                      </Typography>

                      <Typography variant="caption" color="text.secondary">
                        Messages Sent:
                      </Typography>
                      <Typography variant="caption">
                        {stats.messagesSent.toLocaleString()}
                      </Typography>

                      <Typography variant="caption" color="text.secondary">
                        Messages Received:
                      </Typography>
                      <Typography variant="caption">
                        {stats.messagesReceived.toLocaleString()}
                      </Typography>

                      <Typography variant="caption" color="text.secondary">
                        Data Transferred:
                      </Typography>
                      <Typography variant="caption">
                        {formatBytes(stats.bytesTransferred)}
                      </Typography>

                      {stats.averageLatency > 0 && (
                        <>
                          <Typography variant="caption" color="text.secondary">
                            Avg Latency:
                          </Typography>
                          <Typography variant="caption">
                            {Math.round(stats.averageLatency)}ms
                          </Typography>
                        </>
                      )}
                    </Box>
                  </Stack>
                </>
              )}

              {/* Actions */}
              <Divider />
              <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
                <IconButton
                  size="small"
                  onClick={handleReconnect}
                  disabled={isConnected}
                  title="Reconnect"
                >
                  <Refresh />
                </IconButton>
              </Box>
            </Stack>
          </Paper>
        </Popover>
      </>
    );
  }

  return (
    <Tooltip title={statusInfo.tooltip}>
      <IconButton
        size={size}
        onClick={handleClick}
        sx={{
          color:
            statusInfo.color === "success"
              ? "success.main"
              : statusInfo.color === "warning"
                ? "warning.main"
                : "error.main",
        }}
      >
        {statusInfo.icon}
      </IconButton>
    </Tooltip>
  );
};
