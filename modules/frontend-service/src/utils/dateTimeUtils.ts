/**
 * Date and Time Utility Functions
 *
 * Centralized utilities for date/time formatting and manipulation.
 * Eliminates duplicate formatting logic across the codebase.
 */

/**
 * Get current timestamp as ISO string
 *
 * Used for: Bot timestamps, activity logs, session tracking
 * Replaces: new Date().toISOString() (13+ occurrences)
 */
export const getCurrentISOTimestamp = (): string => {
  return new Date().toISOString();
};

/**
 * Format uptime in seconds to human-readable format
 *
 * @param seconds - Uptime in seconds
 * @returns Formatted string like "2d 5h 30m"
 *
 * Used for: Service uptime, session duration, bot runtime
 * Replaces: Duplicate formatUptime functions in Settings components
 */
export const formatUptime = (seconds: number): string => {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  const parts: string[] = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);

  return parts.join(" ");
};

/**
 * Format duration in milliseconds to human-readable format
 *
 * @param ms - Duration in milliseconds
 * @returns Formatted string like "2d 5h 30m"
 *
 * Used for: Processing times, latency, session duration
 */
export const formatDuration = (ms: number): string => {
  return formatUptime(ms / 1000);
};

/**
 * Format duration in milliseconds to short format
 *
 * @param ms - Duration in milliseconds
 * @returns Formatted string like "1.5s" or "250ms"
 *
 * Used for: Performance metrics, response times
 */
export const formatDurationShort = (ms: number): string => {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  } else if (ms < 3600000) {
    return `${(ms / 60000).toFixed(1)}m`;
  } else {
    return `${(ms / 3600000).toFixed(1)}h`;
  }
};

/**
 * Format relative time from ISO string or Date
 *
 * @param date - ISO string or Date object
 * @returns Relative time string like "2 minutes ago", "just now"
 *
 * Used for: Activity logs, last seen, recent events
 */
export const formatRelativeTime = (date: string | Date): string => {
  const now = new Date();
  const then = typeof date === "string" ? new Date(date) : date;
  const diffMs = now.getTime() - then.getTime();

  if (isNaN(then.getTime())) {
    return "Invalid date";
  }

  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSeconds < 30) return "just now";
  if (diffSeconds < 60) return `${diffSeconds} seconds ago`;
  if (diffMinutes === 1) return "1 minute ago";
  if (diffMinutes < 60) return `${diffMinutes} minutes ago`;
  if (diffHours === 1) return "1 hour ago";
  if (diffHours < 24) return `${diffHours} hours ago`;
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
  return `${Math.floor(diffDays / 365)} years ago`;
};

/**
 * Format ISO timestamp to locale string
 *
 * @param isoString - ISO timestamp string
 * @param options - Intl.DateTimeFormat options
 * @returns Formatted date string
 *
 * Used for: Display timestamps, logs, reports
 */
export const formatTimestamp = (
  isoString: string,
  options?: Intl.DateTimeFormatOptions,
): string => {
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) {
      return "Invalid date";
    }
    return date.toLocaleString(
      undefined,
      options || {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      },
    );
  } catch {
    return "Invalid date";
  }
};

/**
 * Format ISO timestamp to date only
 *
 * @param isoString - ISO timestamp string
 * @returns Formatted date string like "Jan 15, 2025"
 */
export const formatDate = (isoString: string): string => {
  return formatTimestamp(isoString, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
};

/**
 * Format ISO timestamp to time only
 *
 * @param isoString - ISO timestamp string
 * @returns Formatted time string like "14:30:45"
 */
export const formatTime = (isoString: string): string => {
  return formatTimestamp(isoString, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

/**
 * Check if a timestamp is recent (within specified milliseconds)
 *
 * @param isoString - ISO timestamp string
 * @param thresholdMs - Threshold in milliseconds (default: 5 minutes)
 * @returns true if timestamp is within threshold
 *
 * Used for: Activity detection, health checks, freshness validation
 */
export const isRecent = (
  isoString: string,
  thresholdMs: number = 300000,
): boolean => {
  try {
    const date = new Date(isoString);
    const now = new Date();
    return now.getTime() - date.getTime() < thresholdMs;
  } catch {
    return false;
  }
};

/**
 * Parse duration string to milliseconds
 *
 * @param duration - Duration string like "2d 5h 30m"
 * @returns Duration in milliseconds
 *
 * Used for: Configuration parsing, duration inputs
 */
export const parseDuration = (duration: string): number => {
  const regex = /(\d+)([dhms])/g;
  let totalMs = 0;
  let match;

  while ((match = regex.exec(duration)) !== null) {
    const value = parseInt(match[1], 10);
    const unit = match[2];

    switch (unit) {
      case "d":
        totalMs += value * 86400000;
        break;
      case "h":
        totalMs += value * 3600000;
        break;
      case "m":
        totalMs += value * 60000;
        break;
      case "s":
        totalMs += value * 1000;
        break;
    }
  }

  return totalMs;
};
