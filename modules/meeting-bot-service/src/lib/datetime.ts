import { Logger } from 'winston';

/**
 * Returns a formatted timestamp string suitable for filenames.
 * Format: YYYY-MM-DD_HH-MM-SS
 *
 * @param timezone - Optional IANA timezone (e.g., 'America/New_York')
 * @param logger - Optional logger for debugging
 */
export function getTimeString(timezone?: string, logger?: Logger): string {
  const now = new Date();

  try {
    if (timezone) {
      // Format in the specified timezone
      const options: Intl.DateTimeFormatOptions = {
        timeZone: timezone,
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      };

      const formatter = new Intl.DateTimeFormat('en-CA', options);
      const parts = formatter.formatToParts(now);

      const get = (type: string) => parts.find(p => p.type === type)?.value || '00';

      return `${get('year')}-${get('month')}-${get('day')}_${get('hour')}-${get('minute')}-${get('second')}`;
    }
  } catch (error) {
    logger?.warn('Failed to format time with timezone, falling back to local', { timezone, error });
  }

  // Fallback to local time
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');

  return `${year}-${month}-${day}_${hours}-${minutes}-${seconds}`;
}

/**
 * Returns an ISO timestamp string.
 */
export function getISOTimestamp(): string {
  return new Date().toISOString();
}
