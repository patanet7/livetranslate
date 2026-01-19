/**
 * Notifications Hook
 *
 * Simplified interface for dispatching notifications.
 * Eliminates duplicate addNotification dispatch patterns.
 */

import { useCallback } from "react";
import { useAppDispatch } from "@/store";
import { addNotification } from "@/store/slices/uiSlice";

/**
 * Hook for managing notifications
 *
 * Used for: User feedback, operation status, error messages
 * Replaces: 31 duplicate dispatch(addNotification({...})) calls
 *
 * @example
 * const { notifySuccess, notifyError, notifyInfo } = useNotifications();
 *
 * // Simple usage
 * notifySuccess('Settings saved successfully!');
 *
 * // With custom title
 * notifyError('Operation Failed', 'Failed to connect to server');
 *
 * // With custom auto-hide
 * notifyInfo('Processing', 'Your request is being processed...', false);
 */
export const useNotifications = () => {
  const dispatch = useAppDispatch();

  /**
   * Display success notification
   *
   * @param title - Notification title
   * @param message - Notification message (optional)
   * @param autoHide - Auto-hide after delay (default: true)
   */
  const notifySuccess = useCallback(
    (title: string, message?: string, autoHide = true) => {
      dispatch(
        addNotification({
          type: "success",
          title,
          message: message || title,
          autoHide,
        }),
      );
    },
    [dispatch],
  );

  /**
   * Display error notification
   *
   * @param title - Notification title
   * @param message - Notification message (optional)
   * @param autoHide - Auto-hide after delay (default: false)
   */
  const notifyError = useCallback(
    (title: string, message?: string, autoHide = false) => {
      dispatch(
        addNotification({
          type: "error",
          title,
          message: message || title,
          autoHide,
        }),
      );
    },
    [dispatch],
  );

  /**
   * Display warning notification
   *
   * @param title - Notification title
   * @param message - Notification message (optional)
   * @param autoHide - Auto-hide after delay (default: true)
   */
  const notifyWarning = useCallback(
    (title: string, message?: string, autoHide = true) => {
      dispatch(
        addNotification({
          type: "warning",
          title,
          message: message || title,
          autoHide,
        }),
      );
    },
    [dispatch],
  );

  /**
   * Display info notification
   *
   * @param title - Notification title
   * @param message - Notification message (optional)
   * @param autoHide - Auto-hide after delay (default: true)
   */
  const notifyInfo = useCallback(
    (title: string, message?: string, autoHide = true) => {
      dispatch(
        addNotification({
          type: "info",
          title,
          message: message || title,
          autoHide,
        }),
      );
    },
    [dispatch],
  );

  /**
   * Display notification from API error
   *
   * @param error - Error object
   * @param title - Custom title (default: "Error")
   * @param autoHide - Auto-hide after delay (default: false)
   */
  const notifyFromError = useCallback(
    (error: any, title = "Error", autoHide = false) => {
      const message =
        error?.message ||
        error?.detail ||
        error?.error ||
        (typeof error === "string" ? error : "An unexpected error occurred");

      dispatch(
        addNotification({
          type: "error",
          title,
          message,
          autoHide,
        }),
      );
    },
    [dispatch],
  );

  /**
   * Display notification for API operation result
   *
   * @param success - Whether operation succeeded
   * @param successMessage - Success message
   * @param errorMessage - Error message
   * @param error - Error object (if failed)
   */
  const notifyApiResult = useCallback(
    (
      success: boolean,
      successMessage: string,
      errorMessage: string,
      error?: any,
    ) => {
      if (success) {
        notifySuccess(successMessage);
      } else {
        const message = error
          ? error?.message || error?.detail || errorMessage
          : errorMessage;
        notifyError("Operation Failed", message);
      }
    },
    [notifySuccess, notifyError],
  );

  return {
    notifySuccess,
    notifyError,
    notifyWarning,
    notifyInfo,
    notifyFromError,
    notifyApiResult,
  };
};

export default useNotifications;
