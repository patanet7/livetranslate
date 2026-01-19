/**
 * useAudioDevices Hook
 *
 * Shared hook for audio device enumeration and management.
 * Consolidates duplicate implementations from:
 * - StreamingProcessor/index.tsx
 * - MeetingTest/index.tsx
 * - AudioTesting/index.tsx
 * - TranscriptionTesting/index.tsx
 */

import { useEffect, useCallback } from "react";
import { useAppDispatch } from "@/store";
import { setAudioDevices, addProcessingLog } from "@/store/slices/audioSlice";

interface AudioDeviceInfo {
  deviceId: string;
  label: string;
  kind: "audioinput";
  groupId: string;
}

export interface UseAudioDevicesOptions {
  /**
   * Auto-select first device if no device is currently selected
   */
  autoSelect?: boolean;

  /**
   * Currently selected device ID (for auto-selection logic)
   */
  selectedDevice?: string;

  /**
   * Callback when devices are loaded
   */
  onDevicesLoaded?: (devices: AudioDeviceInfo[]) => void;

  /**
   * Callback to set selected device (for auto-selection)
   */
  onDeviceSelected?: (deviceId: string) => void;
}

/**
 * Hook to enumerate and manage audio input devices
 */
export const useAudioDevices = (options: UseAudioDevicesOptions = {}) => {
  const dispatch = useAppDispatch();
  const {
    autoSelect = false,
    selectedDevice,
    onDevicesLoaded,
    onDeviceSelected,
  } = options;

  const initializeDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioDevices = devices
        .filter((device) => device.kind === "audioinput")
        .map((device) => ({
          deviceId: device.deviceId,
          label:
            device.label || `Microphone ${device.deviceId.substring(0, 8)}`,
          kind: device.kind as "audioinput",
          groupId: device.groupId || "",
        }));

      dispatch(setAudioDevices(audioDevices));

      // Auto-select first device if requested and no device selected
      if (
        autoSelect &&
        audioDevices.length > 0 &&
        !selectedDevice &&
        onDeviceSelected
      ) {
        onDeviceSelected(audioDevices[0].deviceId);
      }

      dispatch(
        addProcessingLog({
          level: "INFO",
          message: `Found ${audioDevices.length} audio input devices`,
          timestamp: Date.now(),
        }),
      );

      // Call optional callback
      if (onDevicesLoaded) {
        onDevicesLoaded(audioDevices);
      }
    } catch (error) {
      dispatch(
        addProcessingLog({
          level: "ERROR",
          message: `Failed to load audio devices: ${error}`,
          timestamp: Date.now(),
        }),
      );
    }
  }, [dispatch, autoSelect, selectedDevice, onDevicesLoaded, onDeviceSelected]);

  // Auto-initialize on mount
  useEffect(() => {
    initializeDevices();
  }, [initializeDevices]);

  return {
    /**
     * Manually refresh device list
     */
    refreshDevices: initializeDevices,
  };
};

export default useAudioDevices;
