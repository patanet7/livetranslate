#define WIN32_LEAN_AND_MEAN
#include <winsock2.h> // Include this first to prevent conflicts
#include <windows.h>

#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include "../include/utils/compat_fix.hpp"
#include "../include/audio/audio_capture.hpp"
#include "../include/audio/nvidia_effects.hpp"

int main() {
    // Initialize COM for audio device enumeration
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM: " << hr << std::endl;
        return 1;
    }

    std::cout << "LiveTranslate Audio Capture" << std::endl;
    std::cout << "--------------------------" << std::endl;
    
    // Check NVIDIA Audio Effects SDK availability
    std::cout << "NVIDIA Audio Effects SDK availability: " 
              << (NvidiaAudioEffects::isAvailable() ? "Available" : "Not available") << std::endl;
    std::cout << std::endl;
    
    // Initialize audio capture
    AudioCapture audioCapture(48000, 1);
    
    // List available input devices
    std::cout << "Available input devices:" << std::endl;
    auto inputDevices = audioCapture.getInputDevices();
    for (size_t i = 0; i < inputDevices.size(); ++i) {
        std::cout << i << ": " << inputDevices[i].name << " (ID: " << wstring_to_string(inputDevices[i].id) << ")" << std::endl;
    }
    
    // List available output devices
    std::cout << "\nAvailable output devices:" << std::endl;
    auto outputDevices = audioCapture.getOutputDevices();
    for (size_t i = 0; i < outputDevices.size(); ++i) {
        std::cout << i << ": " << outputDevices[i].name << " (ID: " << wstring_to_string(outputDevices[i].id) << ")" << std::endl;
    }
    
    // Ask user to select devices
    int inputDeviceIndex = 0;
    int outputDeviceIndex = 0;
    bool enablePassthrough = false;
    
    std::cout << "\nSelect input device (0-" << inputDevices.size() - 1 << "): ";
    std::cin >> inputDeviceIndex;
    
    std::cout << "Select output device (0-" << outputDevices.size() - 1 << "): ";
    std::cin >> outputDeviceIndex;
    
    std::cout << "Enable audio passthrough? (0/1): ";
    std::cin >> enablePassthrough;
    
    // Start capture with selected devices
    std::cout << "\nAttempting to start audio capture with:" << std::endl;
    std::cout << "- Input device: " << inputDeviceIndex << ": " << (inputDevices.size() > inputDeviceIndex ? inputDevices[inputDeviceIndex].name : "Invalid device") << std::endl;
    std::cout << "- Output device: " << outputDeviceIndex << ": " << (outputDevices.size() > outputDeviceIndex ? outputDevices[outputDeviceIndex].name : "Invalid device") << std::endl;
    std::cout << "- Passthrough: " << (enablePassthrough ? "Enabled" : "Disabled") << std::endl;
    
    if (audioCapture.startCapture(inputDeviceIndex, outputDeviceIndex, enablePassthrough)) {
        std::cout << "\nCapture started successfully. Press Enter to stop..." << std::endl;
        std::cin.ignore(); // Clear the input buffer
        std::cin.get();    // Wait for Enter
        
        // Stop capture
        audioCapture.stopCapture();
        std::cout << "Capture stopped." << std::endl;
    } else {
        std::cout << "\nFailed to start capture." << std::endl;
    }
    
    // Uninitialize COM
    CoUninitialize();
    
    return 0;
} 