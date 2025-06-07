#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h> // Include this first to prevent conflicts
#include <windows.h>
#include <objbase.h>  // Add this for COM support
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <memory>
#include <functional>
#include <string>
#include "gui.hpp"
#include "../include/audio/audio_capture.hpp"
#include "../include/audio/nvidia_effects.hpp"

// Global callback declarations
extern std::function<void(const float*, size_t)> g_audioLevelCallback;
extern std::function<void(const float*, const float*, size_t)> g_spectrumCallback;

void setupAudioCallbacks(AudioCapture& capture, GUI& gui) {
    std::cout << "Setting up audio callbacks..." << std::endl;
    
    // Set up global audio level callback
    g_audioLevelCallback = [&gui](const float* buffer, size_t numSamples) {
        gui.updateAudioLevel(buffer, numSamples);
    };
    
    // Set up global spectrum callback
    g_spectrumCallback = [&gui](const float* inputBuffer, const float* outputBuffer, size_t numSamples) {
        gui.updateSpectrum(inputBuffer, outputBuffer, numSamples);
    };
    
    std::cout << "Audio callbacks setup completed" << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize COM for the main thread
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM: " << hr << std::endl;
        return -1;
    }
    
    // Create GUI and audio capture instances
    GUI gui;
    auto audioCapture = std::make_shared<AudioCapture>();
    
    // Print availability of NVIDIA Audio Effects
    std::cout << "NVIDIA Audio Effects SDK availability: " 
              << (NvidiaAudioEffects::isAvailable() ? "Available" : "Not available") << std::endl;
    
    if (!gui.initialize(1024, 768, "LiveTranslate - ImGui Version")) {
        std::cerr << "Failed to initialize GUI" << std::endl;
        CoUninitialize();
        return -1;
    }
    
    // Setup audio callbacks
    setupAudioCallbacks(*audioCapture, gui);
    
    // Set up gain change callback - this will be applied before NVIDIA effects
    gui.setGainChangeCallback([audioCapture](float gain) {
        std::cout << "Setting input gain to: " << gain << std::endl;
        audioCapture->setInputGain(gain);
    });
    
    // Set callbacks for start/stop capture
    gui.setCallbacks(
        // Start capture callback
        [&](int inputDeviceIndex, bool enablePassthrough) -> bool {
            // Configure audio capture with GUI settings
            audioCapture->setSampleRate(gui.getSampleRate());
            
            // Configure NVIDIA effects based on UI selections
            NvidiaEffectType effectType = NvidiaEffectType::Denoiser; // Default
            
            switch (gui.getSelectedEffect()) {
                case 0: // Noise Removal
                    effectType = NvidiaEffectType::Denoiser;
                    std::cout << "Enabling NVIDIA Denoiser effect" << std::endl;
                    break;
                case 1: // Room Echo Removal
                    effectType = NvidiaEffectType::Dereverb;
                    std::cout << "Enabling NVIDIA Dereverb effect" << std::endl;
                    break;
                case 2: // Noise + Echo Removal
                    effectType = NvidiaEffectType::DereverbDenoiser;
                    std::cout << "Enabling NVIDIA Dereverb+Denoiser effect" << std::endl;
                    break;
                case 3: // Audio Super-Resolution
                    effectType = NvidiaEffectType::SuperRes;
                    std::cout << "Enabling NVIDIA Super-Resolution effect" << std::endl;
                    break;
                case 4: // Acoustic Echo Cancellation
                    effectType = NvidiaEffectType::AEC;
                    std::cout << "Enabling NVIDIA AEC effect" << std::endl;
                    break;
            }
            
            // Set audio processing parameters
            audioCapture->setNvidiaEffectType(effectType);
            
            if (gui.getEnableNoiseEffects()) {
                std::cout << "Configuring NVIDIA noise reduction:" << std::endl;
                std::cout << "  - Noise reduction enabled: " << gui.getEnableNoiseReduction() << std::endl;
                std::cout << "  - Noise reduction level: " << gui.getNoiseReductionLevel() << std::endl;
                audioCapture->setNoiseEffectsEnabled(gui.getEnableNoiseEffects());
                audioCapture->setNoiseReductionEnabled(gui.getEnableNoiseReduction());
                audioCapture->setNoiseReductionLevel(gui.getNoiseReductionLevel());
            }
            
            if (gui.getEnableEchoEffects()) {
                std::cout << "Configuring NVIDIA echo cancellation:" << std::endl;
                std::cout << "  - Room echo removal enabled: " << gui.getEnableRoomEchoRemoval() << std::endl;
                std::cout << "  - Room echo removal level: " << gui.getRoomEchoRemovalLevel() << std::endl;
                audioCapture->setEchoEffectsEnabled(gui.getEnableEchoEffects());
                audioCapture->setRoomEchoRemovalEnabled(gui.getEnableRoomEchoRemoval());
                audioCapture->setRoomEchoRemovalLevel(gui.getRoomEchoRemovalLevel());
            }
            
            if (gui.getEnableResolutionEffects()) {
                std::cout << "Configuring NVIDIA super resolution:" << std::endl;
                std::cout << "  - Super resolution enabled: " << gui.getEnableSuperResolution() << std::endl;
                std::cout << "  - Output sample rate: " << gui.getSuperResOutSampleRate() << " Hz" << std::endl;
                audioCapture->setResolutionEffectsEnabled(gui.getEnableResolutionEffects());
                audioCapture->setSuperResolutionEnabled(gui.getEnableSuperResolution());
                audioCapture->setSuperResOutSampleRate(gui.getSuperResOutSampleRate());
            }
            
            if (gui.getEnableVAD()) {
                std::cout << "Enabling NVIDIA Voice Activity Detection" << std::endl;
            }
            audioCapture->setVADEnabled(gui.getEnableVAD());
            
            // Start the capture with all configured settings
            return audioCapture->startCapture(inputDeviceIndex, 0, enablePassthrough);
        },
        // Stop capture callback
        [&]() {
            std::cout << "Stopping audio capture" << std::endl;
            audioCapture->stopCapture();
        }
    );
    
    // Setup audio recording callbacks
    gui.setSaveAudioCallbacks(
        // Save input audio callback
        [audioCapture](const std::string& filename) -> bool {
            // Start recording
            audioCapture->startRecording();
            return true;
        },
        
        // Save output audio callback
        [audioCapture](const std::string& filename) -> bool {
            // Stop recording and save
            audioCapture->stopRecording();
            return true;
        }
    );
    
    // Setup recording state callback
    gui.setRecordingStateCallback(
        [audioCapture](bool isRecording) {
            if (isRecording) {
                audioCapture->startRecording();
            } else {
                audioCapture->stopRecording();
            }
        }
    );
    
    // Setup auto-save callbacks
    gui.setAutoSaveCallbacks(
        // Auto-save state callback
        [audioCapture](bool enabled) {
            audioCapture->setAutoSaveEnabled(enabled);
        },
        // Auto-save directory callback
        [audioCapture](const std::string& directory) {
            audioCapture->setAutoSaveDirectory(directory);
        }
    );
    
    // Run the GUI main loop
    gui.run();
    
    // Cleanup
    gui.shutdown();
    CoUninitialize();
    
    return 0;
}

void GUI::renderAudioControls() {
    if (ImGui::CollapsingHeader("Audio Controls")) {
        renderAudioVisualization();
        ImGui::Separator();
        renderRecordingControls();
        ImGui::Separator();
        renderABTestingControls();
    }
}

void GUI::renderAudioVisualization() {
    // Implementation of renderAudioVisualization
}

void GUI::renderRecordingControls() {
    // Implementation of renderRecordingControls
}

void GUI::renderABTestingControls() {
    // Implementation of renderABTestingControls
} 