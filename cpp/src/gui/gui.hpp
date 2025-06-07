#pragma once
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define NOMINMAX

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <deque>
#include "imgui.h"
#include <commdlg.h>
#include <shlobj.h>

// Include muFFT
#include "../../external/muFFT/fft.h"

// Constants for spectrum visualization
#define SPECTRUM_WIDTH 400
#define SPECTRUM_HEIGHT 200
#define FFT_SIZE 4096
#define NUM_MEL_BINS 128
#define MIN_FREQ 10.0f
#define MAX_FREQ 22000.0f

struct GLFWwindow;

// Forward declarations
class AudioCapture;
class AudioProcessor;

class Spectrogram {
public:
    Spectrogram(int width, int height, int fftSize = 4096, int sampleRate = 48000);
    ~Spectrogram();
    void update(const float* audioData, size_t numSamples);
    void draw(ImDrawList* draw_list, ImVec2 pos, ImVec2 size);

private:
    int width_;
    int height_;
    int fftSize_;
    int sampleRate_;  // Sample rate for frequency calculations
    std::deque<std::vector<float>> buffer_;
    mufft_plan_1d* fftContext_;  // muFFT context
    std::vector<float> fftIn_;
    std::vector<float> fftOut_;
    std::vector<float> window_;
    std::mutex bufferMutex_;  // Mutex for thread-safe buffer access
};

class GUI {
public:
    GUI();
    ~GUI();
    
    bool initialize(int width = 800, int height = 600, const char* title = "LiveTranslate");
    void run();
    void shutdown();
    
    // Callbacks
    using StartCaptureCallback = std::function<bool(int inputDeviceIndex, bool enablePassthrough)>;
    using StopCaptureCallback = std::function<void()>;
    using SaveAudioCallback = std::function<bool(const std::string& filename)>;
    using RecordingStateCallback = std::function<void(bool enabled)>;
    using AutoSaveStateCallback = std::function<void(bool enabled)>;
    using AutoSaveDirCallback = std::function<void(const std::string& dir)>;
    using GainChangeCallback = std::function<void(float)>;
    void performFFT(const float* buffer, size_t bufferSize, std::vector<float>& magnitudes);

    void setCallbacks(StartCaptureCallback startCb, StopCaptureCallback stopCb);
    void setSaveAudioCallbacks(SaveAudioCallback saveInputCb, SaveAudioCallback saveOutputCb);
    void setRecordingStateCallback(RecordingStateCallback recordingCb);
    void setAutoSaveCallbacks(AutoSaveStateCallback autoSaveCb, AutoSaveDirCallback dirCb);
    void setGainChangeCallback(GainChangeCallback callback);
    void setDeviceChangeCallback(std::function<void(const std::string&)> callback) { deviceChangeCallback_ = callback; }
    
    // Audio level and spectrum visualization
    void updateAudioLevel(const float* buffer, size_t numSamples);
    void updateSpectrum(const float* inputBuffer, const float* outputBuffer, size_t numSamples);
    
    // Getters for audio settings
    int getSampleRate() const { return sampleRate; }
    int getBufferSize() const { return bufferSize; }
    bool getEnablePassthrough() const { return enablePassthrough; }
    
    // Getters for NVIDIA Audio Effects settings
    int getSelectedEffect() const { return selectedEffect; }
    bool getEnableNoiseReduction() const { return enableNoiseReduction; }
    float getNoiseReductionLevel() const { return noiseReductionLevel; }
    bool getEnableRoomEchoRemoval() const { return enableRoomEchoRemoval; }
    float getRoomEchoRemovalLevel() const { return roomEchoRemovalLevel; }
    bool getEnableSuperResolution() const { return enableSuperResolution; }
    int getSuperResOutSampleRate() const { return superResOutSampleRate; }
    bool getEnableAEC() const { return enableAEC; }
    bool getEnableVAD() const { return enableVAD; }
    
    // Getters for effect categories
    bool getEnableNoiseEffects() const { return enableNoiseEffects; }
    bool getEnableEchoEffects() const { return enableEchoEffects; }
    bool getEnableResolutionEffects() const { return enableResolutionEffects; }
    
private:
    GLFWwindow* window;
    bool initialized;
    bool isCapturing;
        
    // Audio device lists
    std::vector<std::string> inputDevices;
    std::vector<std::string> outputDevices;
    std::vector<std::string> audioDevices_;  // Combined list of audio devices
    std::string selectedDevice_;  // Currently selected device
    int selectedInputDevice;
    int selectedOutputDevice;
    bool enablePassthrough;
    
    // Audio settings
    int sampleRate;
    int bufferSize;
    float inputGain_;  // Added input gain member
    float gainValue_;  // Current gain value
    std::function<void(const std::string&)> deviceChangeCallback_;  // Device change callback
    
    // NVIDIA Audio Effects settings
    bool enableNoiseReduction;
    float noiseReductionLevel;
    
    // Extended NVIDIA Audio Effect settings
    int selectedEffect;              // 0=Denoiser, 1=Dereverb, 2=Dereverb+Denoiser, 3=SuperRes, 4=AEC
    bool enableVAD;                  // Voice Activity Detection
    bool enableRoomEchoRemoval;      // Room Echo Removal (Dereverb)
    float roomEchoRemovalLevel;      // Room Echo Removal level (0.0-1.0)
    bool enableSuperResolution;      // Audio Super-Resolution
    int superResOutSampleRate;
    bool enableAEC;                  // Acoustic Echo Cancellation
    
    // Effect categories
    bool enableNoiseEffects;
    bool enableEchoEffects;
    bool enableResolutionEffects;
    
    // Recording settings
    bool enableRecording;
    bool enableAutoSave;
    std::string autoSaveDirectory;
    
    // Audio level and spectrum visualization
    float currentLevel_;
    float peakLevel_;
    std::vector<float> inputSpectrumData_;
    std::vector<float> outputSpectrumData_;
    std::vector<float> inputMelSpectrumData_;
    std::vector<float> outputMelSpectrumData_;
    std::mutex levelMutex_;
    std::mutex spectrumMutex_;
    
    // FFT data
    mufft_plan_1d* fftContext_;
    std::vector<float> fftIn_;
    std::vector<float> fftOut_;
    std::vector<std::vector<float>> melFilterbank_;
    
    // Callbacks
    StartCaptureCallback onStartCapture;
    StopCaptureCallback onStopCapture;
    SaveAudioCallback onSaveInputAudio;
    SaveAudioCallback onSaveOutputAudio;
    RecordingStateCallback onRecordingStateChange;
    AutoSaveStateCallback onAutoSaveStateChange;
    AutoSaveDirCallback onAutoSaveDirChange;
    GainChangeCallback onGainChange;  // Added gain change callback
    
    // UI rendering
    void renderMainWindow();
    void renderDeviceSelection();
    void renderAudioSettings();
    void renderProcessingSettings();
    void renderStatusBar();
    void renderAudioVisualization();
    void renderRecordingControls();
    void renderABTestingControls();
    void renderAudioControls();  // Added this declaration
    
    // Device management
    void refreshDeviceList();
    
    // FFT and visualization helpers
    void drawSpectrum(const std::vector<float>& magnitudes, const char* label, ImVec2 size);
    void drawMelSpectrogram(const std::vector<float>& melSpectrum, const char* label, ImVec2 size);
    void initializeMelFilterbank();
    void computeMelSpectrogram(const std::vector<float>& magnitudes, std::vector<float>& melSpectrum);
    
    // Spectrogram instances
    std::unique_ptr<Spectrogram> inputSpectrogram_;
    std::unique_ptr<Spectrogram> outputSpectrogram_;
    std::unique_ptr<AudioCapture> audioCapture_;
}; 