#pragma once

// Windows headers first
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#include <windows.h>
#include <mmsystem.h>
#include <mmreg.h>
#include <ks.h>
#include <ksmedia.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <functiondiscoverykeys_devpkey.h>
#include <propkey.h>

// Standard library headers
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <memory>
#include <mutex>
#include <filesystem>
#include <chrono>
#include <nlohmann/json.hpp>

// Project headers
#include "../utils/compat_fix.hpp"
#include "nvidia_effects.hpp"
#include "../utils/websocket_client.hpp"

// Global callback declarations
extern std::function<void(const float*, size_t)> g_audioLevelCallback;
extern std::function<void(const float*, const float*, size_t)> g_spectrumCallback;

// Forward declarations
struct IAudioClient;
struct IAudioCaptureClient;
struct IAudioRenderClient;
struct IMMDevice;
struct IMMDeviceEnumerator;

struct AudioDevice {
    std::string name;
    std::wstring id;
    bool isDefault;
    bool isInput;
    bool isOutput;

    AudioDevice(const std::string& name_, const std::wstring& id_, bool isInput_)
        : name(name_), id(id_), isDefault(false), isInput(isInput_), isOutput(!isInput_) {}
};

class AudioCapture {
public:
    using AudioCallback = std::function<void(const float*, const float*, size_t)>;
    
    AudioCapture(int sampleRate = 16000, int channels = 1);
    ~AudioCapture();

    bool initialize();
    bool start(AudioCallback callback);
    void stop();
    
    // Device selection
    bool setInputDevice(int deviceIndex);
    bool setOutputDevice(int deviceIndex);
    
    // Start/stop simplified interface
    bool startCapture(int inputDeviceIndex, int outputDeviceIndex, bool enablePassthrough);
    void stopCapture();
    
    // Get device lists
    std::vector<AudioDevice> getInputDevices();
    std::vector<AudioDevice> getOutputDevices();
    
    // Settings
    void enablePassthrough(bool enable);
    
    // NVIDIA Audio Effects settings
    void setNoiseReductionEnabled(bool enabled);
    void setNoiseReductionLevel(float level);
    void setRoomEchoRemovalEnabled(bool enabled);
    void setRoomEchoRemovalLevel(float level);
    void setSuperResolutionEnabled(bool enabled);
    void setSuperResOutSampleRate(int sampleRate);
    void setAECEnabled(bool enabled);
    void setVADEnabled(bool enabled);
    void setFrameSize(int frameSize);
    
    // Effect categories
    void setNoiseEffectsEnabled(bool enabled);
    void setEchoEffectsEnabled(bool enabled);
    void setResolutionEffectsEnabled(bool enabled);
    
    // Effect type selection
    void setNvidiaEffectType(NvidiaEffectType type);
    
    // WebSocket audio streaming
    bool startWebSocketServer(int port = 8765);
    
    // Enable/disable audio buffer recording
    void setRecordingEnabled(bool enabled);
    bool isRecordingEnabled() const { return recordingEnabled_; }
    
    // Clear recorded audio buffers
    void clearRecordedAudio();
    
    // A/B Testing features
    void setAutoSaveEnabled(bool enabled);
    bool isAutoSaveEnabled() const { return autoSaveEnabled_; }
    void setAutoSaveDirectory(const std::string& directory);
    std::string getAutoSaveDirectory() const { return autoSaveDirectory_; }
    void saveComparisonFiles(const std::vector<float>& inputBuffer, const std::vector<float>& outputBuffer);
    
    // Additional settings
    void setSampleRate(int rate);
    void setEnableNoiseEffects(bool enabled);
    void setEnableEchoEffects(bool enabled);
    void setEnableResolutionEffects(bool enabled);
    void setEnableVAD(bool enabled);

    void setInputGain(float gain);

    // Recording control
    void startRecording();
    void stopRecording();

private:
    // Device pointers
    IMMDeviceEnumerator* pEnumerator_;
    IMMDevice* pInputDevice_;
    IMMDevice* pOutputDevice_;
    IAudioClient* pAudioClient_;
    IAudioCaptureClient* pCaptureClient_;
    IAudioClient* pRenderClient_;
    IAudioRenderClient* pAudioRenderClient_;
    WAVEFORMATEX* pwfx_;  // Audio format structure
    
    // Audio format
    int sampleRate_;
    int channels_;
    int bufferSize_;
    
    // State flags
    bool running_;
    bool enablePassthrough_;
    bool recordingEnabled_;
    bool autoSaveEnabled_;
    bool initialized_;
    
    // NVIDIA effects
    NvidiaEffectType nvidiaEffectType_;
    bool enableNoiseReduction_;
    float noiseReductionLevel_;
    bool enableRoomEchoRemoval_;
    float roomEchoRemovalLevel_;
    bool enableSuperResolution_;
    int superResOutSampleRate_;
    bool enableAEC_;
    bool enableVAD_;
    int frameSize_;
    bool enableNoiseEffects_;
    bool enableEchoEffects_;
    bool enableResolutionEffects_;
    
    // Thread and callback
    std::unique_ptr<std::thread> captureThread_;
    AudioCallback callback_;
    
    // Buffers
    std::vector<float> processBuffer_;
    std::vector<float> recordedInputBuffer_;
    std::vector<float> recordedOutputBuffer_;
    
    // Device lists
    std::vector<IMMDevice*> inputDevices_;
    std::vector<IMMDevice*> outputDevices_;
    
    // Mutex
    std::mutex recordMutex_;
    
    // Auto-save
    std::string autoSaveDirectory_;
    
    // NVIDIA effects processor
    NvidiaAudioEffects nvidiaEffects_;
    
    float inputGain_;  // Input gain control
    
    // Private methods
    void captureThread();
    bool setupWASAPI();
    bool setupRenderDevice();
    void cleanup();
    bool initializeNvidiaEffects();
    
    // Additional private methods used in implementation
    void enumerateDevices();
    bool initializeAudioClient(IMMDevice* device, IAudioClient** audioClientToInit, WAVEFORMATEX** waveFormatEx, bool isInput);
    void cleanupAudioClients();
    void setAudioProcessingCallback(std::function<void(const float*, const float*, size_t)> callback);
    void setBufferSize(int sizeFrames);
    void convertToFloat(const BYTE* inputSamples, float* outputSamples, size_t numSamples);
    void convertFromFloat(const float* inputSamples, BYTE* outputSamples, size_t numSamples);
    
    // Helper functions
    std::vector<AudioDevice> getDeviceList(EDataFlow dataFlow);
    std::wstring getDeviceId(int index, EDataFlow dataFlow);
    bool getDeviceInfo(IMMDevice* pDevice, std::string& name, std::wstring& id);
    
    // FFT processing
    void performFFT(const float* buffer, size_t bufferSize, std::vector<float>& magnitudes);

    // HRESULT helper
    static std::string HResultToString(HRESULT hr);

    // A/B comparison helpers
    void saveWavFile(const std::string& filename, const std::vector<float>& buffer, int sampleRate);
    void saveMetadata(const std::string& filename);
}; 