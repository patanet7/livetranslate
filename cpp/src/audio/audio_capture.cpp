#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

// Windows headers first
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <shlobj.h>
#include <objbase.h>
#include <comdef.h>

// Windows multimedia headers - must be in this order
#include <mmsystem.h>
#include <mmreg.h>
#include <ks.h>  // Must come before ksmedia.h
#include <ksmedia.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <propkey.h>
#include <functiondiscoverykeys_devpkey.h>

// Standard library headers
#include <mutex>
#include <thread>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>  // Add this for std::stringstream
#include <cmath>  // For M_PI and math functions

// FFT library
#include "../../include/audio/audio_capture.hpp"
#include "../../include/audio/nvidia_effects.hpp"
#include "../../include/utils/compat_fix.hpp"
#include "../../external/muFFT/fft.h"

// Global callback functions
extern std::function<void(const float*, size_t)> g_audioLevelCallback;
extern std::function<void(const float*, const float*, size_t)> g_spectrumCallback;

// Function to clamp values between min and max (replacement for std::clamp which requires C++17)
template <typename T>
T clamp(T val, T min, T max) {
    return (val < min) ? min : ((val > max) ? max : val);
}

// WAV file format structures
#pragma pack(push, 1)
struct WAVHeader {
    // RIFF chunk
    char riffId[4] = {'R', 'I', 'F', 'F'};
    uint32_t riffSize = 0;
    char waveId[4] = {'W', 'A', 'V', 'E'};
    
    // fmt sub-chunk
    char fmtId[4] = {'f', 'm', 't', ' '};
    uint32_t fmtSize = 16;
    uint16_t audioFormat = 1; // PCM = 1
    uint16_t numChannels = 1;
    uint32_t sampleRate = 44100;
    uint32_t byteRate = 0;    // SampleRate * NumChannels * BitsPerSample/8
    uint16_t blockAlign = 0;  // NumChannels * BitsPerSample/8
    uint16_t bitsPerSample = 16;
    
    // data sub-chunk
    char dataId[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize = 0;
};
#pragma pack(pop)

AudioCapture::AudioCapture(int sampleRate, int channels)
    : sampleRate_(sampleRate)
    , channels_(channels)
    , bufferSize_(512)
    , pEnumerator_(nullptr)
    , pInputDevice_(nullptr)
    , pOutputDevice_(nullptr)
    , pAudioClient_(nullptr)
    , pCaptureClient_(nullptr)
    , pRenderClient_(nullptr)
    , pAudioRenderClient_(nullptr)
    , pwfx_(nullptr)
    , running_(false)
    , enablePassthrough_(false)
    , nvidiaEffectType_(NvidiaEffectType::None)
    , enableNoiseReduction_(false)
    , noiseReductionLevel_(0.0f)
    , enableRoomEchoRemoval_(false)
    , roomEchoRemovalLevel_(0.0f)
    , enableSuperResolution_(false)
    , superResOutSampleRate_(0)
    , enableAEC_(false)
    , enableVAD_(false)
    , frameSize_(0)
    , enableNoiseEffects_(false)
    , enableEchoEffects_(false)
    , enableResolutionEffects_(false)
    , recordingEnabled_(false)
    , autoSaveEnabled_(false)
    , initialized_(false)
    , inputGain_(1.0f) {}

AudioCapture::~AudioCapture() {
    stopCapture();
    cleanup();
}

bool AudioCapture::initialize() {
    HRESULT hr = CoCreateInstance(
        __uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&pEnumerator_));
    if (FAILED(hr)) {
        std::cerr << "Failed to create device enumerator\n";
        return false;
    }
    initialized_ = true;
    return true;
}

void AudioCapture::enumerateDevices() {
    if (!pEnumerator_) return;

    inputDevices_.clear();
    outputDevices_.clear();

    IMMDeviceCollection* collection = nullptr;
    pEnumerator_->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &collection);
    if (collection) {
        UINT count;
        collection->GetCount(&count);
        for (UINT i = 0; i < count; ++i) {
            IMMDevice* device = nullptr;
            collection->Item(i, &device);
            if (device) inputDevices_.push_back(device);
        }
        collection->Release();
    }

    collection = nullptr;
    pEnumerator_->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &collection);
    if (collection) {
        UINT count;
        collection->GetCount(&count);
        for (UINT i = 0; i < count; ++i) {
            IMMDevice* device = nullptr;
            collection->Item(i, &device);
            if (device) outputDevices_.push_back(device);
        }
        collection->Release();
    }
}

bool AudioCapture::startCapture(int deviceIndex, int outputDeviceIndex, bool enablePassthrough) {
    std::cout << "AudioCapture::startCapture: Starting capture process..." << std::endl;
    
    if (running_) {
        std::cout << "AudioCapture::startCapture: Capture already in progress." << std::endl;
        return true;
    }

    // Always ensure we have a valid enumerator
    if (!pEnumerator_) {
        std::cout << "AudioCapture::startCapture: Initializing device enumerator..." << std::endl;
        if (!initialize()) {
            std::cerr << "AudioCapture::startCapture: Failed to initialize device enumerator." << std::endl;
            return false;
        }
    }

    std::cout << "AudioCapture::startCapture: Cleaning up previous state..." << std::endl;
    cleanup(); // Clean up any previous state

    // Re-initialize enumerator after cleanup
    if (!pEnumerator_) {
        std::cout << "AudioCapture::startCapture: Re-initializing device enumerator after cleanup..." << std::endl;
        if (!initialize()) {
            std::cerr << "AudioCapture::startCapture: Failed to re-initialize device enumerator." << std::endl;
            return false;
        }
    }

    // Get the combined list of devices
    std::vector<AudioDevice> devices = getInputDevices();
    if (deviceIndex < 0 || deviceIndex >= static_cast<int>(devices.size())) {
        std::cerr << "AudioCapture::startCapture: Invalid device index: " << deviceIndex << std::endl;
        return false;
    }

    const AudioDevice& selectedDevice = devices[deviceIndex];
    bool isOutputDevice = !selectedDevice.isInput;

    // Enumerate devices before trying to use them
    enumerateDevices();

    // Initialize NVIDIA effects if enabled
    if (nvidiaEffectType_ != NvidiaEffectType::None) {
        std::cout << "AudioCapture::startCapture: Initializing NVIDIA effects..." << std::endl;
        std::cout << "Effect type: " << static_cast<int>(nvidiaEffectType_) << std::endl;
        
        // Initialize NVIDIA effects with current settings
        if (!nvidiaEffects_.initialize(sampleRate_, channels_)) {
            std::cerr << "AudioCapture::startCapture: Failed to initialize NVIDIA effects" << std::endl;
            // Continue without effects rather than failing
        } else {
            std::cout << "AudioCapture::startCapture: NVIDIA effects initialized successfully" << std::endl;
            
            // Set effect parameters
            nvidiaEffects_.setEffectType(nvidiaEffectType_);
            nvidiaEffects_.setNoiseReductionLevel(noiseReductionLevel_);
            nvidiaEffects_.setRoomEchoRemovalLevel(roomEchoRemovalLevel_);
            nvidiaEffects_.setEnableVAD(enableVAD_);
            
            if (nvidiaEffectType_ == NvidiaEffectType::SuperRes) {
                nvidiaEffects_.setSuperResInputSampleRate(sampleRate_);
                nvidiaEffects_.setSuperResOutputSampleRate(superResOutSampleRate_);
            }
            
            std::cout << "AudioCapture::startCapture: NVIDIA effect parameters configured" << std::endl;
        }
    }

    if (isOutputDevice) {
        // For output devices, we'll use loopback capture
        std::cout << "AudioCapture::startCapture: Setting up output device for loopback capture..." << std::endl;
        
        // Find the output device in our list
        for (auto& device : outputDevices_) {
            LPWSTR deviceId = nullptr;
            if (SUCCEEDED(device->GetId(&deviceId))) {
                if (std::wstring(deviceId) == selectedDevice.id) {
                    pOutputDevice_ = device;
                    pOutputDevice_->AddRef();
                    CoTaskMemFree(deviceId);
                    break;
                }
                CoTaskMemFree(deviceId);
            }
        }

        if (!pOutputDevice_) {
            std::cerr << "AudioCapture::startCapture: Failed to find selected output device" << std::endl;
            return false;
        }

        // Initialize audio client with the output device for loopback capture
        WAVEFORMATEX* tempFormat = nullptr;
        if (!initializeAudioClient(pOutputDevice_, &pAudioClient_, &tempFormat, false)) {
            std::cerr << "AudioCapture::startCapture: Failed to initialize output audio client for loopback." << std::endl;
            if(tempFormat) {
                std::cout << "AudioCapture::startCapture: Freeing temporary format..." << std::endl;
                CoTaskMemFree(tempFormat);
            }
            cleanup();
            return false;
        }

        // Free previous pwfx_ if it exists, then assign the new one
        if (pwfx_) {
            std::cout << "AudioCapture::startCapture: Freeing previous wave format..." << std::endl;
            CoTaskMemFree(pwfx_);
        }
        pwfx_ = tempFormat; // pwfx_ now owns this memory

        std::cout << "AudioCapture::startCapture: Getting capture client service..." << std::endl;
        HRESULT hr = pAudioClient_->GetService(__uuidof(IAudioCaptureClient), (void**)&pCaptureClient_);
        if (FAILED(hr)) {
            std::cerr << "AudioCapture::startCapture: Failed to get capture client service. HRESULT: 0x"
                      << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
            cleanupAudioClients();
            return false;
        }
        std::cout << "AudioCapture::startCapture: Capture client service obtained successfully." << std::endl;
    } else {
        // For input devices, use normal capture
        std::cout << "AudioCapture::startCapture: Setting up input device..." << std::endl;
        
        // Find the input device in our list
        for (auto& device : inputDevices_) {
            LPWSTR deviceId = nullptr;
            if (SUCCEEDED(device->GetId(&deviceId))) {
                if (std::wstring(deviceId) == selectedDevice.id) {
                    pInputDevice_ = device;
                    pInputDevice_->AddRef();
                    CoTaskMemFree(deviceId);
                    break;
                }
                CoTaskMemFree(deviceId);
            }
        }

        if (!pInputDevice_) {
            std::cerr << "AudioCapture::startCapture: Failed to find selected input device" << std::endl;
            return false;
        }

        std::cout << "AudioCapture::startCapture: Initializing audio client..." << std::endl;
        WAVEFORMATEX* tempInputFormat = nullptr;
        if (!initializeAudioClient(pInputDevice_, &pAudioClient_, &tempInputFormat, true)) {
            std::cerr << "AudioCapture::startCapture: Failed to initialize input audio client." << std::endl;
            if(tempInputFormat) {
                std::cout << "AudioCapture::startCapture: Freeing temporary input format..." << std::endl;
                CoTaskMemFree(tempInputFormat);
            }
            cleanup();
            return false;
        }

        // Free previous pwfx_ if it exists, then assign the new one
        if (pwfx_) {
            std::cout << "AudioCapture::startCapture: Freeing previous wave format..." << std::endl;
            CoTaskMemFree(pwfx_);
        }
        pwfx_ = tempInputFormat; // pwfx_ now owns this memory

        std::cout << "AudioCapture::startCapture: Getting capture client service..." << std::endl;
        HRESULT hr = pAudioClient_->GetService(__uuidof(IAudioCaptureClient), (void**)&pCaptureClient_);
        if (FAILED(hr)) {
            std::cerr << "AudioCapture::startCapture: Failed to get capture client service. HRESULT: 0x"
                      << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
            cleanupAudioClients();
            return false;
        }
        std::cout << "AudioCapture::startCapture: Capture client service obtained successfully." << std::endl;
    }

    // Handle passthrough if enabled
    enablePassthrough_ = enablePassthrough;
    if (enablePassthrough_) {
        // If we're already capturing from an output device, we don't need to set up another one
        if (!isOutputDevice) {
            if (outputDeviceIndex >= 0 && outputDeviceIndex < static_cast<int>(outputDevices_.size())) {
                pOutputDevice_ = outputDevices_[outputDeviceIndex];
                pOutputDevice_->AddRef();
            } else {
                HRESULT hr_default_out = pEnumerator_->GetDefaultAudioEndpoint(eRender, eConsole, &pOutputDevice_);
                if (FAILED(hr_default_out) || !pOutputDevice_) {
                    std::cerr << "AudioCapture::startCapture: Passthrough enabled, but invalid output device index ("
                              << outputDeviceIndex << ") and failed to get default output device. HRESULT: 0x"
                              << std::hex << hr_default_out << std::dec << " (" << AudioCapture::HResultToString(hr_default_out) << ")" << std::endl;
                    if (pOutputDevice_) { 
                        pOutputDevice_->Release(); 
                        pOutputDevice_ = nullptr; 
                    }
                    enablePassthrough_ = false;
                } else {
                    std::cout << "AudioCapture::startCapture: Using default output device for passthrough." << std::endl;
                }
            }

            if (enablePassthrough_ && pOutputDevice_) {
                WAVEFORMATEX* tempOutputFormat = nullptr;
                if (!initializeAudioClient(pOutputDevice_, &pRenderClient_, &tempOutputFormat, false)) {
                    std::cerr << "AudioCapture::startCapture: Failed to initialize output audio client for passthrough." << std::endl;
                    if(tempOutputFormat) CoTaskMemFree(tempOutputFormat);
                    enablePassthrough_ = false;
                    if (pAudioRenderClient_) { 
                        pAudioRenderClient_->Release(); 
                        pAudioRenderClient_ = nullptr; 
                    }
                    if (pRenderClient_) { 
                        pRenderClient_->Release(); 
                        pRenderClient_ = nullptr; 
                    }
                    if (pOutputDevice_) { 
                        pOutputDevice_->Release(); 
                        pOutputDevice_ = nullptr; 
                    }
                } else {
                    if(tempOutputFormat) CoTaskMemFree(tempOutputFormat);
                }

                if (enablePassthrough_ && pRenderClient_) {
                    HRESULT hr = pRenderClient_->GetService(__uuidof(IAudioRenderClient), (void**)&pAudioRenderClient_);
                    if (FAILED(hr)) {
                        std::cerr << "AudioCapture::startCapture: Failed to get render client service. HRESULT: 0x"
                                  << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
                        enablePassthrough_ = false;
                        if (pAudioRenderClient_) {
                            pAudioRenderClient_->Release(); 
                            pAudioRenderClient_ = nullptr;
                        }
                        if (pRenderClient_) {
                            pRenderClient_->Release(); 
                            pRenderClient_ = nullptr;
                        }
                        if (pOutputDevice_) {
                            pOutputDevice_->Release(); 
                            pOutputDevice_ = nullptr;
                        }
                    } else {
                        std::cout << "AudioCapture::startCapture: Render client service obtained." << std::endl;
                    }
                }
            }
        }
    }

    HRESULT hr = pAudioClient_->Start();
    if (FAILED(hr)) {
        std::cerr << "AudioCapture::startCapture: Failed to start audio client. HRESULT: 0x"
                  << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
        cleanupAudioClients();
        return false;
    }
    std::cout << "AudioCapture::startCapture: Audio client started." << std::endl;

    if (enablePassthrough_ && pRenderClient_ && pAudioRenderClient_) {
        hr = pRenderClient_->Start();
        if (FAILED(hr)) {
            std::cerr << "AudioCapture::startCapture: Failed to start output audio client. HRESULT: 0x"
                      << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
            pAudioClient_->Stop();
            cleanupAudioClients();
            return false;
        }
        std::cout << "AudioCapture::startCapture: Output audio client started for passthrough." << std::endl;
    }

    running_ = true;
    captureThread_ = std::make_unique<std::thread>(&AudioCapture::captureThread, this);
    std::cout << "AudioCapture::startCapture: Capture thread started." << std::endl;

    return true;
}

void AudioCapture::stopCapture() {
    running_ = false;
    if (captureThread_ && captureThread_->joinable()) {
        captureThread_->join();
    }
    if (pAudioClient_) pAudioClient_->Stop();
    if (pRenderClient_)  pRenderClient_->Stop();

    // Save recordings if enabled
    if (recordingEnabled_ && !autoSaveDirectory_.empty()) {
        // Create save directory if it doesn't exist
        std::filesystem::create_directories(autoSaveDirectory_);

        // Generate timestamp for filenames
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();
        
        std::string timestampStr = std::to_string(timestamp);
        
        // Save input file
        std::string inputFilename = autoSaveDirectory_ + "/input_" + timestampStr + ".wav";
        saveWavFile(inputFilename, recordedInputBuffer_, sampleRate_);
        
        // Save output file
        std::string outputFilename = autoSaveDirectory_ + "/output_" + timestampStr + ".wav";
        saveWavFile(outputFilename, recordedOutputBuffer_, sampleRate_);
        
        // Save metadata
        std::string metadataFilename = autoSaveDirectory_ + "/metadata_" + timestampStr + ".json";
        saveMetadata(metadataFilename);
    }
}

void AudioCapture::cleanup() {
    cleanupAudioClients();
    if (pEnumerator_) { pEnumerator_->Release(); pEnumerator_ = nullptr; }
    for (auto& dev : inputDevices_) dev->Release();
    for (auto& dev : outputDevices_) dev->Release();
    inputDevices_.clear();
    outputDevices_.clear();
}

void AudioCapture::cleanupAudioClients() {
    std::cout << "AudioCapture::cleanupAudioClients: Starting cleanup process..." << std::endl;

    // First stop any running audio clients
    if (pAudioClient_) {
        std::cout << "AudioCapture::cleanupAudioClients: Stopping audio client..." << std::endl;
        pAudioClient_->Stop();
    }
    if (pRenderClient_) {
        std::cout << "AudioCapture::cleanupAudioClients: Stopping render client..." << std::endl;
        pRenderClient_->Stop();
    }

    // Release capture client first
    if (pCaptureClient_) { 
        std::cout << "AudioCapture::cleanupAudioClients: Releasing capture client..." << std::endl;
        pCaptureClient_->Release(); 
        pCaptureClient_ = nullptr; 
    }
    
    // Release audio clients
    if (pAudioClient_) { 
        std::cout << "AudioCapture::cleanupAudioClients: Releasing audio client..." << std::endl;
        pAudioClient_->Release(); 
        pAudioClient_ = nullptr; 
    }
    
    if (pRenderClient_) { 
        std::cout << "AudioCapture::cleanupAudioClients: Releasing render client..." << std::endl;
        pRenderClient_->Release(); 
        pRenderClient_ = nullptr; 
    }

    // Release render client
    if (pAudioRenderClient_) { 
        std::cout << "AudioCapture::cleanupAudioClients: Releasing audio render client..." << std::endl;
        pAudioRenderClient_->Release(); 
        pAudioRenderClient_ = nullptr; 
    }
    
    // Release devices last
    if (pInputDevice_) { 
        std::cout << "AudioCapture::cleanupAudioClients: Releasing input device..." << std::endl;
        pInputDevice_->Release(); 
        pInputDevice_ = nullptr; 
    }
    
    if (pOutputDevice_) { 
        std::cout << "AudioCapture::cleanupAudioClients: Releasing output device..." << std::endl;
        pOutputDevice_->Release(); 
        pOutputDevice_ = nullptr; 
    }

    // Free wave format
    if (pwfx_) { 
        std::cout << "AudioCapture::cleanupAudioClients: Freeing wave format..." << std::endl;
        CoTaskMemFree(pwfx_); 
        pwfx_ = nullptr; 
    }

    // Clear device lists without releasing (they are managed by enumerateDevices)
    std::cout << "AudioCapture::cleanupAudioClients: Clearing device lists..." << std::endl;
    inputDevices_.clear();
    outputDevices_.clear();

    std::cout << "AudioCapture::cleanupAudioClients: Cleanup completed successfully." << std::endl;
}

bool AudioCapture::initializeAudioClient(IMMDevice* device, IAudioClient** client, WAVEFORMATEX** format, bool isInput) {
    std::cout << "AudioCapture::initializeAudioClient: Starting initialization..." << std::endl;
    
    if (!device) {
        std::cerr << "AudioCapture::initializeAudioClient: Device pointer is null!" << std::endl;
        return false;
    }

    // Activate the audio client
    std::cout << "AudioCapture::initializeAudioClient: Activating audio client..." << std::endl;
    HRESULT hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(client));
    if (FAILED(hr)) {
        std::cerr << "AudioCapture::initializeAudioClient: Failed to activate audio client. HRESULT: 0x"
                  << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
        return false;
    }
    std::cout << "AudioCapture::initializeAudioClient: Audio client activated successfully." << std::endl;

    // Get the mix format
    std::cout << "AudioCapture::initializeAudioClient: Getting mix format..." << std::endl;
    hr = (*client)->GetMixFormat(format);
    if (FAILED(hr)) {
        std::cerr << "AudioCapture::initializeAudioClient: Failed to get mix format. HRESULT: 0x"
                  << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
        (*client)->Release();
        *client = nullptr;
        return false;
    }
    std::cout << "AudioCapture::initializeAudioClient: Mix format obtained successfully. Format: "
              << (*format)->nSamplesPerSec << "Hz, " << (*format)->nChannels << "ch, "
              << (*format)->wBitsPerSample << "bits" << std::endl;

    // Try to initialize with the mix format first
    std::cout << "AudioCapture::initializeAudioClient: Attempting to initialize with mix format..." << std::endl;
    DWORD streamFlags = 0;
    if (!isInput) {
        streamFlags = AUDCLNT_STREAMFLAGS_LOOPBACK;  // Only use LOOPBACK for output devices
    }
    
    hr = (*client)->Initialize(AUDCLNT_SHAREMODE_SHARED,
        streamFlags,
        10000000, 0, *format, nullptr);
    
    if (FAILED(hr)) {
        std::cout << "AudioCapture::initializeAudioClient: Mix format initialization failed, trying alternative formats..." << std::endl;
        
        // Define a set of common formats to try
        struct AudioFormat {
            int sampleRate;
            int channels;
            int bitsPerSample;
            WORD formatTag;
        };

        AudioFormat formats[] = {
            {44100, 2, 16, WAVE_FORMAT_PCM},
            {48000, 2, 16, WAVE_FORMAT_PCM},
            {44100, 1, 16, WAVE_FORMAT_PCM},
            {48000, 1, 16, WAVE_FORMAT_PCM},
            {44100, 2, 24, WAVE_FORMAT_PCM},
            {48000, 2, 24, WAVE_FORMAT_PCM},
            {44100, 2, 32, WAVE_FORMAT_IEEE_FLOAT},
            {48000, 2, 32, WAVE_FORMAT_IEEE_FLOAT}
        };

        bool initialized = false;
        for (const auto& fmt : formats) {
            // Create format structure
            WAVEFORMATEX* testFormat = new WAVEFORMATEX();
            testFormat->wFormatTag = fmt.formatTag;
            testFormat->nChannels = fmt.channels;
            testFormat->nSamplesPerSec = fmt.sampleRate;
            testFormat->wBitsPerSample = fmt.bitsPerSample;
            testFormat->nBlockAlign = (testFormat->nChannels * testFormat->wBitsPerSample) / 8;
            testFormat->nAvgBytesPerSec = testFormat->nSamplesPerSec * testFormat->nBlockAlign;
            testFormat->cbSize = 0;

            std::cout << "AudioCapture::initializeAudioClient: Trying format: "
                      << testFormat->nSamplesPerSec << "Hz, " << testFormat->nChannels << "ch, "
                      << testFormat->wBitsPerSample << "bits" << std::endl;

            // Try to initialize with this format
            hr = (*client)->Initialize(AUDCLNT_SHAREMODE_SHARED,
                streamFlags,  // Use the same stream flags as above
                10000000, 0, testFormat, nullptr);

            if (SUCCEEDED(hr)) {
                std::cout << "AudioCapture::initializeAudioClient: Successfully initialized with format: "
                          << testFormat->nSamplesPerSec << "Hz, " << testFormat->nChannels << "ch, "
                          << testFormat->wBitsPerSample << "bits" << std::endl;
                
                // Free the previous format and set the new one
                CoTaskMemFree(*format);
                *format = testFormat;
                initialized = true;
                break;
            }

            delete testFormat;
        }

        if (!initialized) {
            std::cerr << "AudioCapture::initializeAudioClient: Failed to initialize with any format." << std::endl;
            CoTaskMemFree(*format);
            *format = nullptr;
            (*client)->Release();
            *client = nullptr;
            return false;
        }
    }

    std::cout << "AudioCapture::initializeAudioClient: Audio client initialized successfully." << std::endl;
    return true;
}

void AudioCapture::captureThread() {
    if (!pwfx_ || !pCaptureClient_) {
        std::cerr << "AudioCapture::captureThread: Audio format (pwfx_) or pCaptureClient_ is null. Cannot proceed." << std::endl;
        running_ = false;
        return;
    }

    UINT32 packetLengthFrames = 0; // GetNextPacketSize returns frames for shared mode
    BYTE* pData = nullptr;
    UINT32 numFramesAvailable = 0;
    DWORD flags;
    // UINT64 devicePosition, qpcPosition; // Not used

    std::cout << "AudioCapture::captureThread: Starting capture loop with format "
              << pwfx_->nSamplesPerSec << "Hz, " << pwfx_->nChannels << "ch, "
              << pwfx_->wBitsPerSample << "bits (expected float32)." << std::endl;

    // Initialize FFT buffers
    const size_t fftSize = 2048;  // Power of 2 for efficient FFT
    std::vector<float> fftBuffer(fftSize);
    std::vector<float> window(fftSize);
    
    // Create Hann window
    for (size_t i = 0; i < fftSize; ++i) {
        window[i] = 0.5f * (1.0f - cos(2.0f * M_PI * i / (fftSize - 1)));
    }

    while (running_) {
        // Get the next packet size
        HRESULT hr = pCaptureClient_->GetNextPacketSize(&packetLengthFrames);
        if (FAILED(hr)) {
            std::cerr << "AudioCapture::captureThread: GetNextPacketSize failed. HRESULT: 0x"
                      << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
            continue;
        }

        if (packetLengthFrames == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::cout << "AudioCapture::captureThread: Got packet of " << packetLengthFrames << " frames" << std::endl;

        // Get the capture buffer
        hr = pCaptureClient_->GetBuffer(&pData, &numFramesAvailable, &flags, nullptr, nullptr);
        if (FAILED(hr)) {
            std::cerr << "AudioCapture::captureThread: GetBuffer failed. HRESULT: 0x"
                      << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
            continue;
        }

        // Calculate total number of samples (frames * channels)
        size_t totalFloatSamples = numFramesAvailable * pwfx_->nChannels;
        std::cout << "AudioCapture::captureThread: Processing " << totalFloatSamples << " samples" << std::endl;

        // Ensure process buffer is large enough
        if (processBuffer_.size() < totalFloatSamples) {
            std::cout << "AudioCapture::captureThread: Resizing process buffer to " << totalFloatSamples << " samples" << std::endl;
            processBuffer_.resize(totalFloatSamples);
        }

        if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
            std::cout << "AudioCapture::captureThread: Received silent buffer" << std::endl;
            std::fill(processBuffer_.begin(), processBuffer_.end(), 0.0f);
        } else {
            // Assuming pwfx_ is the 32-bit float format we initialized with
            if (pwfx_->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
                ((WAVEFORMATEXTENSIBLE*)pwfx_)->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT &&
                pwfx_->wBitsPerSample == 32) {
                memcpy(processBuffer_.data(), pData, totalFloatSamples * sizeof(float));
                std::cout << "AudioCapture::captureThread: Copied " << totalFloatSamples << " float samples" << std::endl;
            } else if (pwfx_->wFormatTag == WAVE_FORMAT_IEEE_FLOAT && pwfx_->wBitsPerSample == 32) { // Non-EXTENSIBLE float
                memcpy(processBuffer_.data(), pData, totalFloatSamples * sizeof(float));
                std::cout << "AudioCapture::captureThread: Copied " << totalFloatSamples << " float samples" << std::endl;
            }
            // Add other conversions if needed, e.g., from device's actual capture format if not float
            else {
                std::cerr << "AudioCapture::captureThread: Unexpected capture format tag " << pwfx_->wFormatTag
                          << " or bits " << pwfx_->wBitsPerSample << ". Expected float32. Filling with silence." << std::endl;
                std::fill(processBuffer_.begin(), processBuffer_.end(), 0.0f);
            }
        }

        hr = pCaptureClient_->ReleaseBuffer(numFramesAvailable);
        if (FAILED(hr)) {
            std::cerr << "AudioCapture::captureThread: ReleaseBuffer failed. HRESULT: 0x"
                      << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
            // Potentially critical, consider stopping
        }

        // Apply input gain
        if (inputGain_ != 1.0f) {
            for (size_t i = 0; i < totalFloatSamples; ++i) {
                processBuffer_[i] *= inputGain_;
            }
        }

        // Process the captured audio with the callback if set
        if (callback_) {
            std::cout << "AudioCapture::captureThread: Calling audio processing callback" << std::endl;
            
            // Apply NVIDIA effects if enabled
            if (nvidiaEffectType_ != NvidiaEffectType::None) {
                std::cout << "AudioCapture::captureThread: Applying NVIDIA effects" << std::endl;
                if (!nvidiaEffects_.processAudio(processBuffer_.data(), totalFloatSamples)) {
                    std::cerr << "AudioCapture::captureThread: Failed to process audio with NVIDIA effects" << std::endl;
                }
            }
            
            callback_(processBuffer_.data(), nullptr, totalFloatSamples);
        } else {
            std::cout << "AudioCapture::captureThread: No callback set" << std::endl;
        }

        // Process audio for visualization
        if (totalFloatSamples > 0) {
            // Calculate audio level
            float level = 0.0f;
            for (size_t i = 0; i < totalFloatSamples; ++i) {
                level = std::max(level, std::abs(processBuffer_[i]));
            }
            
            // Update audio level callback
            if (g_audioLevelCallback) {
                g_audioLevelCallback(processBuffer_.data(), totalFloatSamples);
            }
            
            // Process spectrum data
            if (g_spectrumCallback) {
                // Prepare FFT buffer
                size_t samplesToProcess = std::min(totalFloatSamples, fftSize);
                std::vector<float> magnitudes(fftSize / 2);
                
                // Process the FFT with our consistent algorithm
                performFFT(processBuffer_.data(), samplesToProcess, magnitudes);
                
                // Call spectrum callback with processed data
                g_spectrumCallback(processBuffer_.data(), magnitudes.data(), magnitudes.size());
            }
        }

        if (recordingEnabled_) {
            // Record the processed audio
            recordedOutputBuffer_.insert(recordedOutputBuffer_.end(), processBuffer_.begin(), processBuffer_.end());
            std::cout << "AudioCapture::captureThread: Recorded " << totalFloatSamples << " samples" << std::endl;
        }

        if (enablePassthrough_ && pRenderClient_ && pAudioRenderClient_) {
            UINT32 framesPadding;
            HRESULT hr_padding = pAudioClient_->GetCurrentPadding(&framesPadding);
            if (SUCCEEDED(hr_padding)) {
                UINT32 renderBufferSizeFrames;
                HRESULT hr_buf = pAudioClient_->GetBufferSize(&renderBufferSizeFrames);
                if (SUCCEEDED(hr_buf)) {
                    UINT32 framesAvailableToRender = renderBufferSizeFrames - framesPadding;
                    UINT32 framesToWrite = std::min(numFramesAvailable, framesAvailableToRender);

                    if (framesToWrite > 0) {
                        BYTE* pRenderData = nullptr;
                        hr = pAudioRenderClient_->GetBuffer(framesToWrite, &pRenderData);
                        if (SUCCEEDED(hr)) {
                            // Copy processed data to render buffer
                            memcpy(pRenderData, processBuffer_.data(), framesToWrite * pwfx_->nBlockAlign);
                            pAudioRenderClient_->ReleaseBuffer(framesToWrite, 0);
                            std::cout << "AudioCapture::captureThread: Wrote " << framesToWrite << " frames to render buffer" << std::endl;
                        } else {
                            std::cerr << "AudioCapture::captureThread: Render GetBuffer failed. HRESULT: 0x"
                                      << std::hex << hr << std::dec << " (" << AudioCapture::HResultToString(hr) << ")" << std::endl;
                        }
                    }
                } else {
                    std::cerr << "AudioCapture::captureThread: Render GetBufferSize failed. HRESULT: 0x"
                              << std::hex << hr_buf << std::dec << " (" << AudioCapture::HResultToString(hr_buf) << ")" << std::endl;
                }
            } else {
                std::cerr << "AudioCapture::captureThread: Render GetCurrentPadding failed. HRESULT: 0x"
                          << std::hex << hr_padding << std::dec << " (" << AudioCapture::HResultToString(hr_padding) << ")" << std::endl;
            }
        }
    }
    std::cout << "AudioCapture::captureThread: Exiting capture loop." << std::endl;
}

void AudioCapture::performFFT(const float* buffer, size_t bufferSize, std::vector<float>& magnitudes) {
    if (!buffer || bufferSize == 0 || magnitudes.empty()) {
        std::cerr << "AudioCapture::performFFT: Invalid buffer or magnitudes" << std::endl;
        return;
    }

    std::cout << "AudioCapture::performFFT: Processing FFT with " << bufferSize << " samples" << std::endl;

    // Create muFFT plan
    static mufft_plan_1d* fftContext = nullptr;
    static std::vector<float> fftIn;
    static std::vector<float> fftOut;
    static std::vector<float> window;
    static size_t lastBufferSize = 0;
    
    // Reallocate if buffer size changed
    if (lastBufferSize != bufferSize) {
        if (fftContext) {
            mufft_free_plan_1d(fftContext);
        }
        
        fftContext = mufft_create_plan_1d_r2c(static_cast<int>(bufferSize), MUFFT_FLAG_CPU_ANY);
        if (!fftContext) {
            std::cerr << "AudioCapture::performFFT: Failed to create muFFT plan" << std::endl;
            return;
        }
        
        fftIn.resize(bufferSize);
        fftOut.resize(bufferSize * 2); // Complex output (real + imag)
        
        // Create Hann window
        window.resize(bufferSize);
        float windowSum = 0.0f;
        for (size_t i = 0; i < bufferSize; ++i) {
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (bufferSize - 1)));
            windowSum += window[i];
        }
        // Normalize window to preserve signal energy
        float windowScale = bufferSize / windowSum;
        for (size_t i = 0; i < bufferSize; ++i) {
            window[i] *= windowScale;
        }
        
        lastBufferSize = bufferSize;
    }

    // Apply Hann window to input data
    for (size_t i = 0; i < bufferSize; ++i) {
        fftIn[i] = buffer[i] * window[i];
    }

    // Perform FFT using muFFT
    mufft_execute_plan_1d(fftContext, fftOut.data(), fftIn.data());

    // Calculate magnitudes with proper scaling
    const float fftNormalizationFactor = 2.0f / bufferSize; // Compensate for FFT scaling and window energy loss
    
    // Process FFT bins and convert to dB
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        if (i >= bufferSize/2 + 1) {
            magnitudes[i] = 0.0f;
            continue;
        }
        
        float real = fftOut[i * 2];
        float imag = fftOut[i * 2 + 1];
        float magnitude = std::sqrt(real * real + imag * imag) * fftNormalizationFactor;
        
        // Convert to dB with proper reference level (standard audio -100dB to 0dB range)
        float dB = 20.0f * std::log10(magnitude + 1e-9f); // avoid log(0)
        
        // Apply frequency-dependent weighting (A-weighting approximation)
        float freq = static_cast<float>(i) * sampleRate_ / bufferSize;
        float weight = 1.0f;
        if (freq > 20.0f) {
            float f2 = freq * freq;
            float f4 = f2 * f2;
            // Simplified A-weighting approximation
            weight = (12200.0f * 12200.0f * f4) / 
                    ((f2 + 20.6f * 20.6f) * 
                     (f2 + 12200.0f * 12200.0f) * 
                     std::sqrt(f2 + 107.7f * 107.7f) * 
                     std::sqrt(f2 + 737.9f * 737.9f));
        }
        
        // Apply weighting and normalize to 0-1 range for visualization
        // Standard range for audio visualization is -60dB to 0dB
        float weightedDb = dB + 20.0f * std::log10(weight);
        float normalizedMagnitude = (weightedDb + 60.0f) / 60.0f;
        magnitudes[i] = std::max(0.0f, std::min(normalizedMagnitude, 1.0f));
    }

    std::cout << "AudioCapture::performFFT: FFT processing complete" << std::endl;
}

void AudioCapture::convertToFloat(const BYTE* inputSamples, float* outputSamples, size_t numSamples) {
    const int16_t* in = reinterpret_cast<const int16_t*>(inputSamples);
    for (size_t i = 0; i < numSamples; ++i) {
        outputSamples[i] = in[i] / 32768.0f;
    }
}

void AudioCapture::convertFromFloat(const float* inputSamples, BYTE* outputSamples, size_t numSamples) {
    int16_t* out = reinterpret_cast<int16_t*>(outputSamples);
    for (size_t i = 0; i < numSamples; ++i) {
        float clamped = std::max(-1.0f, std::min(1.0f, inputSamples[i]));
        out[i] = static_cast<int16_t>(clamped * 32767.0f);
    }
}

void AudioCapture::setAudioProcessingCallback(std::function<void(const float*, const float*, size_t)> cb) {
    callback_ = std::move(cb);
}

void AudioCapture::setRecordingEnabled(bool enabled) {
    recordingEnabled_ = enabled;
}

void AudioCapture::setAutoSaveEnabled(bool enabled) {
    autoSaveEnabled_ = enabled;
}

void AudioCapture::setAutoSaveDirectory(const std::string& dir) {
    autoSaveDirectory_ = dir;
}

void AudioCapture::setSampleRate(int rate) {
    sampleRate_ = rate;
}

void AudioCapture::setBufferSize(int sizeFrames) {
    bufferSize_ = sizeFrames;
}

void AudioCapture::enablePassthrough(bool enable) {
    enablePassthrough_ = enable;
}

void AudioCapture::setNoiseReductionEnabled(bool enabled) {
    enableNoiseReduction_ = enabled;
}

void AudioCapture::setNoiseReductionLevel(float level) {
    noiseReductionLevel_ = level;
}

void AudioCapture::setRoomEchoRemovalEnabled(bool enabled) {
    enableRoomEchoRemoval_ = enabled;
}

void AudioCapture::setRoomEchoRemovalLevel(float level) {
    roomEchoRemovalLevel_ = level;
}

void AudioCapture::setSuperResolutionEnabled(bool enabled) {
    enableSuperResolution_ = enabled;
}

void AudioCapture::setSuperResOutSampleRate(int sampleRate) {
    superResOutSampleRate_ = sampleRate;
}

void AudioCapture::setAECEnabled(bool enabled) {
    enableAEC_ = enabled;
}

void AudioCapture::setVADEnabled(bool enabled) {
    enableVAD_ = enabled;
}

void AudioCapture::setFrameSize(int frameSize) {
    frameSize_ = frameSize;
}

void AudioCapture::setNoiseEffectsEnabled(bool enabled) {
    enableNoiseEffects_ = enabled;
}

void AudioCapture::setEchoEffectsEnabled(bool enabled) {
    enableEchoEffects_ = enabled;
}

void AudioCapture::setResolutionEffectsEnabled(bool enabled) {
    enableResolutionEffects_ = enabled;
}

void AudioCapture::setNvidiaEffectType(NvidiaEffectType type) {
    nvidiaEffectType_ = type;
}

void AudioCapture::clearRecordedAudio() {
    recordedInputBuffer_.clear();
    recordedOutputBuffer_.clear();
}

std::vector<AudioDevice> AudioCapture::getInputDevices() {
    std::vector<AudioDevice> devices;
    if (!pEnumerator_) return devices;

    // Get input devices
    IMMDeviceCollection* collection = nullptr;
    HRESULT hr = pEnumerator_->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &collection);
    if (SUCCEEDED(hr) && collection) {
        UINT count;
        collection->GetCount(&count);
        for (UINT i = 0; i < count; ++i) {
            IMMDevice* device = nullptr;
            hr = collection->Item(i, &device);
            if (SUCCEEDED(hr) && device) {
                // Get device ID
                LPWSTR deviceId = nullptr;
                hr = device->GetId(&deviceId);
                if (SUCCEEDED(hr)) {
                    // Get device friendly name
                    IPropertyStore* props = nullptr;
                    hr = device->OpenPropertyStore(STGM_READ, &props);
                    if (SUCCEEDED(hr)) {
                        PROPVARIANT varName;
                        PropVariantInit(&varName);
                        hr = props->GetValue(PKEY_Device_FriendlyName, &varName);
                        if (SUCCEEDED(hr)) {
                            // Convert wide string to UTF-8 for name
                            int size = WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, nullptr, 0, nullptr, nullptr);
                            if (size > 0) {
                                std::vector<char> buffer(size);
                                WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, buffer.data(), size, nullptr, nullptr);
                                std::string name(buffer.data());
                                // Add "(Input)" suffix to input devices
                                name += " (Input)";
                                devices.emplace_back(name, std::wstring(deviceId), true);
                            }
                        }
                        PropVariantClear(&varName);
                        props->Release();
                    }
                    CoTaskMemFree(deviceId);
                }
                device->Release();
            }
        }
        collection->Release();
    }

    // Get output devices and add them as potential input sources
    collection = nullptr;
    hr = pEnumerator_->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &collection);
    if (SUCCEEDED(hr) && collection) {
        UINT count;
        collection->GetCount(&count);
        for (UINT i = 0; i < count; ++i) {
            IMMDevice* device = nullptr;
            hr = collection->Item(i, &device);
            if (SUCCEEDED(hr) && device) {
                // Get device ID
                LPWSTR deviceId = nullptr;
                hr = device->GetId(&deviceId);
                if (SUCCEEDED(hr)) {
                    // Get device friendly name
                    IPropertyStore* props = nullptr;
                    hr = device->OpenPropertyStore(STGM_READ, &props);
                    if (SUCCEEDED(hr)) {
                        PROPVARIANT varName;
                        PropVariantInit(&varName);
                        hr = props->GetValue(PKEY_Device_FriendlyName, &varName);
                        if (SUCCEEDED(hr)) {
                            // Convert wide string to UTF-8 for name
                            int size = WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, nullptr, 0, nullptr, nullptr);
                            if (size > 0) {
                                std::vector<char> buffer(size);
                                WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, buffer.data(), size, nullptr, nullptr);
                                std::string name(buffer.data());
                                // Add "(Output)" suffix to output devices
                                name += " (Output)";
                                devices.emplace_back(name, std::wstring(deviceId), false);
                            }
                        }
                        PropVariantClear(&varName);
                        props->Release();
                    }
                    CoTaskMemFree(deviceId);
                }
                device->Release();
            }
        }
        collection->Release();
    }

    return devices;
}

std::vector<AudioDevice> AudioCapture::getOutputDevices() {
    std::vector<AudioDevice> devices;
    if (!pEnumerator_) return devices;

    IMMDeviceCollection* collection = nullptr;
    HRESULT hr = pEnumerator_->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &collection);
    if (FAILED(hr) || !collection) return devices;

    UINT count;
    collection->GetCount(&count);
    for (UINT i = 0; i < count; ++i) {
        IMMDevice* device = nullptr;
        hr = collection->Item(i, &device);
        if (SUCCEEDED(hr) && device) {
            // Get device ID
            LPWSTR deviceId = nullptr;
            hr = device->GetId(&deviceId);
            if (SUCCEEDED(hr)) {
                // Get device friendly name
                IPropertyStore* props = nullptr;
                hr = device->OpenPropertyStore(STGM_READ, &props);
                if (SUCCEEDED(hr)) {
                    PROPVARIANT varName;
                    PropVariantInit(&varName);
                    hr = props->GetValue(PKEY_Device_FriendlyName, &varName);
                    if (SUCCEEDED(hr)) {
                        // Convert wide string to UTF-8 for name
                        int size = WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, nullptr, 0, nullptr, nullptr);
                        if (size > 0) {
                            std::vector<char> buffer(size);
                            WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, buffer.data(), size, nullptr, nullptr);
                            std::string name(buffer.data());
                            // Create AudioDevice with the proper constructor
                            devices.emplace_back(name, std::wstring(deviceId), false);
                        }
                    }
                    PropVariantClear(&varName);
                    props->Release();
                }
                CoTaskMemFree(deviceId);
            }
            device->Release();
        }
    }
    collection->Release();
    return devices;
}

std::string AudioCapture::HResultToString(HRESULT hr) {
    char buffer[256];
    sprintf_s(buffer, "HRESULT 0x%08X", static_cast<unsigned int>(hr));
    return std::string(buffer);
}

void AudioCapture::saveWavFile(const std::string& filename, const std::vector<float>& buffer, int sampleRate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // WAV header
    struct WavHeader {
        char riff[4] = {'R', 'I', 'F', 'F'};
        uint32_t chunkSize;
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt[4] = {'f', 'm', 't', ' '};
        uint32_t fmtChunkSize = 16;
        uint16_t audioFormat = 1;  // PCM
        uint16_t numChannels = 1;  // Mono
        uint32_t sampleRate;
        uint32_t byteRate;
        uint16_t blockAlign;
        uint16_t bitsPerSample = 32;  // 32-bit float
        char data[4] = {'d', 'a', 't', 'a'};
        uint32_t dataChunkSize;
    } header;

    header.sampleRate = sampleRate;
    header.byteRate = sampleRate * sizeof(float);
    header.blockAlign = sizeof(float);
    header.dataChunkSize = buffer.size() * sizeof(float);
    header.chunkSize = 36 + header.dataChunkSize;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
}

void AudioCapture::saveMetadata(const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open metadata file for writing: " << filename << std::endl;
        return;
    }

    // Create JSON metadata
    nlohmann::json metadata = {
        {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()},
        {"sample_rate", sampleRate_},
        {"effects", {
            {"noise_reduction", {
                {"enabled", enableNoiseReduction_},
                {"level", noiseReductionLevel_}
            }},
            {"room_echo_removal", {
                {"enabled", enableRoomEchoRemoval_},
                {"level", roomEchoRemovalLevel_}
            }},
            {"super_resolution", {
                {"enabled", enableSuperResolution_},
                {"target_rate", superResOutSampleRate_}
            }},
            {"echo_cancellation", {
                {"enabled", enableAEC_}
            }},
            {"voice_activity_detection", {
                {"enabled", enableVAD_}
            }}
        }}
    };

    file << metadata.dump(4);
}

void AudioCapture::setInputGain(float gain) {
    inputGain_ = gain;
}

void AudioCapture::saveComparisonFiles(const std::vector<float>& inputBuffer, const std::vector<float>& outputBuffer) {
    if (inputBuffer.empty() || outputBuffer.empty()) {
        return;
    }

    // Create save directory if it doesn't exist
    std::filesystem::create_directories(autoSaveDirectory_);

    // Generate timestamp for filenames
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    
    std::string timestampStr = std::to_string(timestamp);
    
    // Save input file
    std::string inputFilename = autoSaveDirectory_ + "/input_" + timestampStr + ".wav";
    saveWavFile(inputFilename, inputBuffer, sampleRate_);
    
    // Save output file
    std::string outputFilename = autoSaveDirectory_ + "/output_" + timestampStr + ".wav";
    saveWavFile(outputFilename, outputBuffer, sampleRate_);
    
    // Save metadata
    std::string metadataFilename = autoSaveDirectory_ + "/metadata_" + timestampStr + ".json";
    saveMetadata(metadataFilename);
}
