#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define UNICODE
#define _UNICODE
#define _WIN32_WINNT 0x0A00
#define WINVER 0x0A00

// Note: This file requires linking with comdlg32.lib for GetSaveFileNameW
#include <windows.h>
#include <winsock2.h>
#include <objbase.h>
#include <commdlg.h>
#include <shlobj.h>
#include <commctrl.h>

#include "gui.hpp"
#include "../../include/utils/compat_fix.hpp"
#include "../../include/gui/gl_loader.h"
#include "../../include/audio/audio_capture.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#include <initguid.h>
#include <algorithm>
#include "../../include/utils/math_helpers.h"
#include <deque>
#include <cmath>

// ImGui Platform/Renderer backends
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// Define FFT size constant
#define FFT_SIZE 4096  // Increased from 2048 for better frequency resolution

// Improved log2 function for better accuracy
inline float better_log2(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = static_cast<float>(vx.i);
    y *= 1.1920928955078125e-7f;
    return y - 127.0f - 1.498030302f * (1.0f - x * std::ldexp(1.0f, -static_cast<int>(y - 127.0f)));
}

// Helper function to convert log2 to log10
inline float log2_to_log10(float x) {
    return x * 0.30102999566398119521373889472449f; // log10(2)
}

// Helper function to get device sample rate
int GetDeviceSampleRate(IMMDevice* pDevice) {
    if (!pDevice) return 48000; // Default fallback
    
    HRESULT hr;
    IPropertyStore* pProps = nullptr;
    PROPVARIANT varSampleRate;
    PropVariantInit(&varSampleRate);
    
    hr = pDevice->OpenPropertyStore(STGM_READ, &pProps);
    if (SUCCEEDED(hr)) {
        // Try to get the sample rate from the device properties
        hr = pProps->GetValue(PKEY_AudioEngine_DeviceFormat, &varSampleRate);
        if (SUCCEEDED(hr) && varSampleRate.vt == VT_BLOB) {
            // The format is stored as a WAVEFORMATEX structure in the blob
            WAVEFORMATEX* pWaveFormat = (WAVEFORMATEX*)varSampleRate.blob.pBlobData;
            if (pWaveFormat) {
                int rate = pWaveFormat->nSamplesPerSec;
                PropVariantClear(&varSampleRate);
                pProps->Release();
                return rate;
            }
        }
        PropVariantClear(&varSampleRate);
        pProps->Release();
    }
    return 48000; // Default fallback
}

// Spectrogram implementation
Spectrogram::Spectrogram(int width, int height, int fftSize, int sampleRate) 
    : width_(width), height_(height), fftSize_(fftSize), sampleRate_(sampleRate) {
    // Initialize the spectrogram buffer
    buffer_.resize(width_);
    for (auto& column : buffer_) {
        column.resize(height_, 0.0f);
    }
    
    // Initialize muFFT with SIMD optimizations
    fftContext_ = mufft_create_plan_1d_r2c(fftSize_, MUFFT_FLAG_CPU_ANY);
    if (!fftContext_) {
        std::cerr << "Failed to create muFFT plan" << std::endl;
    }
    
    // Initialize FFT buffers
    fftIn_.resize(fftSize_);
    fftOut_.resize(fftSize_ * 2);  // Complex output (real + imag)
    
    // Initialize window function (Hann window)
    window_.resize(fftSize_);
    float windowSum = 0.0f;
    for (int i = 0; i < fftSize_; ++i) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (fftSize_ - 1)));
        windowSum += window_[i];
    }
    // Normalize window to preserve signal energy
    float windowScale = fftSize_ / windowSum;
    for (int i = 0; i < fftSize_; ++i) {
        window_[i] *= windowScale;
    }
}

Spectrogram::~Spectrogram() {
    if (fftContext_) {
        mufft_free_plan_1d(fftContext_);
    }
}

void Spectrogram::update(const float* audioData, size_t numSamples) {
    if (!fftContext_ || numSamples == 0) return;
    
    // Copy and window the input data
    size_t copySize = std::min(numSamples, static_cast<size_t>(fftSize_));
    for (size_t i = 0; i < copySize; ++i) {
        fftIn_[i] = audioData[i] * window_[i];
    }
    
    // Zero-pad if necessary
    for (size_t i = copySize; i < fftSize_; ++i) {
        fftIn_[i] = 0.0f;
    }
    
    // Perform FFT using muFFT
    mufft_execute_plan_1d(fftContext_, fftOut_.data(), fftIn_.data());
    
    // Calculate magnitudes and convert to dB with improved dynamic range
    std::vector<float> magnitudes(fftSize_ / 2);
    const float fftNormalizationFactor = 2.0f / fftSize_;
    
    // Define dynamic range for visualization
    const float dBMin = -90.0f;  // Floor for visualization
    const float dBMax = 0.0f;    // Ceiling for visualization
    const float dBRange = dBMax - dBMin;
    
    // Apply frequency-dependent gain to emphasize higher frequencies
    const float highFreqGain = 1.5f;  // Boost for high frequencies
    
    for (int i = 0; i < fftSize_ / 2; ++i) {
        float real = fftOut_[i * 2];
        float imag = fftOut_[i * 2 + 1];
        float magnitude = std::sqrt(real * real + imag * imag) * fftNormalizationFactor;
        
        // Convert to dB with proper reference level
        float dB = 20.0f * log2_to_log10(better_log2(magnitude + 1e-9f));
        
        // Apply frequency-dependent gain
        float freq = static_cast<float>(i) * sampleRate_ / fftSize_;
        float freqGain = 1.0f + (highFreqGain - 1.0f) * (freq / (sampleRate_ / 2.0f));
        dB += 20.0f * log2_to_log10(better_log2(freqGain));
        
        // Normalize to 0-1 range with improved dynamic range
        float normalizedMagnitude = (dB - dBMin) / dBRange;
        normalizedMagnitude = std::max(0.0f, std::min(normalizedMagnitude, 1.0f));
        
        // Apply gamma correction for better visibility of low-level signals
        normalizedMagnitude = std::pow(normalizedMagnitude, 0.7f);
        
        magnitudes[i] = normalizedMagnitude;
    }
    
    // Thread-safe update of buffer
    {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        buffer_.pop_front();
        buffer_.push_back(magnitudes);
    }
}

void Spectrogram::draw(ImDrawList* draw_list, ImVec2 pos, ImVec2 size) {
    // Draw pure black background with full opacity
    draw_list->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), 
                            IM_COL32(0, 0, 0, 255));
    
    // Thread-safe access to buffer
    std::vector<std::vector<float>> bufferCopy;
    {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        bufferCopy.reserve(buffer_.size());
        for (const auto& column : buffer_) {
            bufferCopy.push_back(column);
        }
    }
    
    float cellWidth = size.x / width_;
    float cellHeight = size.y / height_;
    
    // Frequency range for logarithmic scaling
    const float freqMin = MIN_FREQ;    // 10 Hz
    const float freqMax = MAX_FREQ;    // 22 kHz
    const float freqLogMin = log2_to_log10(better_log2(freqMin));
    const float freqLogMax = log2_to_log10(better_log2(freqMax));
    
    for (int x = 0; x < width_; ++x) {
        if (x >= bufferCopy.size()) continue;  // Safety check
        const auto& column = bufferCopy[x];
        int nBins = static_cast<int>(column.size());
        
        // Pre-calculate frequency bin mapping for better distribution
        std::vector<int> binMapping(height_);
        for (int y = 0; y < height_; ++y) {
            // Map y position to frequency using logarithmic scaling
            float yNorm = 1.0f - (float)y / height_;
            float freq = freqMin * std::pow(freqMax / freqMin, yNorm);
            
            // Convert frequency to bin index
            float binFreq = freq * fftSize_ / sampleRate_;
            int binIdx = static_cast<int>(binFreq);
            binIdx = std::max(0, std::min(binIdx, nBins - 1));
            binMapping[y] = binIdx;
        }
        
        // Draw the spectrogram column with improved color mapping
        for (int y = 0; y < height_; ++y) {
            int binIdx = binMapping[y];
            float value = column[binIdx];
            
            // Use a perceptually balanced color map (similar to Inferno)
            ImVec4 colorVec;
            if (value <= 0.0f) {
                colorVec = ImVec4(0, 0, 0, 1);
            } else if (value < 0.25f) {
                // Dark blue to purple
                float t = value / 0.25f;
                colorVec = ImVec4(0.1f * t, 0, 0.2f * t, 1);
            } else if (value < 0.5f) {
                // Purple to red
                float t = (value - 0.25f) / 0.25f;
                colorVec = ImVec4(0.1f + 0.9f * t, 0, 0.2f + 0.8f * t, 1);
            } else if (value < 0.75f) {
                // Red to yellow
                float t = (value - 0.5f) / 0.25f;
                colorVec = ImVec4(1, t, 0, 1);
            } else {
                // Yellow to white
                float t = (value - 0.75f) / 0.25f;
                colorVec = ImVec4(1, 1, t, 1);
            }
            
            // Apply gamma correction to the color for better visibility
            colorVec.x = std::pow(colorVec.x, 0.7f);
            colorVec.y = std::pow(colorVec.y, 0.7f);
            colorVec.z = std::pow(colorVec.z, 0.7f);
            
            ImU32 color = ImGui::ColorConvertFloat4ToU32(colorVec);
            
            // Calculate y position with logarithmic scaling
            float yPos = pos.y + size.y * (1.0f - (float)y / height_);
            ImVec2 cellPos(pos.x + x * cellWidth, yPos);
            ImVec2 cellSize(cellWidth, cellHeight);
            draw_list->AddRectFilled(cellPos, 
                                   ImVec2(cellPos.x + cellSize.x, cellPos.y + cellSize.y),
                                   color);
        }
    }
    
    // Draw frequency labels with improved visibility
    const float freqMarkers[] = {10.0f, 50.0f, 100.0f, 500.0f, 1000.0f, 5000.0f, 10000.0f, 22000.0f};
    for (float freq : freqMarkers) {
        float freqLog = log2_to_log10(better_log2(freq));
        float y = pos.y + size.y * (1.0f - (freqLog - freqLogMin) / (freqLogMax - freqLogMin));
        
        std::string label;
        if (freq >= 1000.0f) {
            label = std::to_string(static_cast<int>(freq/1000)) + "k";
        } else {
            label = std::to_string(static_cast<int>(freq));
        }
        
        ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
        draw_list->AddText(ImVec2(pos.x - textSize.x - 5, y - textSize.y/2),
                         IM_COL32(200, 200, 200, 255),
                         label.c_str());
    }
}

GUI::GUI()
    : window(nullptr)
    , initialized(false)
    , isCapturing(false)
    , selectedInputDevice(0)
    , selectedOutputDevice(0)
    , enablePassthrough(false)
    , sampleRate(48000)
    , bufferSize(1024)
    , inputGain_(1.0f)
    , gainValue_(1.0f)
    , selectedDevice_("")
    , enableNoiseReduction(false)
    , noiseReductionLevel(0.5f)
    , selectedEffect(0)
    , enableVAD(false)
    , enableRoomEchoRemoval(false)
    , roomEchoRemovalLevel(0.5f)
    , enableSuperResolution(false)
    , superResOutSampleRate(48000)
    , enableAEC(false)
    , enableNoiseEffects(true)
    , enableEchoEffects(true)
    , enableResolutionEffects(true)
    , enableRecording(false)
    , enableAutoSave(false)
    , currentLevel_(0.0f)
    , peakLevel_(0.0f)
    , onStartCapture(nullptr)
    , onStopCapture(nullptr)
    , onSaveInputAudio(nullptr)
    , onSaveOutputAudio(nullptr)
    , onRecordingStateChange(nullptr)
    , onAutoSaveStateChange(nullptr)
    , onAutoSaveDirChange(nullptr)
    , onGainChange(nullptr)
    , fftContext_(nullptr)
    , audioCapture_(std::make_unique<AudioCapture>()) {
    
    // Initialize FFT buffers
    fftIn_.resize(FFT_SIZE);
    fftOut_.resize(FFT_SIZE * 2);  // Complex output (real + imag)
    
    // Initialize spectrum data
    inputSpectrumData_.resize(FFT_SIZE / 2);
    outputSpectrumData_.resize(FFT_SIZE / 2);
    inputMelSpectrumData_.resize(NUM_MEL_BINS);
    outputMelSpectrumData_.resize(NUM_MEL_BINS);
    
    // Initialize muFFT with SIMD optimizations
    std::cout << "GUI::GUI: Initializing FFT with size " << FFT_SIZE << std::endl;
    fftContext_ = mufft_create_plan_1d_r2c(FFT_SIZE, MUFFT_FLAG_CPU_ANY);
    if (!fftContext_) {
        std::cerr << "GUI::GUI: Failed to create muFFT plan" << std::endl;
    } else {
        std::cout << "GUI::GUI: muFFT plan created successfully" << std::endl;
    }
    
    // Initialize mel filterbank
    initializeMelFilterbank();
    
    // Initialize spectrograms
    inputSpectrogram_ = std::make_unique<Spectrogram>(SPECTRUM_WIDTH, SPECTRUM_HEIGHT, FFT_SIZE, sampleRate);
    outputSpectrogram_ = std::make_unique<Spectrogram>(SPECTRUM_WIDTH, SPECTRUM_HEIGHT, FFT_SIZE, sampleRate);
}

GUI::~GUI() {
    if (fftContext_) {
        mufft_free_plan_1d(fftContext_);
    }
    shutdown();
}

bool GUI::initialize(int width, int height, const char* title) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    
    // Create window with graphics context
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Initialize OpenGL loader
    if (!loadOpenGLFunctions()) {
        std::cerr << "Failed to initialize OpenGL loader" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }
    
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Refresh device lists
    refreshDeviceList();
    
    initialized = true;
    return true;
}

void GUI::run() {
    if (!initialized || !window) {
        return;
    }
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Render our UI
        renderMainWindow();
        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
}

void performFFT(const float* buffer, size_t bufferSize, std::vector<float>& magnitudes);

void GUI::shutdown() {
    if (initialized) {
        // Stop capture if running
        if (isCapturing && onStopCapture) {
            onStopCapture();
            isCapturing = false;
        }
        
        // Cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        glfwTerminate();
        
        initialized = false;
    }
}

void GUI::setCallbacks(StartCaptureCallback startCb, StopCaptureCallback stopCb) {
    onStartCapture = startCb;
    onStopCapture = stopCb;
}

void GUI::setSaveAudioCallbacks(SaveAudioCallback saveInputCb, SaveAudioCallback saveOutputCb) {
    onSaveInputAudio = saveInputCb;
    onSaveOutputAudio = saveOutputCb;
}

void GUI::setRecordingStateCallback(RecordingStateCallback recordingCb) {
    onRecordingStateChange = recordingCb;
}

void GUI::setAutoSaveCallbacks(AutoSaveStateCallback autoSaveCb, AutoSaveDirCallback dirCb) {
    onAutoSaveStateChange = autoSaveCb;
    onAutoSaveDirChange = dirCb;
}

void GUI::setGainChangeCallback(std::function<void(float)> callback) {
    onGainChange = std::move(callback);
}

void GUI::updateAudioLevel(const float* buffer, size_t numSamples) {
    if (!buffer || numSamples == 0) {
        std::cout << "GUI::updateAudioLevel: No buffer or zero samples" << std::endl;
        return;
    }
    
    std::cout << "GUI::updateAudioLevel: Processing " << numSamples << " samples" << std::endl;
    
    // Calculate RMS level
    float sumSquares = 0.0f;
    for (size_t i = 0; i < numSamples; i++) {
        sumSquares += buffer[i] * buffer[i];
    }
    
    float rms = math::sqrt(sumSquares / numSamples);
    
    // Update levels with thread safety
    {
        std::lock_guard<std::mutex> lock(levelMutex_);
        currentLevel_ = rms;
        if (rms > peakLevel_) {
            peakLevel_ = rms;
        }
        
        // Decay peak level over time
        peakLevel_ = peakLevel_ * 0.95f;
    }
    
    std::cout << "GUI::updateAudioLevel: Updated level - RMS: " << rms << ", Peak: " << peakLevel_ << std::endl;
}

void GUI::updateSpectrum(const float* inputBuffer, const float* outputBuffer, size_t numSamples) {
    if ((!inputBuffer && !outputBuffer) || numSamples == 0) {
        std::cout << "GUI::updateSpectrum: No buffers or zero samples" << std::endl;
        return;
    }
    
    std::cout << "GUI::updateSpectrum: Processing " << numSamples << " samples" << std::endl;
    
    // Update spectrograms
    if (inputBuffer && inputSpectrogram_) {
        inputSpectrogram_->update(inputBuffer, numSamples);
    }
    if (outputBuffer && outputSpectrogram_) {
        outputSpectrogram_->update(outputBuffer, numSamples);
    }
    
    // Also update the regular spectrum display
    std::lock_guard<std::mutex> lock(spectrumMutex_);
    
    if (inputBuffer) {
        performFFT(inputBuffer, numSamples, inputSpectrumData_);
        computeMelSpectrogram(inputSpectrumData_, inputMelSpectrumData_);
    }
    
    if (outputBuffer) {
        performFFT(outputBuffer, numSamples, outputSpectrumData_);
        computeMelSpectrogram(outputSpectrumData_, outputMelSpectrumData_);
    }
}

void GUI::performFFT(const float* buffer, size_t bufferSize, std::vector<float>& magnitudes) {
    if (!buffer || bufferSize == 0) {
        return;
    }
    
    static mufft_plan_1d* fftContext = nullptr;
    static std::vector<float> fftIn(FFT_SIZE);
    static std::vector<float> fftOut(FFT_SIZE * 2); // Complex output (real + imag)
    static std::vector<float> window(FFT_SIZE);
    static bool initialized = false;
    
    // Initialize FFT context and window function if not already done
    if (!initialized) {
        // Create muFFT plan for real-to-complex transform
        fftContext = mufft_create_plan_1d_r2c(FFT_SIZE, MUFFT_FLAG_CPU_ANY);
        if (!fftContext) {
            std::cerr << "GUI::performFFT: Failed to create muFFT plan" << std::endl;
            return;
        }
        
        // Initialize Hann window
        float windowSum = 0.0f;
        for (int i = 0; i < FFT_SIZE; ++i) {
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (FFT_SIZE - 1)));
            windowSum += window[i];
        }
        // Normalize window to preserve signal energy
        float windowScale = FFT_SIZE / windowSum;
        for (int i = 0; i < FFT_SIZE; ++i) {
            window[i] *= windowScale;
        }
        
        initialized = true;
    }
    
    // Copy and window the input data
    size_t copySize = std::min(bufferSize, static_cast<size_t>(FFT_SIZE));
    
    // Apply Hann window function
    for (size_t i = 0; i < copySize; ++i) {
        fftIn[i] = buffer[i] * window[i];
    }
    
    // Zero-pad the rest
    for (size_t i = copySize; i < FFT_SIZE; ++i) {
        fftIn[i] = 0.0f;
    }
    
    // Perform FFT using muFFT
    mufft_execute_plan_1d(fftContext, fftOut.data(), fftIn.data());
    
    // Calculate magnitudes with proper scaling
    const float fftNormalizationFactor = 2.0f / FFT_SIZE; // Compensate for FFT scaling and window energy loss
    
    // Define dB range for visualization
    const float MIN_DB = -50.0f;  // Minimum dB level to display
    const float MAX_DB = 10.0f;   // Maximum dB level to display
    const float DB_RANGE = MAX_DB - MIN_DB;
    
    // Process FFT bins and convert to dB
    // For real input, FFT output has complex conjugate symmetry
    // We only need to process up to FFT_SIZE/2 + 1 bins
    for (size_t i = 0; i < FFT_SIZE / 2 + 1; ++i) {
        float real = fftOut[i * 2];
        float imag = fftOut[i * 2 + 1];
        float magnitude = std::sqrt(real * real + imag * imag) * fftNormalizationFactor;
        
        // Convert to dB with proper reference level using better_log2
        float dB = 20.0f * log2_to_log10(better_log2(magnitude + 1e-9f));
        
        // Apply frequency-dependent weighting (A-weighting approximation)
        float freq = static_cast<float>(i) * sampleRate / FFT_SIZE;
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
        float weightedDb = dB + 20.0f * log2_to_log10(better_log2(weight));
        
        // Clamp to our dB range and normalize to 0-1
        weightedDb = std::max(MIN_DB, std::min(weightedDb, MAX_DB));
        float normalizedMagnitude = (weightedDb - MIN_DB) / DB_RANGE;
        
        // Apply temporal smoothing with consistent time constants
        if (i < magnitudes.size()) {
            const float attackTime = 0.2f;  // Fast response to peaks
            const float decayTime = 0.8f;   // Slower decay
            
            if (normalizedMagnitude > magnitudes[i]) {
                magnitudes[i] = attackTime * normalizedMagnitude + (1.0f - attackTime) * magnitudes[i];
            } else {
                magnitudes[i] = decayTime * magnitudes[i] + (1.0f - decayTime) * normalizedMagnitude;
            }
        } else {
            magnitudes[i] = normalizedMagnitude;
        }
    }
}

void GUI::drawSpectrum(const std::vector<float>& magnitudes, const char* label, ImVec2 size) {
    if (magnitudes.empty()) {
        ImGui::Text("No spectrum data available");
        return;
    }

    // Create a unique ID for this spectrum
    std::string uniqueId = std::string(label) + "##spectrum";
    
    // Draw the spectrum title
    ImGui::Text("%s", label);
    
    // Calculate the actual size needed for the spectrum
    // We'll use the full width but add padding for labels
    ImVec2 paddedSize = ImVec2(size.x, size.y + 30.0f);  // Only add vertical padding for frequency labels
    
    // Create a plot with a unique ID
    if (ImGui::BeginChild(uniqueId.c_str(), paddedSize, true)) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size = ImGui::GetContentRegionAvail();
        
        // Reserve space for dB labels on the left
        const float dBLabelWidth = 60.0f;
        pos.x += dBLabelWidth;
        canvas_size.x -= dBLabelWidth;
        
        // Reserve space for frequency labels at the bottom
        canvas_size.y -= 30.0f;
        
        // Draw background grid
        const ImU32 gridColor = IM_COL32(40, 40, 40, 255);
        const ImU32 textColor = IM_COL32(200, 200, 200, 255);
        
        // Frequency markers (logarithmic scale) - standard analyzer frequencies
        const float freqMin = 10.0f;  // 10 Hz
        const float freqMax = 22000.0f;  // 22 kHz
        const float freqLogMin = log2_to_log10(better_log2(freqMin));
        const float freqLogMax = log2_to_log10(better_log2(freqMax));
        
        // Draw frequency grid lines and labels - use standard analyzer frequencies
        const float freqMarkers[] = {20.0f, 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f, 20000.0f};
        for (float freq : freqMarkers) {
            float freqLog = log2_to_log10(better_log2(freq));
            float x = pos.x + canvas_size.x * (freqLog - freqLogMin) / (freqLogMax - freqLogMin);
            
            // Draw vertical grid line
            draw_list->AddLine(
                ImVec2(x, pos.y),
                ImVec2(x, pos.y + canvas_size.y),
                gridColor
            );
            
            // Draw frequency label
            std::string label;
            if (freq >= 1000.0f) {
                label = std::to_string(static_cast<int>(freq/1000)) + "k";
            } else {
                label = std::to_string(static_cast<int>(freq));
            }
            ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
            draw_list->AddText(
                ImVec2(x - textSize.x/2, pos.y + canvas_size.y + 2),
                textColor,
                label.c_str()
            );
        }
        
        // Draw dB grid lines and labels - use professional analyzer levels: 0, -3, -6, -12, -24, -36, -48, -60
        const float dbMarkers[] = {0.0f, -3.0f, -6.0f, -12.0f, -24.0f, -36.0f, -48.0f, -60.0f};
        for (float db : dbMarkers) {
            float normalizedDb = (db + 60.0f) / 60.0f;  // Convert to 0-1 range based on our -60dB range
            float y = pos.y + canvas_size.y * (1.0f - normalizedDb);
            
            // Draw horizontal grid line
            draw_list->AddLine(
                ImVec2(pos.x, y),
                ImVec2(pos.x + canvas_size.x, y),
                gridColor
            );
            
            // Draw dB label
            std::string label = std::to_string(static_cast<int>(db)) + " dB";
            ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
            draw_list->AddText(
                ImVec2(pos.x - textSize.x - 5, y - textSize.y/2),
                textColor,
                label.c_str()
            );
        }
        
        // Prepare spectrum points with logarithmic frequency scaling
        const int numPoints = magnitudes.size();
        std::vector<ImVec2> points;
        points.reserve(numPoints);
        
        // Create more points in low frequency range for smoother curve (professional analyzers use this technique)
        const int extraPointFactor = 4; // Generate 4x more interpolated points for a smoother curve
        const int totalPoints = numPoints * extraPointFactor;
        
        for (int i = 0; i < totalPoints; ++i) {
            // Map the interpolated index back to original array with logarithmic distribution
            float t = static_cast<float>(i) / (totalPoints - 1);
            float logIndex = std::pow(t, 0.5) * (numPoints - 1); // Use square root for more points in low range
            int lowIndex = static_cast<int>(logIndex);
            int highIndex = std::min(lowIndex + 1, numPoints - 1);
            float frac = logIndex - lowIndex;
            
            // Interpolate magnitude value
            float mag;
            if (lowIndex == highIndex) {
                mag = magnitudes[lowIndex];
            } else {
                mag = magnitudes[lowIndex] * (1.0f - frac) + magnitudes[highIndex] * frac;
            }
            
            // Map frequency logarithmically
            float freq = freqMin * std::pow(freqMax/freqMin, t);
            float freqLog = log2_to_log10(better_log2(freq));
            float x = pos.x + canvas_size.x * (freqLog - freqLogMin) / (freqLogMax - freqLogMin);
            float y = pos.y + canvas_size.y * (1.0f - mag);
            
            points.push_back(ImVec2(x, y));
        }
        
        // Draw filled spectrum with gradient
        if (points.size() >= 2) {
            // Professional analyzers use color gradients based on level, not just frequency
            for (size_t i = 0; i < points.size() - 1; ++i) {
                ImVec2 p1 = points[i];
                ImVec2 p2 = points[i + 1];
                
                // Calculate color based on height (level) with a smooth blue-to-red gradient
                float level = 1.0f - ((p1.y - pos.y) / canvas_size.y);
                level = std::max(0.0f, std::min(level, 1.0f));
                
                ImU32 color = ImGui::ColorConvertFloat4ToU32(ImVec4(
                    std::min(1.0f, level * 2.0f),          // R increases faster with level
                    level < 0.5f ? level * 2.0f : (1.0f - level) * 2.0f, // G peaks at middle levels
                    std::max(0.0f, 1.0f - level * 2.0f),  // B decreases with level
                    0.7f                                  // Alpha
                ));
                
                // Draw filled polygon for this segment
                ImVec2 points[4] = {
                    p1,
                    p2,
                    ImVec2(p2.x, pos.y + canvas_size.y),
                    ImVec2(p1.x, pos.y + canvas_size.y)
                };
                draw_list->AddConvexPolyFilled(points, 4, color);
            }
            
            // Draw the curve line with anti-aliased stroke for professional look
            draw_list->AddPolyline(points.data(), static_cast<int>(points.size()), 
                                  IM_COL32(255, 255, 255, 220), false, 1.5f);
        }
    }
    ImGui::EndChild();
}

void GUI::initializeMelFilterbank() {
    // Convert frequency to mel scale
    auto hzToMel = [](float hz) -> float {
        return 2595.0f * log2_to_log10(better_log2(1.0f + hz / 700.0f));
    };
    
    // Convert mel to frequency
    auto melToHz = [](float mel) -> float {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };
    
    // Create mel filterbank
    melFilterbank_.resize(NUM_MEL_BINS);
    
    // Convert min/max frequencies to mel scale
    float minMel = hzToMel(MIN_FREQ);
    float maxMel = hzToMel(MAX_FREQ);
    float melStep = (maxMel - minMel) / (NUM_MEL_BINS + 1);
    
    // Create mel points
    std::vector<float> melPoints(NUM_MEL_BINS + 2);
    for (int i = 0; i < NUM_MEL_BINS + 2; ++i) {
        melPoints[i] = minMel + i * melStep;
    }
    
    // Convert mel points back to Hz
    std::vector<float> hzPoints(NUM_MEL_BINS + 2);
    for (int i = 0; i < NUM_MEL_BINS + 2; ++i) {
        hzPoints[i] = melToHz(melPoints[i]);
    }
    
    // Convert Hz to FFT bin indices
    std::vector<int> fftBins(NUM_MEL_BINS + 2);
    for (int i = 0; i < NUM_MEL_BINS + 2; ++i) {
        fftBins[i] = static_cast<int>(std::floor(hzPoints[i] * FFT_SIZE / sampleRate));
    }
    
    // Create triangular filters
    for (int i = 0; i < NUM_MEL_BINS; ++i) {
        int leftBin = fftBins[i];
        int centerBin = fftBins[i + 1];
        int rightBin = fftBins[i + 2];
        
        melFilterbank_[i].resize(FFT_SIZE / 2, 0.0f);
        
        // Create triangular filter
        for (int j = leftBin; j <= rightBin; ++j) {
            if (j < 0 || j >= FFT_SIZE / 2) continue;
            
            if (j <= centerBin) {
                melFilterbank_[i][j] = static_cast<float>(j - leftBin) / (centerBin - leftBin);
            } else {
                melFilterbank_[i][j] = static_cast<float>(rightBin - j) / (rightBin - centerBin);
            }
        }
        
        // Normalize filter
        float sum = 0.0f;
        for (size_t j = 0; j < melFilterbank_[i].size(); ++j) {
            sum += melFilterbank_[i][j];
        }
        if (sum > 0.0f) {
            for (size_t j = 0; j < melFilterbank_[i].size(); ++j) {
                melFilterbank_[i][j] /= sum;
            }
        }
    }
}

void GUI::computeMelSpectrogram(const std::vector<float>& magnitudes, std::vector<float>& melSpectrum) {
    if (magnitudes.empty() || melSpectrum.size() != NUM_MEL_BINS) return;
    
    // Apply mel filterbank
    for (int i = 0; i < NUM_MEL_BINS; ++i) {
        float melEnergy = 0.0f;
        for (size_t j = 0; j < magnitudes.size(); ++j) {
            melEnergy += magnitudes[j] * melFilterbank_[i][j];
        }
        
        // Convert to log scale and normalize using better_log2
        melSpectrum[i] = log2_to_log10(better_log2(melEnergy + 1e-10f));
    }
    
    // Normalize to 0-1 range
    float minVal = *std::min_element(melSpectrum.begin(), melSpectrum.end());
    float maxVal = *std::max_element(melSpectrum.begin(), melSpectrum.end());
    float range = maxVal - minVal;
    
    if (range > 0.0f) {
        for (size_t i = 0; i < melSpectrum.size(); ++i) {
            melSpectrum[i] = (melSpectrum[i] - minVal) / range;
        }
    }
}

void GUI::drawMelSpectrogram(const std::vector<float>& melSpectrum, const char* label, ImVec2 size) {
    if (melSpectrum.empty()) {
        std::cerr << "GUI::drawMelSpectrogram: Empty mel spectrum data for " << label << std::endl;
        return;
    }
    
    std::cout << "GUI::drawMelSpectrogram: Drawing mel spectrogram for " << label 
              << " with " << melSpectrum.size() << " mel bands" << std::endl;
    
    ImGui::Text("%s", label);
    ImGui::PlotHistogram("##melspectrum", melSpectrum.data(), static_cast<int>(melSpectrum.size()),
                        0, nullptr, 0.0f, 1.0f, size);
}

void GUI::renderAudioVisualization() {
    ImGui::BeginChild("AudioVisualization", ImVec2(0, 0), true);
    
    // Draw spectrograms
    ImGui::Text("Spectrogram Analysis");
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 size = ImVec2(ImGui::GetContentRegionAvail().x, 200);
    
    if (inputSpectrogram_) {
        ImGui::Text("Input Spectrogram");
        inputSpectrogram_->draw(draw_list, pos, size);
        ImGui::Dummy(ImVec2(0, size.y + 20));
    }
    
    pos = ImGui::GetCursorScreenPos();
    if (outputSpectrogram_) {
        ImGui::Text("Output Spectrogram");
        outputSpectrogram_->draw(draw_list, pos, size);
        ImGui::Dummy(ImVec2(0, size.y + 20));
    }
    
    ImGui::Separator();
    
    // Draw audio levels
    ImGui::Text("Audio Levels");
    ImGui::ProgressBar(currentLevel_, ImVec2(-1, 20), "Current");
    ImGui::ProgressBar(peakLevel_, ImVec2(-1, 20), "Peak");
    
    ImGui::Separator();
    
    // Add input gain control
    ImGui::Text("Input Gain");
    if (ImGui::SliderFloat("##InputGain", &inputGain_, 0.0f, 2.0f, "%.2f")) {
        if (onGainChange) {
            onGainChange(inputGain_);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset##gain")) {
        inputGain_ = 1.0f;
        if (onGainChange) {
            onGainChange(inputGain_);
        }
    }
    
    ImGui::EndChild();
}

void GUI::renderRecordingControls() {
    ImGui::BeginGroup();
    ImGui::Text("Audio Recording");
    
    // Record button
    if (ImGui::Button(enableRecording ? "Stop Recording" : "Start Recording", ImVec2(150, 30))) {
        enableRecording = !enableRecording;
        if (onRecordingStateChange) {
            onRecordingStateChange(enableRecording);
        }
    }
    
    // Auto-save controls
    if (ImGui::Checkbox("Enable Auto-Save", &enableAutoSave)) {
        if (onAutoSaveStateChange) {
            onAutoSaveStateChange(enableAutoSave);
        }
    }
    
    // Auto-save directory
    static char saveDir[256] = "";
    if (ImGui::InputText("Save Directory", saveDir, sizeof(saveDir), ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (onAutoSaveDirChange) {
            onAutoSaveDirChange(saveDir);
        }
    }
    
    ImGui::EndGroup();
}

void GUI::renderABTestingControls() {
    ImGui::BeginGroup();
    ImGui::Text("A/B Testing");
    
    // Auto-save checkbox
    if (ImGui::Checkbox("Auto-Save Recordings", &enableAutoSave)) {
        if (onAutoSaveStateChange) {
            onAutoSaveStateChange(enableAutoSave);
        }
    }
    
    // Save directory controls
    ImGui::Text("Save Directory:");
    ImGui::SameLine();
    ImGui::Text("%s", autoSaveDirectory.empty() ? "Not set" : autoSaveDirectory.c_str());
    
    if (ImGui::Button("Browse...", ImVec2(90, 30))) {
        // Show folder browser dialog
        std::string dir = ".";
        if (onAutoSaveDirChange) {
            onAutoSaveDirChange(dir);
        }
    }
    
    ImGui::EndGroup();
}

void GUI::renderMainWindow() {
    // Create a fixed-size window
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, viewport->WorkSize.y));
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | 
                                   ImGuiWindowFlags_NoMove | 
                                   ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus;
    
    ImGui::Begin("LiveTranslate", nullptr, window_flags);
    
    // Main content
    ImGui::Text("LiveTranslate - Real-time Audio Processing");
    ImGui::Separator();
    
    // Calculate available height for columns
    float availableHeight = ImGui::GetContentRegionAvail().y - 60.0f; // Reserve space for capture button
    
    // Split layout into two columns for better organization
    const float columnWidth = ImGui::GetContentRegionAvail().x * 0.5f - 10.0f;
    
    // Left column (device selection, audio settings, NVIDIA effects)
    ImGui::BeginChild("LeftColumn", ImVec2(columnWidth, availableHeight), true);
    
    // Different sections
    renderAudioControls();  // Add audio controls first
    ImGui::Separator();
    
    renderDeviceSelection();
    ImGui::Separator();
    
    renderAudioSettings();
    ImGui::Separator();
    
    renderProcessingSettings();
    ImGui::Separator();
    
    // Add recording controls under the effects
    renderRecordingControls();
    
    ImGui::EndChild();
    
    ImGui::SameLine();
    
    // Right column (visualizations)
    ImGui::BeginChild("RightColumn", ImVec2(0, availableHeight), true);
    
    // Add visualization section
    renderAudioVisualization();
    
    ImGui::EndChild();
    
    // Center the capture button at the bottom
    ImGui::Separator();
    float windowWidth = ImGui::GetWindowWidth();
    float buttonWidth = 120.0f;
    ImGui::SetCursorPosX((windowWidth - buttonWidth) * 0.5f);
    
    if (!isCapturing) {
        if (ImGui::Button("Start Capture", ImVec2(buttonWidth, 40))) {
            if (onStartCapture && onStartCapture(selectedInputDevice, enablePassthrough)) {
                isCapturing = true;
            }
        }
    } else {
        if (ImGui::Button("Stop Capture", ImVec2(buttonWidth, 40))) {
            if (onStopCapture) {
                onStopCapture();
                isCapturing = false;
            }
        }
        
        // Show capture status
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Capturing...");
    }
    
    ImGui::End();
}

void GUI::renderDeviceSelection() {
    ImGui::BeginGroup();
    ImGui::Text("Audio Device Selection");
    
    // Input device dropdown
    if (ImGui::BeginCombo("Input Device", inputDevices.empty() ? "No devices" : inputDevices[selectedInputDevice].c_str())) {
        for (int i = 0; i < inputDevices.size(); i++) {
            const bool isSelected = (selectedInputDevice == i);
            if (ImGui::Selectable(inputDevices[i].c_str(), isSelected)) {
                selectedInputDevice = i;
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    
    // Output device dropdown
    if (ImGui::BeginCombo("Output Device", outputDevices.empty() ? "No devices" : outputDevices[selectedOutputDevice].c_str())) {
        for (int i = 0; i < outputDevices.size(); i++) {
            const bool isSelected = (selectedOutputDevice == i);
            if (ImGui::Selectable(outputDevices[i].c_str(), isSelected)) {
                selectedOutputDevice = i;
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    
    // Enable audio passthrough checkbox
    ImGui::Checkbox("Enable Audio Passthrough", &enablePassthrough);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Play captured audio through the output device");
    }
    
    ImGui::EndGroup();
}

void GUI::renderAudioSettings() {
    ImGui::BeginGroup();
    ImGui::Text("Audio Settings");
    
    // Processing sample rate selection (only 16kHz or 48kHz)
    const char* processingRates[] = { "16000 Hz (Effects)", "48000 Hz (Super Resolution)" };
    int processingRateIndex = (sampleRate == 48000) ? 1 : 0;
    
    if (ImGui::Combo("Processing Sample Rate", &processingRateIndex, processingRates, IM_ARRAYSIZE(processingRates))) {
        sampleRate = (processingRateIndex == 0) ? 16000 : 48000;
    }
    
    // Buffer size selection
    const char* bufferSizes[] = { "512", "1024", "2048", "4096" };
    int bufferSizeIndex = 1; // Default to 1024
    
    switch (bufferSize) {
        case 512: bufferSizeIndex = 0; break;
        case 1024: bufferSizeIndex = 1; break;
        case 2048: bufferSizeIndex = 2; break;
        case 4096: bufferSizeIndex = 3; break;
        default: bufferSizeIndex = 1;
    }
    
    if (ImGui::Combo("Buffer Size", &bufferSizeIndex, bufferSizes, IM_ARRAYSIZE(bufferSizes))) {
        switch (bufferSizeIndex) {
            case 0: bufferSize = 512; break;
            case 1: bufferSize = 1024; break;
            case 2: bufferSize = 2048; break;
            case 3: bufferSize = 4096; break;
        }
    }
    
    ImGui::EndGroup();
}

void GUI::renderProcessingSettings() {
    ImGui::BeginGroup();
    ImGui::Text("NVIDIA Audio Effects");
    
    // Sample rate selection
    const char* sampleRateOptions[] = { "16 kHz", "48 kHz" };
    int sampleRateIdx = (sampleRate == 48000) ? 1 : 0;
    if (ImGui::Combo("Sample Rate", &sampleRateIdx, sampleRateOptions, IM_ARRAYSIZE(sampleRateOptions))) {
        sampleRate = (sampleRateIdx == 0) ? 16000 : 48000;
    }
    
    ImGui::SameLine(200);
    
    // Frame size selection
    const char* frameSizeOptions[] = { "10ms", "20ms" };
    int frameSizeIdx = (bufferSize > 1024) ? 1 : 0;
    if (ImGui::Combo("Frame Size", &frameSizeIdx, frameSizeOptions, IM_ARRAYSIZE(frameSizeOptions))) {
        bufferSize = (frameSizeIdx == 0) ? 1024 : 2048;
    }
    
    // Effect categories
    ImGui::Text("Effect Categories:");
    
    if (ImGui::Checkbox("Noise Reduction", &enableNoiseEffects)) {
        // If enabling noise effects, update related controls
        if (enableNoiseEffects) {
            enableNoiseReduction = true;
        }
    }
    
    ImGui::SameLine(200);
    
    if (ImGui::Checkbox("Echo Cancellation", &enableEchoEffects)) {
        // If enabling echo effects, update related controls
        if (enableEchoEffects) {
            enableRoomEchoRemoval = true;
        }
    }
    
    if (ImGui::Checkbox("Super Resolution", &enableResolutionEffects)) {
        // If enabling resolution effects, update related controls
        if (enableResolutionEffects) {
            enableSuperResolution = true;
        }
    }
    
    ImGui::SameLine(200);
    
    if (ImGui::Checkbox("Voice Activity Detection", &enableVAD)) {
        // VAD is a common option for all effects
    }
    
    ImGui::Separator();
    
    // Effect quality sliders
    ImGui::Text("Effect Quality:");
    
    // Noise reduction level slider
    if (enableNoiseEffects) {
        ImGui::Text("Noise Reduction:");
        ImGui::SameLine(140);
        ImGui::SliderFloat("##NoiseLevel", &noiseReductionLevel, 0.0f, 1.0f, "%.2f");
        ImGui::SameLine();
        if (ImGui::Button("Reset##Noise")) {
            noiseReductionLevel = 0.5f;
        }
    }
    
    // Echo removal level slider
    if (enableEchoEffects) {
        ImGui::Text("Echo Removal:");
        ImGui::SameLine(140);
        ImGui::SliderFloat("##EchoLevel", &roomEchoRemovalLevel, 0.0f, 1.0f, "%.2f");
        ImGui::SameLine();
        if (ImGui::Button("Reset##Echo")) {
            roomEchoRemovalLevel = 0.5f;
        }
    }
    
    // Super-Resolution settings
    if (enableResolutionEffects) {
        ImGui::Text("Output Sample Rate:");
        ImGui::SameLine(140);
        
        const char* outRates[] = { "16000 Hz", "48000 Hz" };
        int outRateIdx = (superResOutSampleRate == 48000) ? 1 : 0;
        
        if (ImGui::Combo("##OutSampleRate", &outRateIdx, outRates, 2)) {
            superResOutSampleRate = (outRateIdx == 1) ? 48000 : 16000;
        }
    }
    
    ImGui::Separator();
    
    // Effect type selection
    const char* effectTypes[] = { 
        "Noise Removal", 
        "Room Echo Removal", 
        "Noise + Echo Removal", 
        "Audio Super-Resolution", 
        "Acoustic Echo Cancellation" 
    };
    
    ImGui::Text("Combined Effect:");
    if (ImGui::Combo("Effect Type", &selectedEffect, effectTypes, IM_ARRAYSIZE(effectTypes))) {
        // Reset appropriate options based on selected effect
        switch (selectedEffect) {
            case 0: // Denoiser
                enableNoiseReduction = true;
                enableRoomEchoRemoval = false;
                enableSuperResolution = false;
                enableAEC = false;
                enableNoiseEffects = true;
                enableEchoEffects = false;
                enableResolutionEffects = false;
                break;
            case 1: // Dereverb
                enableNoiseReduction = false;
                enableRoomEchoRemoval = true;
                enableSuperResolution = false;
                enableAEC = false;
                enableNoiseEffects = false;
                enableEchoEffects = true;
                enableResolutionEffects = false;
                break;
            case 2: // Dereverb + Denoiser
                enableNoiseReduction = true;
                enableRoomEchoRemoval = true;
                enableSuperResolution = false;
                enableAEC = false;
                enableNoiseEffects = true;
                enableEchoEffects = true;
                enableResolutionEffects = false;
                break;
            case 3: // Super-Resolution
                enableNoiseReduction = false;
                enableRoomEchoRemoval = false;
                enableSuperResolution = true;
                enableAEC = false;
                enableNoiseEffects = false;
                enableEchoEffects = false;
                enableResolutionEffects = true;
                break;
            case 4: // AEC
                enableNoiseReduction = false;
                enableRoomEchoRemoval = false;
                enableSuperResolution = false;
                enableAEC = true;
                enableNoiseEffects = false;
                enableEchoEffects = true;
                enableResolutionEffects = false;
                break;
        }
    }
    
    ImGui::EndGroup();
}

void GUI::renderStatusBar() {
    ImGui::Separator();
    ImGui::Text("Status: %s", isCapturing ? "Capturing" : "Idle");
}

void GUI::refreshDeviceList() {
    // Initialize COM if not already initialized
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    bool uninitializeCom = SUCCEEDED(hr);
    
    // Create the device enumerator
    IMMDeviceEnumerator* pEnumerator = nullptr;
    hr = CoCreateInstance(
        CLSID_MMDeviceEnumerator, nullptr,
        CLSCTX_ALL, IID_IMMDeviceEnumerator,
        (void**)&pEnumerator);
    
    if (SUCCEEDED(hr)) {
        // Get the default input device
        IMMDevice* pDevice = nullptr;
        hr = pEnumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &pDevice);
        if (SUCCEEDED(hr)) {
            // Get the actual sample rate from the device
            sampleRate = GetDeviceSampleRate(pDevice);
            std::cout << "Detected input device sample rate: " << sampleRate << " Hz" << std::endl;
            pDevice->Release();
        }
        
        // Get all audio devices
        IMMDeviceCollection* pCollection = nullptr;
        hr = pEnumerator->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &pCollection);
        if (SUCCEEDED(hr)) {
            UINT count;
            pCollection->GetCount(&count);
            
            // Clear existing lists
            inputDevices.clear();
            audioDevices_.clear();
            
            for (UINT i = 0; i < count; i++) {
                IMMDevice* pDevice = nullptr;
                hr = pCollection->Item(i, &pDevice);
                if (SUCCEEDED(hr)) {
                    IPropertyStore* pProps = nullptr;
                    hr = pDevice->OpenPropertyStore(STGM_READ, &pProps);
                    if (SUCCEEDED(hr)) {
                        PROPVARIANT varName;
                        PropVariantInit(&varName);
                        hr = pProps->GetValue(PKEY_Device_FriendlyName, &varName);
                        if (SUCCEEDED(hr) && varName.vt == VT_LPWSTR) {
                            // Convert wide string to UTF-8
                            int size = WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, nullptr, 0, nullptr, nullptr);
                            if (size > 0) {
                                std::string deviceName(size - 1, 0);
                                WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, &deviceName[0], size, nullptr, nullptr);
                                inputDevices.push_back(deviceName);
                                audioDevices_.push_back(deviceName);
                            }
                        }
                        PropVariantClear(&varName);
                        pProps->Release();
                    }
                    pDevice->Release();
                }
            }
            pCollection->Release();
            
            // Set default selected device if none is selected
            if (selectedDevice_.empty() && !audioDevices_.empty()) {
                selectedDevice_ = audioDevices_[0];
            }
        }
        pEnumerator->Release();
    }
    
    if (uninitializeCom) {
        CoUninitialize();
    }
}

void GUI::renderAudioControls() {
    if (ImGui::CollapsingHeader("Audio Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Device selection
        if (ImGui::BeginCombo("Input Device", selectedDevice_.c_str())) {
            for (const auto& device : audioDevices_) {
                bool isSelected = (selectedDevice_ == device);
                if (ImGui::Selectable(device.c_str(), isSelected)) {
                    selectedDevice_ = device;
                    // Notify the application about device change
                    if (deviceChangeCallback_) {
                        deviceChangeCallback_(device);
                    }
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        // Gain control
        float gain = gainValue_;
        if (ImGui::SliderFloat("Input Gain", &gain, 0.0f, 2.0f, "%.2f")) {
            gainValue_ = gain;
            if (onGainChange) {
                onGainChange(gain);
            }
        }
    }
} 