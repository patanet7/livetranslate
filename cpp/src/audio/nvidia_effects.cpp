#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _WINSOCK_DEPRECATED_NO_WARNINGS

// Windows headers first
#include <windows.h>

// Standard library headers
#include <iostream>
#include <algorithm>

// Project headers
#include "../../include/audio/nvidia_effects.hpp"

// Function name definitions
#define NVAFX_CREATEEFFECT_FUNC "NvAFX_CreateEffect"
#define NVAFX_CREATECHAINEDEFFECT_FUNC "NvAFX_CreateChainedEffect"
#define NVAFX_DESTROYEFFECT_FUNC "NvAFX_DestroyEffect"
#define NVAFX_SETSTRING_FUNC "NvAFX_SetString"
#define NVAFX_SETSTRINGLIST_FUNC "NvAFX_SetStringList"
#define NVAFX_SETFLOAT_FUNC "NvAFX_SetFloat"
#define NVAFX_SETU32_FUNC "NvAFX_SetU32"
#define NVAFX_GETU32_FUNC "NvAFX_GetU32"
#define NVAFX_LOAD_FUNC "NvAFX_Load"
#define NVAFX_RUN_FUNC "NvAFX_Run"
#define NVAFX_RESET_FUNC "NvAFX_Reset"

NvidiaAudioEffects::NvidiaAudioEffects()
    : initialized_(false)
    , handle_(nullptr)
    , sdkModule_(nullptr)
    , effectType_(NvidiaEffectType::Denoiser) // Default to denoiser
    , noiseReductionLevel_(0.5f)
    , roomEchoRemovalLevel_(0.5f)
    , enableVAD_(false)
    , sampleRate_(48000)
    , channels_(1)
    , superResInputSampleRate_(16000)
    , superResOutputSampleRate_(48000)
    , chainedEffectType_(nullptr) {
}

NvidiaAudioEffects::~NvidiaAudioEffects() {
    cleanup();
}

bool NvidiaAudioEffects::isAvailable() {
    HMODULE hModule = LoadLibraryA("NVAudioEffects.dll");
    if (hModule != nullptr) {
        FreeLibrary(hModule);
        return true;
    }
    return false;
}

bool NvidiaAudioEffects::loadSDK() {
    // First try to load from system path
    sdkModule_ = LoadLibraryA("NVAudioEffects.dll");
    
    // If that fails, try to load from our local copy
    if (sdkModule_ == nullptr) {
        sdkModule_ = LoadLibraryA("external\\NVIDIA Audio Effects SDK\\NVAudioEffects.dll");
    }
    
    if (sdkModule_ == nullptr) {
        std::cerr << "Failed to load NVIDIA Audio Effects SDK" << std::endl;
        return false;
    }
    
    // Load function pointers
    createEffect_ = reinterpret_cast<CreateEffectFunc>(GetProcAddress((HMODULE)sdkModule_, NVAFX_CREATEEFFECT_FUNC));
    createChainedEffect_ = reinterpret_cast<CreateChainedEffectFunc>(GetProcAddress((HMODULE)sdkModule_, NVAFX_CREATECHAINEDEFFECT_FUNC));
    destroyEffect_ = reinterpret_cast<DestroyEffectFunc>(GetProcAddress((HMODULE)sdkModule_, NVAFX_DESTROYEFFECT_FUNC));
    setString_ = reinterpret_cast<SetStringFunc>(GetProcAddress((HMODULE)sdkModule_, NVAFX_SETSTRING_FUNC));
    setStringList_ = reinterpret_cast<SetStringListFunc>(GetProcAddress((HMODULE)sdkModule_, NVAFX_SETSTRINGLIST_FUNC));
    setFloat_ = reinterpret_cast<SetFloatFunc>(GetProcAddress((HMODULE)sdkModule_, NVAFX_SETFLOAT_FUNC));
    setU32_ = reinterpret_cast<SetU32Func>(GetProcAddress((HMODULE)sdkModule_, NVAFX_SETU32_FUNC));
    getU32_ = reinterpret_cast<GetU32Func>(GetProcAddress((HMODULE)sdkModule_, NVAFX_GETU32_FUNC));
    load_ = reinterpret_cast<LoadFunc>(GetProcAddress((HMODULE)sdkModule_, NVAFX_LOAD_FUNC));
    run_ = reinterpret_cast<RunFunc>(GetProcAddress((HMODULE)sdkModule_, NVAFX_RUN_FUNC));
    reset_ = reinterpret_cast<ResetFunc>(GetProcAddress((HMODULE)sdkModule_, NVAFX_RESET_FUNC));
    
    if (!createEffect_ || !destroyEffect_ || !setString_ || !setFloat_ || !setU32_ || !getU32_ || !load_ || 
        !run_ || !reset_ || !createChainedEffect_ || !setStringList_) {
        std::cerr << "Failed to load one or more functions from NVIDIA Audio Effects SDK" << std::endl;
        return false;
    }
    
    return true;
}

// Add this helper function to get the effect type as a string
std::string NvidiaAudioEffects::getNvidiaEffectTypeString(NvidiaEffectType effectType) const {
    switch (effectType) {
        case NvidiaEffectType::Denoiser:
            return "Denoiser";
        case NvidiaEffectType::Dereverb:
            return "Dereverb";
        case NvidiaEffectType::DereverbDenoiser:
            return "Dereverb+Denoiser";
        case NvidiaEffectType::SuperRes:
            return "Super-Resolution";
        case NvidiaEffectType::AEC:
            return "Acoustic Echo Cancellation";
        case NvidiaEffectType::ChainedEffect:
            return "Chained Effect";
        default:
            return "Unknown";
    }
}

bool NvidiaAudioEffects::initialize(int sampleRate, int channels) {
    sampleRate_ = sampleRate;
    channels_ = channels;
    
    std::cout << "Initializing NVIDIA Audio Effects with sampleRate: " << sampleRate_ 
              << ", channels: " << channels_ << std::endl;
              
    // Try loading the SDK
    if (!loadSDK()) {
        std::cout << "NVIDIA Audio Effects SDK not available - using passthrough mode" << std::endl;
        // Initialize buffer for future use even without SDK
        // Always use 10ms frame size as required by NVIDIA models
        tempBuffer_.resize((sampleRate_ / 100) * channels_);  // 10ms at specified sample rate
        return true; // Return true to allow passthrough mode
    }
    
    std::cout << "NVIDIA Audio Effects SDK loaded successfully" << std::endl;
    std::cout << "Initializing effect type: " << getNvidiaEffectTypeString(effectType_) << std::endl;
    
    // Initialize based on effect type
    bool result = false;
    switch (effectType_) {
        case NvidiaEffectType::Denoiser:
            result = initializeDenoiser();
            break;
        case NvidiaEffectType::Dereverb:
            result = initializeDereverb();
            break;
        case NvidiaEffectType::DereverbDenoiser:
            result = initializeDenoiserDereverb();
            break;
        case NvidiaEffectType::SuperRes:
            result = initializeSuperRes();
            break;
        case NvidiaEffectType::AEC:
            result = initializeAEC();
            break;
        case NvidiaEffectType::ChainedEffect:
            result = initializeChainedEffect();
            break;
        default:
            std::cerr << "Unknown effect type" << std::endl;
            break;
    }
    
    if (result) {
        initialized_ = true;
        std::cout << "NVIDIA Audio Effects SDK initialized successfully for " 
                  << getNvidiaEffectTypeString(effectType_) << " effect" << std::endl;
        std::cout << "Temp buffer size: " << tempBuffer_.size() 
                  << " samples, for " << channels_ << " channels" << std::endl;
        
        // Verify we have a 10ms buffer as required by NVIDIA models
        size_t expectedSize = (sampleRate_ / 100) * channels_; // 10ms buffer
        if (tempBuffer_.size() != expectedSize) {
            std::cout << "Warning: Buffer size " << tempBuffer_.size() << " doesn't match expected 10ms frame size (" 
                      << expectedSize << "). This may cause processing failures." << std::endl;
        }
    } else {
        std::cerr << "Failed to initialize NVIDIA Audio Effects" << std::endl;
    }
    
    return result;
}

bool NvidiaAudioEffects::initializeDenoiser() {
    // Create effect handle
    NvAFX_Handle effectHandle = nullptr;
    if (createEffect_(NVAFX_EFFECT_DENOISER, &effectHandle) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to create denoiser effect" << std::endl;
        return false;
    }
    handle_ = effectHandle;
    
    // Set model path based on sample rate
    bool modelFound = false;
    std::string baseName;
    
    if (sampleRate_ == 48000) {
        baseName = "denoiser_48k.trtpkg";
    } else if (sampleRate_ == 16000) {
        baseName = "denoiser_16k.trtpkg";
    } else {
        std::cerr << "Unsupported sample rate for denoiser: " << sampleRate_ << std::endl;
        return false;
    }
    
    // Try several possible locations for the model file
    std::vector<std::string> possiblePaths = {
        "models\\" + baseName,                                 // Local models directory
        ".\\models\\" + baseName,                              // Explicit local path
        "..\\models\\" + baseName,                             // Up one level in models
        "external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // External directory
        "..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up one level 
        "..\\..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up two levels
        // Absolute paths for common installation locations
        "C:\\Program Files\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName,
        "C:\\Program Files (x86)\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName
    };
    
    for (const auto& path : possiblePaths) {
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fclose(f);
            modelPath_ = path;
            modelFound = true;
            std::cout << "Found model at: " << path << std::endl;
            break;
        } else {
            std::cout << "Tried path (not found): " << path << std::endl;
        }
    }
    
    if (!modelFound) {
        std::cerr << "Failed to find model file for denoiser effect" << std::endl;
        std::cerr << "Make sure model files are in the 'models' directory next to the executable" << std::endl;
        std::cerr << "Required file: " << baseName << std::endl;
        return false;
    }
    
    // Set common parameters
    if (setString_(handle_, NVAFX_PARAM_MODEL_PATH, modelPath_.c_str()) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set model path" << std::endl;
        return false;
    }
    
    if (setU32_(handle_, NVAFX_PARAM_INPUT_SAMPLE_RATE, sampleRate_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set input sample rate: " << sampleRate_ << std::endl;
        return false;
    }
    
    // Disable CUDA graph for better compatibility
    setU32_(handle_, NVAFX_PARAM_DISABLE_CUDA_GRAPH, 1);
    
    // Set intensity ratio 
    setFloat_(handle_, NVAFX_PARAM_INTENSITY_RATIO, noiseReductionLevel_);
    
    // Enable VAD if requested
    if (enableVAD_) {
        setU32_(handle_, NVAFX_PARAM_ENABLE_VAD, 1);
    }
    
    // Load the model
    std::cout << "Loading denoiser model..." << std::endl;
    if (load_(handle_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to load denoiser effect" << std::endl;
        return false;
    }
    std::cout << "Denoiser model loaded successfully" << std::endl;
    
    // After loading, query the channel count
    unsigned int numInputChannels = 0;
    unsigned int numOutputChannels = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_CHANNELS, &numInputChannels) != NVAFX_STATUS_SUCCESS ||
        getU32_(handle_, NVAFX_PARAM_NUM_OUTPUT_CHANNELS, &numOutputChannels) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get channel count" << std::endl;
        return false;
    }
    
    // Make sure our channel count matches the effect's requirement
    if (channels_ != numInputChannels) {
        std::cout << "Note: Adjusting channel count from " << channels_ << " to " << numInputChannels << std::endl;
        channels_ = numInputChannels;
    }
    
    // Get buffer size from SDK, but enforce 10ms frame size
    unsigned int samplesPerFrame = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_SAMPLES_PER_FRAME, &samplesPerFrame) == NVAFX_STATUS_SUCCESS && samplesPerFrame > 0) {
        std::cout << "Using SDK-reported frame size: " << samplesPerFrame << std::endl;
        
        // The SDK should report a 10ms frame size based on documentation
        unsigned int expected10msFrameSize = sampleRate_ / 100;
        if (samplesPerFrame != expected10msFrameSize) {
            std::cout << "Warning: SDK-reported frame size differs from expected 10ms frame size (" 
                      << expected10msFrameSize << "). Using SDK value." << std::endl;
        }
        
        tempBuffer_.resize(samplesPerFrame * channels_);
    } else {
        std::cout << "Could not get frame size from SDK, using 10ms frame size (" << (sampleRate_ / 100) << ")" << std::endl;
        // Always use 10ms frame size as required by NVIDIA models
        tempBuffer_.resize((sampleRate_ / 100) * channels_);
    }

    std::cout << "Final buffer size: " << tempBuffer_.size() << " samples" << std::endl;
    return true;
}

bool NvidiaAudioEffects::initializeDereverb() {
    // Create effect handle
    NvAFX_Handle effectHandle = nullptr;
    if (createEffect_(NVAFX_EFFECT_DEREVERB, &effectHandle) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to create dereverb effect" << std::endl;
        return false;
    }
    handle_ = effectHandle;
    
    // Set model path based on sample rate
    bool modelFound = false;
    std::string baseName;
    
    if (sampleRate_ == 48000) {
        baseName = "dereverb_48k.trtpkg";
    } else if (sampleRate_ == 16000) {
        baseName = "dereverb_16k.trtpkg";
    } else {
        std::cerr << "Unsupported sample rate for dereverb: " << sampleRate_ << std::endl;
        return false;
    }
    
    // Try several possible locations for the model file
    std::vector<std::string> possiblePaths = {
        "models\\" + baseName,                                 // Local models directory
        ".\\models\\" + baseName,                              // Explicit local path
        "..\\models\\" + baseName,                             // Up one level in models
        "external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // External directory
        "..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up one level 
        "..\\..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up two levels
        // Absolute paths for common installation locations
        "C:\\Program Files\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName,
        "C:\\Program Files (x86)\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName
    };
    
    for (const auto& path : possiblePaths) {
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fclose(f);
            modelPath_ = path;
            modelFound = true;
            std::cout << "Found model at: " << path << std::endl;
            break;
        } else {
            std::cout << "Tried path (not found): " << path << std::endl;
        }
    }
    
    if (!modelFound) {
        std::cerr << "Failed to find model file for dereverb effect" << std::endl;
        std::cerr << "Make sure model files are in the 'models' directory next to the executable" << std::endl;
        std::cerr << "Required file: " << baseName << std::endl;
        return false;
    }
    
    // Set parameters same as denoiser but for dereverb
    // (rest of the function unchanged)
    
    // Set common parameters
    if (setString_(handle_, NVAFX_PARAM_MODEL_PATH, modelPath_.c_str()) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set model path" << std::endl;
        return false;
    }
    
    if (setU32_(handle_, NVAFX_PARAM_INPUT_SAMPLE_RATE, sampleRate_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set input sample rate: " << sampleRate_ << std::endl;
        return false;
    }
    
    // Disable CUDA graph for better compatibility
    setU32_(handle_, NVAFX_PARAM_DISABLE_CUDA_GRAPH, 1);
    
    // Set intensity ratio
    setFloat_(handle_, NVAFX_PARAM_INTENSITY_RATIO, roomEchoRemovalLevel_);
    
    // Enable VAD if requested
    if (enableVAD_) {
        setU32_(handle_, NVAFX_PARAM_ENABLE_VAD, 1);
    }
    
    // Load the model
    if (load_(handle_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to load dereverb effect" << std::endl;
        return false;
    }
    
    // After loading, query the channel count
    unsigned int numInputChannels = 0;
    unsigned int numOutputChannels = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_CHANNELS, &numInputChannels) != NVAFX_STATUS_SUCCESS ||
        getU32_(handle_, NVAFX_PARAM_NUM_OUTPUT_CHANNELS, &numOutputChannels) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get channel count" << std::endl;
        return false;
    }
    
    // Make sure our channel count matches the effect's requirement
    if (channels_ != numInputChannels) {
        std::cout << "Note: Adjusting channel count from " << channels_ << " to " << numInputChannels << std::endl;
        channels_ = numInputChannels;
    }
    
    // Get buffer size
    unsigned int samplesPerFrame = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_SAMPLES_PER_FRAME, &samplesPerFrame) == NVAFX_STATUS_SUCCESS && 
        samplesPerFrame > 0) {
        tempBuffer_.resize(samplesPerFrame * channels_);
    } else {
        tempBuffer_.resize(480 * channels_);  // 10ms at 48kHz
    }
    
    return true;
}

bool NvidiaAudioEffects::initializeDenoiserDereverb() {
    // Create effect handle
    NvAFX_Handle effectHandle = nullptr;
    if (createEffect_(NVAFX_EFFECT_DEREVERB_DENOISER, &effectHandle) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to create dereverb+denoiser effect" << std::endl;
        return false;
    }
    handle_ = effectHandle;
    
    // Set model path based on sample rate
    bool modelFound = false;
    std::string baseName;
    
    if (sampleRate_ == 48000) {
        baseName = "dereverb_denoiser_48k.trtpkg";
    } else if (sampleRate_ == 16000) {
        baseName = "dereverb_denoiser_16k.trtpkg";
    } else {
        std::cerr << "Unsupported sample rate for dereverb+denoiser: " << sampleRate_ << std::endl;
        return false;
    }
    
    // Try several possible locations for the model file
    std::vector<std::string> possiblePaths = {
        "models\\" + baseName,                                 // Local models directory
        ".\\models\\" + baseName,                              // Explicit local path
        "..\\models\\" + baseName,                             // Up one level in models
        "external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // External directory
        "..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up one level 
        "..\\..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up two levels
        // Absolute paths for common installation locations
        "C:\\Program Files\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName,
        "C:\\Program Files (x86)\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName
    };
    
    for (const auto& path : possiblePaths) {
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fclose(f);
            modelPath_ = path;
            modelFound = true;
            std::cout << "Found model at: " << path << std::endl;
            break;
        } else {
            std::cout << "Tried path (not found): " << path << std::endl;
        }
    }
    
    if (!modelFound) {
        std::cerr << "Failed to find model file for dereverb+denoiser effect" << std::endl;
        std::cerr << "Make sure model files are in the 'models' directory next to the executable" << std::endl;
        std::cerr << "Required file: " << baseName << std::endl;
        return false;
    }
    
    // Set common parameters
    if (setString_(handle_, NVAFX_PARAM_MODEL_PATH, modelPath_.c_str()) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set model path" << std::endl;
        return false;
    }
    
    if (setU32_(handle_, NVAFX_PARAM_INPUT_SAMPLE_RATE, sampleRate_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set input sample rate: " << sampleRate_ << std::endl;
        return false;
    }
    
    // Disable CUDA graph for better compatibility
    setU32_(handle_, NVAFX_PARAM_DISABLE_CUDA_GRAPH, 1);
    
    // Set intensity ratio - use the average of both levels
    float combinedLevel = (noiseReductionLevel_ + roomEchoRemovalLevel_) / 2.0f;
    setFloat_(handle_, NVAFX_PARAM_INTENSITY_RATIO, combinedLevel);
    
    // Enable VAD if requested
    if (enableVAD_) {
        setU32_(handle_, NVAFX_PARAM_ENABLE_VAD, 1);
    }
    
    // Load the model
    if (load_(handle_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to load dereverb+denoiser effect" << std::endl;
        return false;
    }
    
    // After loading, query the channel count
    unsigned int numInputChannels = 0;
    unsigned int numOutputChannels = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_CHANNELS, &numInputChannels) != NVAFX_STATUS_SUCCESS ||
        getU32_(handle_, NVAFX_PARAM_NUM_OUTPUT_CHANNELS, &numOutputChannels) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get channel count" << std::endl;
        return false;
    }
    
    // Make sure our channel count matches the effect's requirement
    if (channels_ != numInputChannels) {
        std::cout << "Note: Adjusting channel count from " << channels_ << " to " << numInputChannels << std::endl;
        channels_ = numInputChannels;
    }
    
    // Get buffer size
    unsigned int samplesPerFrame = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_SAMPLES_PER_FRAME, &samplesPerFrame) == NVAFX_STATUS_SUCCESS && 
        samplesPerFrame > 0) {
        tempBuffer_.resize(samplesPerFrame * channels_);
    } else {
        tempBuffer_.resize(480 * channels_);  // 10ms at 48kHz
    }
    
    return true;
}

bool NvidiaAudioEffects::initializeSuperRes() {
    // Create effect handle
    NvAFX_Handle effectHandle = nullptr;
    if (createEffect_(NVAFX_EFFECT_SUPERRES, &effectHandle) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to create super-resolution effect" << std::endl;
        return false;
    }
    handle_ = effectHandle;
    
    // Set model path based on input and output sample rates
    bool modelFound = false;
    std::string baseName;
    
    // SuperRes has specific input->output sample rate combinations
    if (sampleRate_ == 8000 && superResOutputSampleRate_ == 16000) {
        baseName = "superres_8kto16k.trtpkg";
    } else if (sampleRate_ == 16000 && superResOutputSampleRate_ == 48000) {
        baseName = "superres_16kto48k.trtpkg";
    } else {
        std::cerr << "Unsupported sample rate conversion for super-resolution: " 
                  << sampleRate_ << " -> " << superResOutputSampleRate_ << std::endl;
        std::cerr << "Supported conversions: 8000->16000, 16000->48000" << std::endl;
        return false;
    }
    
    // Try several possible locations for the model file
    std::vector<std::string> possiblePaths = {
        "models\\" + baseName,                                 // Local models directory
        ".\\models\\" + baseName,                              // Explicit local path
        "..\\models\\" + baseName,                             // Up one level in models
        "external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // External directory
        "..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up one level 
        "..\\..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up two levels
        // Absolute paths for common installation locations
        "C:\\Program Files\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName,
        "C:\\Program Files (x86)\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName
    };
    
    for (const auto& path : possiblePaths) {
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fclose(f);
            modelPath_ = path;
            modelFound = true;
            std::cout << "Found model at: " << path << std::endl;
            break;
        } else {
            std::cout << "Tried path (not found): " << path << std::endl;
        }
    }
    
    if (!modelFound) {
        std::cerr << "Failed to find model file for super-resolution effect" << std::endl;
        std::cerr << "Make sure model files are in the 'models' directory next to the executable" << std::endl;
        std::cerr << "Required file: " << baseName << std::endl;
        return false;
    }
    
    // Set parameters
    if (setString_(handle_, NVAFX_PARAM_MODEL_PATH, modelPath_.c_str()) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set model path: " << modelPath_ << std::endl;
        return false;
    }
    
    if (setU32_(handle_, NVAFX_PARAM_INPUT_SAMPLE_RATE, sampleRate_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set input sample rate: " << sampleRate_ << std::endl;
        return false;
    }
    
    if (setU32_(handle_, NVAFX_PARAM_OUTPUT_SAMPLE_RATE, superResOutputSampleRate_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set output sample rate: " << superResOutputSampleRate_ << std::endl;
        return false;
    }
    
    // Disable CUDA graph for better compatibility
    setU32_(handle_, NVAFX_PARAM_DISABLE_CUDA_GRAPH, 1);
    
    // Load the model
    if (load_(handle_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to load super-resolution effect" << std::endl;
        return false;
    }
    
    // After loading, query the channel count
    unsigned int numInputChannels = 0;
    unsigned int numOutputChannels = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_CHANNELS, &numInputChannels) != NVAFX_STATUS_SUCCESS ||
        getU32_(handle_, NVAFX_PARAM_NUM_OUTPUT_CHANNELS, &numOutputChannels) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get channel count" << std::endl;
        return false;
    }
    
    // Make sure our channel count matches the effect's requirement
    if (channels_ != numInputChannels) {
        std::cout << "Note: Adjusting channel count from " << channels_ << " to " << numInputChannels << std::endl;
        channels_ = numInputChannels;
    }
    
    // Get input samples per frame
    unsigned int inputSamplesPerFrame = 0;
    unsigned int outputSamplesPerFrame = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_SAMPLES_PER_FRAME, &inputSamplesPerFrame) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get input samples per frame" << std::endl;
        return false;
    }
    
    if (getU32_(handle_, NVAFX_PARAM_NUM_OUTPUT_SAMPLES_PER_FRAME, &outputSamplesPerFrame) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get output samples per frame" << std::endl;
        return false;
    }
    
    // Resize buffer based on SDK requirements - use the larger of input/output for the temp buffer
    if (inputSamplesPerFrame > 0 && outputSamplesPerFrame > 0) {
        tempBuffer_.resize(std::max<unsigned int>(inputSamplesPerFrame, outputSamplesPerFrame) * channels_);
    } else {
        // Default size if we couldn't get the frame size
        tempBuffer_.resize(480 * channels_);  // 10ms at 48kHz
    }
    
    return true;
}

bool NvidiaAudioEffects::initializeAEC() {
    // Create effect handle
    NvAFX_Handle effectHandle = nullptr;
    if (createEffect_(NVAFX_EFFECT_AEC, &effectHandle) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to create acoustic echo cancellation effect" << std::endl;
        return false;
    }
    handle_ = effectHandle;
    
    // Set model path based on sample rate
    bool modelFound = false;
    std::string baseName;
    
    if (sampleRate_ == 48000) {
        baseName = "aec_48k.trtpkg";
    } else if (sampleRate_ == 16000) {
        baseName = "aec_16k.trtpkg";
    } else {
        std::cerr << "Unsupported sample rate for AEC: " << sampleRate_ << std::endl;
        std::cerr << "Supported sample rates: 16000, 48000" << std::endl;
        return false;
    }
    
    // Try several possible locations for the model file
    std::vector<std::string> possiblePaths = {
        "models\\" + baseName,                                 // Local models directory
        ".\\models\\" + baseName,                              // Explicit local path
        "..\\models\\" + baseName,                             // Up one level in models
        "external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // External directory
        "..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up one level 
        "..\\..\\external\\NVIDIA Audio Effects SDK\\models\\" + baseName, // Up two levels
        // Absolute paths for common installation locations
        "C:\\Program Files\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName,
        "C:\\Program Files (x86)\\NVIDIA Corporation\\NVIDIA Audio Effects SDK\\models\\" + baseName
    };
    
    for (const auto& path : possiblePaths) {
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fclose(f);
            modelPath_ = path;
            modelFound = true;
            std::cout << "Found model at: " << path << std::endl;
            break;
        } else {
            std::cout << "Tried path (not found): " << path << std::endl;
        }
    }
    
    if (!modelFound) {
        std::cerr << "Failed to find model file for acoustic echo cancellation effect" << std::endl;
        std::cerr << "Make sure model files are in the 'models' directory next to the executable" << std::endl;
        std::cerr << "Required file: " << baseName << std::endl;
        return false;
    }
    
    // Set common parameters
    if (setString_(handle_, NVAFX_PARAM_MODEL_PATH, modelPath_.c_str()) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set model path" << std::endl;
        return false;
    }
    
    if (setU32_(handle_, NVAFX_PARAM_INPUT_SAMPLE_RATE, sampleRate_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set input sample rate: " << sampleRate_ << std::endl;
        return false;
    }
    
    // Disable CUDA graph for better compatibility
    setU32_(handle_, NVAFX_PARAM_DISABLE_CUDA_GRAPH, 1);
    
    // Load the model
    if (load_(handle_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to load AEC effect" << std::endl;
        return false;
    }
    
    // After loading, query the channel count - AEC typically has 2 input channels (near-end + far-end)
    unsigned int numInputChannels = 0;
    unsigned int numOutputChannels = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_CHANNELS, &numInputChannels) != NVAFX_STATUS_SUCCESS ||
        getU32_(handle_, NVAFX_PARAM_NUM_OUTPUT_CHANNELS, &numOutputChannels) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get channel count" << std::endl;
        return false;
    }
    
    // AEC typically requires 2 input channels - adjust our internal state
    channels_ = numOutputChannels; // For our process function, we'll use the output channel count
    
    // Get buffer size
    unsigned int samplesPerFrame = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_SAMPLES_PER_FRAME, &samplesPerFrame) == NVAFX_STATUS_SUCCESS && 
        samplesPerFrame > 0) {
        tempBuffer_.resize(samplesPerFrame * numInputChannels); // Use numInputChannels as AEC has 2 input channels
    } else {
        tempBuffer_.resize(480 * numInputChannels);  // 10ms at 48kHz, use numInputChannels
    }
    
    return true;
}

bool NvidiaAudioEffects::initializeChainedEffect() {
    // Create effect handle
    if (!chainedEffectType_) {
        std::cerr << "No chained effect type specified" << std::endl;
        return false;
    }
    
    NvAFX_Handle effectHandle = nullptr;
    if (createChainedEffect_(chainedEffectType_, &effectHandle) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to create chained effect: " << chainedEffectType_ << std::endl;
        return false;
    }
    handle_ = effectHandle;
    
    // For chained effects, we need to set multiple models
    const char* modelFiles[2];
    
    // Determine model files based on chainedEffectType_
    if (strcmp(chainedEffectType_, NVAFX_CHAINED_EFFECT_DENOISER_16k_SUPERRES_16k_TO_48k) == 0) {
        modelFiles[0] = "external\\NVIDIA Audio Effects SDK\\models\\denoiser_16k.trtpkg";
        modelFiles[1] = "external\\NVIDIA Audio Effects SDK\\models\\superres_16kto48k.trtpkg";
    } 
    else if (strcmp(chainedEffectType_, NVAFX_CHAINED_EFFECT_DEREVERB_16k_SUPERRES_16k_TO_48k) == 0) {
        modelFiles[0] = "external\\NVIDIA Audio Effects SDK\\models\\dereverb_16k.trtpkg";
        modelFiles[1] = "external\\NVIDIA Audio Effects SDK\\models\\superres_16kto48k.trtpkg";
    }
    else if (strcmp(chainedEffectType_, NVAFX_CHAINED_EFFECT_DEREVERB_DENOISER_16k_SUPERRES_16k_TO_48k) == 0) {
        modelFiles[0] = "external\\NVIDIA Audio Effects SDK\\models\\dereverb_denoiser_16k.trtpkg";
        modelFiles[1] = "external\\NVIDIA Audio Effects SDK\\models\\superres_16kto48k.trtpkg";
    }
    else if (strcmp(chainedEffectType_, NVAFX_CHAINED_EFFECT_SUPERRES_8k_TO_16k_DENOISER_16k) == 0) {
        modelFiles[0] = "external\\NVIDIA Audio Effects SDK\\models\\superres_8kto16k.trtpkg";
        modelFiles[1] = "external\\NVIDIA Audio Effects SDK\\models\\denoiser_16k.trtpkg";
    }
    else if (strcmp(chainedEffectType_, NVAFX_CHAINED_EFFECT_SUPERRES_8k_TO_16k_DEREVERB_16k) == 0) {
        modelFiles[0] = "external\\NVIDIA Audio Effects SDK\\models\\superres_8kto16k.trtpkg";
        modelFiles[1] = "external\\NVIDIA Audio Effects SDK\\models\\dereverb_16k.trtpkg";
    }
    else if (strcmp(chainedEffectType_, NVAFX_CHAINED_EFFECT_SUPERRES_8k_TO_16k_DEREVERB_DENOISER_16k) == 0) {
        modelFiles[0] = "external\\NVIDIA Audio Effects SDK\\models\\superres_8kto16k.trtpkg";
        modelFiles[1] = "external\\NVIDIA Audio Effects SDK\\models\\dereverb_denoiser_16k.trtpkg";
    }
    else {
        std::cerr << "Unknown chained effect type: " << chainedEffectType_ << std::endl;
        return false;
    }
    
    // Set the model files
    if (setStringList_(handle_, NVAFX_PARAM_MODEL_PATH, modelFiles, 2) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to set chained effect model paths" << std::endl;
        return false;
    }
    
    // Disable CUDA graph for better compatibility
    setU32_(handle_, NVAFX_PARAM_DISABLE_CUDA_GRAPH, 1);
    
    // Set intensity ratios for the chained effects
    // For simplicity, use the same intensity for both effects in the chain
    float intensityRatios[2] = { noiseReductionLevel_, roomEchoRemovalLevel_ };
    
    // Load the model
    if (load_(handle_) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to load chained effect" << std::endl;
        return false;
    }
    
    // After loading, query the channel count
    unsigned int numInputChannels = 0;
    unsigned int numOutputChannels = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_CHANNELS, &numInputChannels) != NVAFX_STATUS_SUCCESS ||
        getU32_(handle_, NVAFX_PARAM_NUM_OUTPUT_CHANNELS, &numOutputChannels) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get channel count" << std::endl;
        return false;
    }
    
    // Make sure our channel count matches the effect's requirement
    if (channels_ != numInputChannels) {
        std::cout << "Note: Adjusting channel count from " << channels_ << " to " << numInputChannels << std::endl;
        channels_ = numInputChannels;
    }
    
    // Get input and output samples per frame
    unsigned int inputSamplesPerFrame = 0;
    unsigned int outputSamplesPerFrame = 0;
    if (getU32_(handle_, NVAFX_PARAM_NUM_INPUT_SAMPLES_PER_FRAME, &inputSamplesPerFrame) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get input samples per frame" << std::endl;
        return false;
    }
    
    if (getU32_(handle_, NVAFX_PARAM_NUM_OUTPUT_SAMPLES_PER_FRAME, &outputSamplesPerFrame) != NVAFX_STATUS_SUCCESS) {
        std::cerr << "Failed to get output samples per frame" << std::endl;
        return false;
    }
    
    // For chained effects with super-resolution, the output buffer will be larger
    tempBuffer_.resize(std::max<unsigned int>(inputSamplesPerFrame, outputSamplesPerFrame) * channels_);
    
    return true;
}

bool NvidiaAudioEffects::processAudio(float* buffer, size_t frames) {
    // If not initialized or no SDK available, pass through the audio
    if (!initialized_ || !handle_) {
        return true;
    }

    // Calculate the expected frame size based on sample rate (10ms)
    const size_t expectedFrameSize = sampleRate_ / 100; // 10ms frame size
    
    // We should process in 10ms chunks as required by NVIDIA models
    std::cout << "Processing audio: " << frames << " frames, expected frame size: " << expectedFrameSize 
              << ", channels: " << channels_ << std::endl;
    
    size_t remaining = frames;
    size_t offset = 0;
    
    // Create arrays of pointers for input and output channels
    // NVIDIA SDK expects deinterleaved audio (separate buffer for each channel)
    std::vector<std::vector<float>> channelBuffers(channels_, std::vector<float>(expectedFrameSize, 0.0f));
    std::vector<const float*> inputPtrs(channels_);
    std::vector<float*> outputPtrs(channels_);
    
    // Set up pointers to channel buffers
    for (int ch = 0; ch < channels_; ch++) {
        inputPtrs[ch] = channelBuffers[ch].data();
        outputPtrs[ch] = channelBuffers[ch].data();
    }

    while (remaining > 0) {
        // Determine how many samples to process
        // NVIDIA models require 10ms frames for optimal processing
        // For partial frames, either pad to full frame or process what we have
        size_t toProcess = std::min<size_t>(expectedFrameSize, remaining);
        
        // Clear the buffers
        for (int ch = 0; ch < channels_; ch++) {
            std::fill(channelBuffers[ch].begin(), channelBuffers[ch].end(), 0.0f);
        }
        
        // Deinterleave the audio data (convert from interleaved to planar)
        for (size_t i = 0; i < toProcess; i++) {
            for (int ch = 0; ch < channels_; ch++) {
                channelBuffers[ch][i] = buffer[(offset + i) * channels_ + ch];
            }
        }
        
        // Process audio - always use the expected frame size
        // For partial frames, we've zero-padded the buffer
        std::cout << "  Calling run_ with " << expectedFrameSize << " samples (actual data: " << toProcess << ")" << std::endl;
        
        int result = run_(handle_, inputPtrs.data(), outputPtrs.data(), 
                        static_cast<unsigned int>(expectedFrameSize), 
                        static_cast<unsigned int>(channels_));
                          
        if (result != NVAFX_STATUS_SUCCESS) {
            std::cerr << "Failed to process audio, error code: " << result << std::endl;
            
            // Return true to allow passthrough mode even if processing fails
            // This ensures audio still works even if effects fail
            break;
        }
        
        // Interleave the processed audio data back to the original buffer
        // Only copy back the actual number of samples we had
        for (size_t i = 0; i < toProcess; i++) {
            for (int ch = 0; ch < channels_; ch++) {
                buffer[(offset + i) * channels_ + ch] = channelBuffers[ch][i];
            }
        }
        
        remaining -= toProcess;
        offset += toProcess;
    }
    
    return true; // Always return true to allow audio passthrough
}

void NvidiaAudioEffects::cleanup() {
    if (handle_) {
        destroyEffect_(handle_);
        handle_ = nullptr;
    }
    
    if (sdkModule_) {
        FreeLibrary((HMODULE)sdkModule_);
        sdkModule_ = nullptr;
    }
    
    initialized_ = false;
} 