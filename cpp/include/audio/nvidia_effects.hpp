#pragma once

#include <memory>
#include <vector>
#include <string>

// Forward declarations based on NVIDIA Audio Effects SDK API
typedef void* NvAFX_Handle;
typedef const char* NvAFX_EffectSelector;
typedef const char* NvAFX_ParameterSelector;

// Common selectors for API
#define NVAFX_EFFECT_DENOISER "denoiser"
#define NVAFX_EFFECT_DEREVERB "dereverb"
#define NVAFX_EFFECT_DEREVERB_DENOISER "dereverb_denoiser"
#define NVAFX_EFFECT_SUPERRES "superres"
#define NVAFX_EFFECT_AEC "aec"
#define NVAFX_CHAINED_EFFECT_DENOISER_16k_SUPERRES_16k_TO_48k "denoiser_16k_superres_16k_to_48k"
#define NVAFX_CHAINED_EFFECT_DEREVERB_16k_SUPERRES_16k_TO_48k "dereverb_16k_superres_16k_to_48k"
#define NVAFX_CHAINED_EFFECT_DEREVERB_DENOISER_16k_SUPERRES_16k_TO_48k "dereverb_denoiser_16k_superres_16k_to_48k"
#define NVAFX_CHAINED_EFFECT_SUPERRES_8k_TO_16k_DENOISER_16k "superres_8k_to_16k_denoiser_16k"
#define NVAFX_CHAINED_EFFECT_SUPERRES_8k_TO_16k_DEREVERB_16k "superres_8k_to_16k_dereverb_16k"
#define NVAFX_CHAINED_EFFECT_SUPERRES_8k_TO_16k_DEREVERB_DENOISER_16k "superres_8k_to_16k_dereverb_denoiser_16k"

#define NVAFX_PARAM_MODEL_PATH "model_path"
#define NVAFX_PARAM_INTENSITY_RATIO "intensity_ratio"
// Current parameter selectors
#define NVAFX_PARAM_INPUT_SAMPLE_RATE "input_sample_rate"
#define NVAFX_PARAM_OUTPUT_SAMPLE_RATE "output_sample_rate"
#define NVAFX_PARAM_NUM_INPUT_CHANNELS "num_input_channels"
#define NVAFX_PARAM_NUM_OUTPUT_CHANNELS "num_output_channels"
#define NVAFX_PARAM_NUM_INPUT_SAMPLES_PER_FRAME "num_input_samples_per_frame"
#define NVAFX_PARAM_NUM_OUTPUT_SAMPLES_PER_FRAME "num_output_samples_per_frame"
#define NVAFX_PARAM_DISABLE_CUDA_GRAPH "disable_cuda_graph"
#define NVAFX_PARAM_ENABLE_VAD "enable_vad"

// NVAFX status codes
#define NVAFX_STATUS_SUCCESS 0
#define NVAFX_STATUS_FAILED 1

// Effect types
enum class NvidiaEffectType {
    None,
    Denoiser,
    Dereverb,
    DereverbDenoiser,
    SuperRes,
    AEC,
    ChainedEffect
};

class NvidiaAudioEffects {
public:
    NvidiaAudioEffects();
    ~NvidiaAudioEffects();

    bool initialize(int sampleRate, int channels);
    bool processAudio(float* buffer, size_t frames);
    void cleanup();
    
    // Effect selection
    void setEffectType(NvidiaEffectType type) { effectType_ = type; }

    // Common parameters
    void setNoiseReductionLevel(float level) { noiseReductionLevel_ = level; }
    void setRoomEchoRemovalLevel(float level) { roomEchoRemovalLevel_ = level; }
    void setEnableVAD(bool enable) { enableVAD_ = enable; }

    // Super-Resolution parameters
    void setSuperResInputSampleRate(int rate) { superResInputSampleRate_ = rate; }
    void setSuperResOutputSampleRate(int rate) { superResOutputSampleRate_ = rate; }
    
    // Chained effects
    void setChainedEffectType(const char* type) { chainedEffectType_ = type; }
    
    // Check if the SDK is available
    static bool isAvailable();

    // Get the effect type as a string
    std::string getNvidiaEffectTypeString(NvidiaEffectType effectType) const;

private:
    NvAFX_Handle handle_;
    bool initialized_;
    std::vector<float> tempBuffer_;
    NvidiaEffectType effectType_;
    std::string modelPath_;
    
    // Effect parameters
    float noiseReductionLevel_;
    float roomEchoRemovalLevel_;
    bool enableVAD_;
    int sampleRate_;
    int channels_;
    int superResInputSampleRate_;
    int superResOutputSampleRate_;
    const char* chainedEffectType_; // For chained effect types
    
    // Internal method to load the SDK DLL
    bool loadSDK();
    
    // Initialization helpers for different effects
    bool initializeDenoiser();
    bool initializeDereverb();
    bool initializeDenoiserDereverb();
    bool initializeSuperRes();
    bool initializeAEC();
    bool initializeChainedEffect();
    
    // SDK function pointers
    typedef int(*CreateEffectFunc)(NvAFX_EffectSelector, NvAFX_Handle*);
    typedef int(*CreateChainedEffectFunc)(NvAFX_EffectSelector, NvAFX_Handle*);
    typedef int(*DestroyEffectFunc)(NvAFX_Handle);
    typedef int(*SetStringFunc)(NvAFX_Handle, NvAFX_ParameterSelector, const char*);
    typedef int(*SetStringListFunc)(NvAFX_Handle, NvAFX_ParameterSelector, const char**, unsigned int);
    typedef int(*SetFloatFunc)(NvAFX_Handle, NvAFX_ParameterSelector, float);
    typedef int(*SetU32Func)(NvAFX_Handle, NvAFX_ParameterSelector, unsigned int);
    typedef int(*GetU32Func)(NvAFX_Handle, NvAFX_ParameterSelector, unsigned int*);
    typedef int(*LoadFunc)(NvAFX_Handle);
    typedef int(*RunFunc)(NvAFX_Handle, const float**, float**, unsigned, unsigned);
    typedef int(*ResetFunc)(NvAFX_Handle);
    
    void* sdkModule_;
    CreateEffectFunc createEffect_;
    CreateChainedEffectFunc createChainedEffect_;
    DestroyEffectFunc destroyEffect_;
    SetStringFunc setString_;
    SetStringListFunc setStringList_;
    SetFloatFunc setFloat_;
    SetU32Func setU32_;
    GetU32Func getU32_;
    LoadFunc load_;
    RunFunc run_;
    ResetFunc reset_;
}; 