#pragma once

// Stub nvAudioEffects.h file
#include "../external/NVIDIA Audio Effects SDK/version.h"

// Define necessary types for NVIDIA Audio Effects SDK
typedef void* NvAudioEffectsSession;
typedef void* NvAudioEffectsDenoiserHandle;

// Error codes
typedef enum {
    NVAE_OK = 0,
    NVAE_INVALID_PARAMETER,
    NVAE_NOT_INITIALIZED,
    NVAE_INVALID_STATE,
    NVAE_STATUS_SUCCESS = NVAE_OK
} NvAudioEffectsStatus;

// Create parameters
typedef struct {
    int sampleRate;
    int numChannels;
    int maxBatchSize;
    bool realTime;
    const char* modelPath;
    int logLevel;
    int numChunkSamples;
    int gpuId;
    bool useAudioFrame;
} NvAudioEffectsCreateParam;

// Denoiser create parameters
typedef struct {
    float intensity;
    bool attenuateInputNoise;
    bool detectVAD;
    float vadThreshold;
    int minVADDuration;
    float intensityLevel;
    int numChunkSamples;
} NvAudioEffectsDenoiserCreateParam;

// Process parameters
typedef struct {
    const float* input;
    float* output;
    int numSamples;
    bool isFirstBuffer;
    bool isLastBuffer;
} NvAudioEffectsDenoiserProcessParam;

// Stub function prototypes
#ifdef __cplusplus
extern "C" {
#endif

NvAudioEffectsStatus NvAudioEffectsCreateSession(NvAudioEffectsCreateParam* createParam, NvAudioEffectsSession* session);
NvAudioEffectsStatus NvAudioEffectsDestroySession(NvAudioEffectsSession session);
NvAudioEffectsStatus NvAudioEffectsSessionSetup(NvAudioEffectsSession session, const NvAudioEffectsCreateParam* createParam);
NvAudioEffectsStatus NvAudioEffectsCreateDenoiser(NvAudioEffectsSession session, NvAudioEffectsDenoiserCreateParam* createParam, NvAudioEffectsDenoiserHandle* handle);
NvAudioEffectsStatus NvAudioEffectsDestroyDenoiser(NvAudioEffectsDenoiserHandle handle);
NvAudioEffectsStatus NvAudioEffectsDenoiserCreate(NvAudioEffectsSession session, NvAudioEffectsDenoiserCreateParam* createParam, NvAudioEffectsDenoiserHandle* handle);
NvAudioEffectsStatus NvAudioEffectsDenoiserProcess(NvAudioEffectsDenoiserHandle handle, NvAudioEffectsDenoiserProcessParam* processParam);
NvAudioEffectsStatus NvAudioEffectsDenoiserRun(NvAudioEffectsDenoiserHandle handle, float* inBuffer, float* outBuffer, int frames);
NvAudioEffectsStatus NvAudioEffectsDenoiserDestroy(NvAudioEffectsDenoiserHandle handle);

#ifdef __cplusplus
}
#endif 