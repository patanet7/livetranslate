/**
 * Shared Audio Processing Configuration
 * 
 * This module provides consistent audio processing parameters and settings
 * across all pages in the LiveTranslate application.
 */

// Global audio processing parameters
window.audioProcessingParams = {
    // Voice Activity Detection
    vad: {
        enabled: true,
        aggressiveness: 2,
        energyThreshold: 0.01,
        speechDuration: 0.5,
        silenceDuration: 0.5,
        voiceFreqMin: 85,
        voiceFreqMax: 300
    },
    
    // Voice Frequency Filtering
    voiceFilter: {
        enabled: true,
        fundamentalMin: 85,
        fundamentalMax: 300,
        formantEnhancement: 0.3,
        sibilanceMin: 4000,
        sibilanceMax: 8000,
        voiceBandGain: 1.2
    },
    
    // Noise Reduction
    noiseReduction: {
        enabled: true,
        strength: 0.5,
        voiceProtection: 0.3,
        spectralSubtraction: 0.4,
        adaptiveGating: true,
        noiseFloorDb: -40
    },
    
    // Voice Enhancement
    voiceEnhancement: {
        enabled: true,
        compressorThreshold: -12,
        compressorRatio: 3,
        compressorKnee: 2,
        clarityEnhancement: 0.4,
        deEsserFreq: 6000,
        deEsserStrength: 0.3,
        dynamicRange: 0.7
    },
    
    // Audio Normalization
    normalization: {
        enabled: true,
        targetLufs: -16,
        peakLimit: -1,
        truePeakLimit: -1.5,
        gateThreshold: -70
    },
    
    // Input Analysis
    inputAnalysis: {
        enabled: true,
        rmsAnalysis: true,
        peakAnalysis: true,
        clippingDetection: true,
        snrCalculation: true,
        frequencyAnalysis: true
    },
    
    // Output Processing
    outputProcessing: {
        enabled: true,
        finalLimiter: true,
        dithering: false,
        bitDepth: 16,
        sampleRate: 16000
    }
};

// Audio processing stage definitions
window.audioProcessingStages = [
    {
        id: 'input',
        name: 'Input Analysis',
        description: 'Analyze input audio characteristics',
        icon: '🎤',
        enabled: true,
        params: ['rmsAnalysis', 'peakAnalysis', 'clippingDetection', 'snrCalculation'],
        metrics: ['RMS Level', 'Peak Level', 'Clipping Count', 'SNR Ratio']
    },
    {
        id: 'vad',
        name: 'Voice Activity Detection',
        description: 'Detect speech segments and voice activity',
        icon: '🗣️',
        enabled: true,
        params: ['aggressiveness', 'energyThreshold', 'speechDuration'],
        metrics: ['Voice Activity', 'Confidence', 'Speech Segments', 'Speech Ratio']
    },
    {
        id: 'voiceFilter',
        name: 'Voice Frequency Filtering',
        description: 'Filter and enhance voice frequencies',
        icon: '🎵',
        enabled: true,
        params: ['fundamentalMin', 'fundamentalMax', 'formantEnhancement'],
        metrics: ['Frequency Range', 'Formant Enhancement', 'Sibilance', 'Voice Band Gain']
    },
    {
        id: 'noiseReduction',
        name: 'Noise Reduction',
        description: 'Remove background noise while preserving speech',
        icon: '🔇',
        enabled: true,
        params: ['strength', 'voiceProtection', 'spectralSubtraction'],
        metrics: ['Noise Floor', 'Reduction Level', 'Voice Protection', 'Artifacts']
    },
    {
        id: 'voiceEnhancement',
        name: 'Voice Enhancement',
        description: 'Enhance speech clarity and presence',
        icon: '✨',
        enabled: true,
        params: ['compressorThreshold', 'compressorRatio', 'clarityEnhancement'],
        metrics: ['Compression Ratio', 'Clarity Enhancement', 'Dynamic Range', 'Presence']
    },
    {
        id: 'normalization',
        name: 'Audio Normalization',
        description: 'Normalize audio levels for consistent output',
        icon: '📏',
        enabled: true,
        params: ['targetLufs', 'peakLimit', 'truePeakLimit'],
        metrics: ['LUFS Level', 'Peak Level', 'True Peak', 'Dynamic Range']
    },
    {
        id: 'outputProcessing',
        name: 'Output Processing',
        description: 'Final processing and format conversion',
        icon: '📤',
        enabled: true,
        params: ['finalLimiter', 'bitDepth', 'sampleRate'],
        metrics: ['Final Level', 'Bit Depth', 'Sample Rate', 'Processing Time']
    }
];

// Preset configurations
window.audioProcessingPresets = {
    speech: {
        name: 'Speech Optimization',
        description: 'Optimized for clear speech transcription',
        params: {
            vad: { aggressiveness: 2, energyThreshold: 0.01 },
            voiceFilter: { enabled: true, fundamentalMin: 85, fundamentalMax: 300 },
            noiseReduction: { enabled: true, strength: 0.6, voiceProtection: 0.4 },
            voiceEnhancement: { enabled: true, compressorRatio: 3, clarityEnhancement: 0.5 },
            normalization: { enabled: true, targetLufs: -16 }
        }
    },
    
    podcast: {
        name: 'Podcast Quality',
        description: 'Balanced processing for podcast-style audio',
        params: {
            vad: { aggressiveness: 1, energyThreshold: 0.005 },
            voiceFilter: { enabled: true, formantEnhancement: 0.4 },
            noiseReduction: { enabled: true, strength: 0.4, voiceProtection: 0.5 },
            voiceEnhancement: { enabled: true, compressorRatio: 2.5, clarityEnhancement: 0.3 },
            normalization: { enabled: true, targetLufs: -18 }
        }
    },
    
    noisy: {
        name: 'Noisy Environment',
        description: 'Aggressive processing for noisy environments',
        params: {
            vad: { aggressiveness: 3, energyThreshold: 0.02 },
            voiceFilter: { enabled: true, voiceBandGain: 1.5 },
            noiseReduction: { enabled: true, strength: 0.8, voiceProtection: 0.2 },
            voiceEnhancement: { enabled: true, compressorRatio: 4, clarityEnhancement: 0.6 },
            normalization: { enabled: true, targetLufs: -14 }
        }
    },
    
    clean: {
        name: 'Clean Audio',
        description: 'Minimal processing for clean audio sources',
        params: {
            vad: { aggressiveness: 1, energyThreshold: 0.005 },
            voiceFilter: { enabled: true, formantEnhancement: 0.2 },
            noiseReduction: { enabled: true, strength: 0.2, voiceProtection: 0.7 },
            voiceEnhancement: { enabled: true, compressorRatio: 2, clarityEnhancement: 0.2 },
            normalization: { enabled: true, targetLufs: -18 }
        }
    },
    
    music: {
        name: 'Music Vocal',
        description: 'Optimized for vocal extraction from music',
        params: {
            vad: { aggressiveness: 2, energyThreshold: 0.015 },
            voiceFilter: { enabled: true, sibilanceMin: 5000, sibilanceMax: 10000 },
            noiseReduction: { enabled: true, strength: 0.7, voiceProtection: 0.3 },
            voiceEnhancement: { enabled: true, compressorRatio: 3.5, clarityEnhancement: 0.5 },
            normalization: { enabled: true, targetLufs: -16 }
        }
    },
    
    broadcast: {
        name: 'Broadcast Quality',
        description: 'Professional broadcast processing',
        params: {
            vad: { aggressiveness: 2, energyThreshold: 0.008 },
            voiceFilter: { enabled: true, formantEnhancement: 0.3 },
            noiseReduction: { enabled: true, strength: 0.5, voiceProtection: 0.4 },
            voiceEnhancement: { enabled: true, compressorRatio: 2.5, clarityEnhancement: 0.4 },
            normalization: { enabled: true, targetLufs: -20, peakLimit: -2 }
        }
    }
};

// Audio processing controls interface
window.audioProcessingControls = {
    // Get current parameter value
    getParameter: function(stage, param) {
        return window.audioProcessingParams[stage] ? window.audioProcessingParams[stage][param] : null;
    },
    
    // Set parameter value
    setParameter: function(stage, param, value) {
        if (window.audioProcessingParams[stage]) {
            window.audioProcessingParams[stage][param] = value;
            this.saveSettings();
            this.notifyParameterChange(stage, param, value);
        }
    },
    
    // Toggle stage enabled/disabled
    toggleStage: function(stageId) {
        const stage = window.audioProcessingStages.find(s => s.id === stageId);
        if (stage) {
            stage.enabled = !stage.enabled;
            this.saveSettings();
            this.notifyStageToggle(stageId, stage.enabled);
        }
    },
    
    // Load preset configuration
    loadPreset: function(presetName) {
        const preset = window.audioProcessingPresets[presetName];
        if (preset) {
            Object.keys(preset.params).forEach(stage => {
                Object.keys(preset.params[stage]).forEach(param => {
                    if (window.audioProcessingParams[stage]) {
                        window.audioProcessingParams[stage][param] = preset.params[stage][param];
                    }
                });
            });
            this.saveSettings();
            this.notifyPresetLoad(presetName);
        }
    },
    
    // Get enabled stages
    getEnabledStages: function() {
        return window.audioProcessingStages.filter(stage => stage.enabled);
    },
    
    // Save settings to localStorage
    saveSettings: function() {
        try {
            localStorage.setItem('livetranslate-audio-processing-params', JSON.stringify(window.audioProcessingParams));
            localStorage.setItem('livetranslate-audio-processing-stages', JSON.stringify(window.audioProcessingStages));
        } catch (error) {
            console.error('Failed to save audio processing settings:', error);
        }
    },
    
    // Load settings from localStorage
    loadSettings: function() {
        try {
            const savedParams = localStorage.getItem('livetranslate-audio-processing-params');
            const savedStages = localStorage.getItem('livetranslate-audio-processing-stages');
            
            if (savedParams) {
                const params = JSON.parse(savedParams);
                Object.assign(window.audioProcessingParams, params);
            }
            
            if (savedStages) {
                const stages = JSON.parse(savedStages);
                stages.forEach(savedStage => {
                    const stage = window.audioProcessingStages.find(s => s.id === savedStage.id);
                    if (stage) {
                        stage.enabled = savedStage.enabled;
                    }
                });
            }
        } catch (error) {
            console.error('Failed to load audio processing settings:', error);
        }
    },
    
    // Reset to defaults
    resetToDefaults: function() {
        localStorage.removeItem('livetranslate-audio-processing-params');
        localStorage.removeItem('livetranslate-audio-processing-stages');
        location.reload();
    },
    
    // Export configuration
    exportConfiguration: function() {
        const config = {
            params: window.audioProcessingParams,
            stages: window.audioProcessingStages,
            timestamp: new Date().toISOString(),
            version: '1.0.0'
        };
        
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `livetranslate-audio-config-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    },
    
    // Import configuration
    importConfiguration: function(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const config = JSON.parse(e.target.result);
                    
                    if (config.params) {
                        Object.assign(window.audioProcessingParams, config.params);
                    }
                    
                    if (config.stages) {
                        config.stages.forEach(importedStage => {
                            const stage = window.audioProcessingStages.find(s => s.id === importedStage.id);
                            if (stage) {
                                stage.enabled = importedStage.enabled;
                            }
                        });
                    }
                    
                    this.saveSettings();
                    resolve(config);
                } catch (error) {
                    reject(error);
                }
            }.bind(this);
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    },
    
    // Event notification methods
    notifyParameterChange: function(stage, param, value) {
        window.dispatchEvent(new CustomEvent('audioParameterChange', {
            detail: { stage, param, value }
        }));
    },
    
    notifyStageToggle: function(stageId, enabled) {
        window.dispatchEvent(new CustomEvent('audioStageToggle', {
            detail: { stageId, enabled }
        }));
    },
    
    notifyPresetLoad: function(presetName) {
        window.dispatchEvent(new CustomEvent('audioPresetLoad', {
            detail: { presetName }
        }));
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    window.audioProcessingControls.loadSettings();
    
    // Dispatch ready event
    window.dispatchEvent(new CustomEvent('audioProcessingReady'));
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        audioProcessingParams: window.audioProcessingParams,
        audioProcessingStages: window.audioProcessingStages,
        audioProcessingPresets: window.audioProcessingPresets,
        audioProcessingControls: window.audioProcessingControls
    };
}