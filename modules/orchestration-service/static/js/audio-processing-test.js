// Enhanced Audio Processing Pipeline Test with Step-by-Step Control

// Audio processing parameters with defaults
const audioProcessingParams = {
    // Voice Activity Detection
    vad: {
        enabled: true,
        aggressiveness: 2,  // 0-3, higher = more aggressive
        frameDuration: 20,  // ms (10, 20, or 30)
        minSpeechDuration: 250,  // ms
        maxSilenceDuration: 500,  // ms
        energyThreshold: 0.01,  // 0-1
        voiceFreqMin: 85,  // Hz - human voice fundamental frequency range
        voiceFreqMax: 300  // Hz
    },
    
    // Voice-Specific Filtering
    voiceFilter: {
        enabled: true,
        // Human voice frequency bands
        fundamentalMin: 85,  // Hz - adult male fundamental
        fundamentalMax: 300,  // Hz - child fundamental
        // Formant frequencies (important for speech clarity)
        formant1Min: 200,  // Hz - first formant range
        formant1Max: 1000,  // Hz
        formant2Min: 500,  // Hz - second formant range
        formant2Max: 3000,  // Hz
        formant3Min: 1500,  // Hz - third formant range
        formant3Max: 4000,  // Hz
        // High frequency preservation for consonants
        sibilanceMin: 4000,  // Hz - 's', 'sh' sounds
        sibilanceMax: 8000,  // Hz
        sibilanceBoost: 1.2,  // Boost factor for clarity
        // Voice band emphasis
        voiceBandGain: 1.1,  // Gain for voice frequencies
        preserveFormants: true,
        adaptiveFiltering: true
    },
    
    // Noise Reduction with Voice Preservation
    noiseReduction: {
        enabled: true,
        strength: 0.7,  // 0-1 (reduced from 0.8 to preserve voice)
        smoothing: 0.95,  // 0-1
        noiseFloor: 0.01,  // 0-1
        gateThreshold: 0.003,  // 0-1 (lowered to preserve quiet speech)
        // Spectral subtraction
        spectralSubtraction: true,
        spectralFloor: 0.002,  // Minimum spectral magnitude
        overSubtraction: 2.0,  // Over-subtraction factor
        // Voice preservation
        voiceProtection: true,
        voiceThreshold: 0.1,  // Threshold for voice detection
        musicNoiseSuppression: 0.95  // Suppress musical noise artifacts
    },
    
    // Voice Enhancement (not just audio enhancement)
    voiceEnhancement: {
        enabled: true,
        // Gentle normalization that preserves voice dynamics
        gain: 1.0,  // 0.5-2.0
        normalize: false,  // Disabled by default to preserve natural voice
        targetLevel: -18,  // dB (raised for more natural sound)
        targetPeak: 0.85,  // 0-1 (reduced to prevent harsh limiting)
        // Voice-specific compression
        compressor: {
            enabled: true,
            threshold: -20,  // dB (raised to be less aggressive)
            ratio: 3,  // 1-20 (reduced for gentler compression)
            attack: 0.005,  // seconds (slower for natural speech)
            release: 0.3,  // seconds
            knee: 2.0,  // dB - soft knee for smoother compression
            makeupGain: 1.2  // Makeup gain after compression
        },
        // Voice clarity enhancement
        clarity: {
            enabled: true,
            amount: 0.3,  // 0-1
            focusFreq: 2500,  // Hz - frequency to enhance for clarity
            bandwidth: 1000  // Hz - bandwidth of enhancement
        },
        // De-esser for harsh sibilants
        deEsser: {
            enabled: true,
            threshold: -20,  // dB
            frequency: 6000,  // Hz
            bandwidth: 2000,  // Hz
            ratio: 3  // Compression ratio for sibilants
        }
    },
    
    // Advanced Voice Processing
    voiceProcessing: {
        enabled: true,
        // Pitch preservation
        preservePitch: true,
        pitchShiftCents: 0,  // -100 to +100 cents
        // Format correction
        formantShift: 0,  // -12 to +12 semitones
        formantPreservation: 1.0,  // 0-1
        // Voice isolation
        voiceIsolation: {
            enabled: false,  // Can be too aggressive
            strength: 0.5,  // 0-1
            algorithm: 'harmonic'  // 'harmonic' or 'percussive'
        },
        // Dynamic range
        expanderGate: {
            enabled: true,
            threshold: -40,  // dB
            ratio: 2,  // 1-10
            attack: 0.01,  // seconds
            release: 0.1  // seconds
        }
    },
    
    // Silence Detection with Voice Awareness
    silence: {
        enabled: true,
        threshold: -45,  // dB (raised to preserve quiet speech)
        minDuration: 200,  // ms (reduced to keep short utterances)
        trimStart: true,
        trimEnd: true,
        padDuration: 150,  // ms padding to keep
        // Voice-aware silence detection
        useVoiceActivity: true,
        breathThreshold: -35,  // dB - threshold for breath sounds
        keepBreaths: false  // Keep breath sounds between words
    },
    
    // Resampling with quality options
    resampling: {
        enabled: true,
        targetRate: 16000,  // Hz
        quality: 'high',  // 'low', 'medium', 'high'
        // Anti-aliasing filter
        antiAlias: true,
        filterOrder: 128,  // Higher = better quality, more CPU
        // Dithering for bit depth conversion
        dithering: true,
        ditherType: 'triangular'  // 'none', 'rectangular', 'triangular'
    }
};

// Processing stages with pause capability
const processingStages = [
    {
        id: 'original',
        name: 'Original Audio',
        description: 'Raw recorded audio without any processing',
        process: async (audioData) => audioData,
        canPause: false
    },
    {
        id: 'decoded',
        name: 'Decoded Audio',
        description: 'Decoded to raw audio buffer',
        process: async (audioData) => audioData,
        canPause: true
    },
    {
        id: 'voiceFilter',
        name: 'Voice Frequency Filter',
        description: 'Filter and enhance human voice frequencies',
        process: async (audioData) => applyVoiceFilter(audioData),
        canPause: true
    },
    {
        id: 'vad',
        name: 'Voice Activity Detection',
        description: 'Detect and extract speech segments',
        process: async (audioData) => applyVAD(audioData),
        canPause: true
    },
    {
        id: 'noise',
        name: 'Voice-Aware Noise Reduction',
        description: 'Remove noise while preserving voice',
        process: async (audioData) => applyNoiseReduction(audioData),
        canPause: true
    },
    {
        id: 'voiceEnhance',
        name: 'Voice Enhancement',
        description: 'Enhance voice clarity and presence',
        process: async (audioData) => applyVoiceEnhancement(audioData),
        canPause: true
    },
    {
        id: 'voiceProcess',
        name: 'Advanced Voice Processing',
        description: 'Apply voice-specific processing',
        process: async (audioData) => applyVoiceProcessing(audioData),
        canPause: true
    },
    {
        id: 'silence',
        name: 'Voice-Aware Silence Trimming',
        description: 'Remove silence while preserving speech',
        process: async (audioData) => applySilenceTrimming(audioData),
        canPause: true
    },
    {
        id: 'resample',
        name: 'High-Quality Resampling',
        description: 'Convert to 16kHz with anti-aliasing',
        process: async (audioData) => applyResampling(audioData),
        canPause: true
    },
    {
        id: 'final',
        name: 'Final Output',
        description: 'Processed audio ready for transcription',
        process: async (audioData) => audioData,
        canPause: false
    }
];

// State for processing pipeline
let processingState = {
    currentStage: 0,
    isPaused: false,
    stageResults: {},
    audioContext: null,
    audioTester: null
};

// Enhanced AudioProcessingTester class
class AudioProcessingTester {
    constructor() {
        this.originalAudio = null;
        this.processedStages = {};
        this.audioContext = null;
        this.currentStage = null;
    }

    async initialize() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log('[AudioTester] Initialized with sample rate:', this.audioContext.sampleRate);
    }

    // Convert blob to audio buffer
    async blobToAudioBuffer(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        return await this.audioContext.decodeAudioData(arrayBuffer);
    }

    // Convert audio buffer to wav blob
    audioBufferToWav(audioBuffer) {
        const length = audioBuffer.length * audioBuffer.numberOfChannels * 2;
        const buffer = new ArrayBuffer(44 + length);
        const view = new DataView(buffer);
        const channels = [];
        let offset = 0;
        let pos = 0;

        // Write WAV header
        const setUint16 = (data) => {
            view.setUint16(pos, data, true);
            pos += 2;
        };
        const setUint32 = (data) => {
            view.setUint32(pos, data, true);
            pos += 4;
        };

        // RIFF identifier
        setUint32(0x46464952); // "RIFF"
        setUint32(36 + length); // file length - 8
        setUint32(0x45564157); // "WAVE"

        // fmt sub-chunk
        setUint32(0x20746d66); // "fmt "
        setUint32(16); // subchunk size
        setUint16(1); // PCM format
        setUint16(audioBuffer.numberOfChannels);
        setUint32(audioBuffer.sampleRate);
        setUint32(audioBuffer.sampleRate * 2 * audioBuffer.numberOfChannels); // byte rate
        setUint16(audioBuffer.numberOfChannels * 2); // block align
        setUint16(16); // bits per sample

        // data sub-chunk
        setUint32(0x61746164); // "data"
        setUint32(length);

        // Write interleaved PCM samples
        for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
            channels.push(audioBuffer.getChannelData(i));
        }

        pos = 44;
        for (let i = 0; i < audioBuffer.length; i++) {
            for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, channels[channel][i]));
                view.setInt16(pos, sample * 0x7FFF, true);
                pos += 2;
            }
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    // Resample audio to target sample rate
    async resampleAudio(audioBuffer, targetSampleRate) {
        if (audioBuffer.sampleRate === targetSampleRate) {
            return audioBuffer;
        }

        const offlineContext = new OfflineAudioContext(
            audioBuffer.numberOfChannels,
            audioBuffer.duration * targetSampleRate,
            targetSampleRate
        );

        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start(0);

        return await offlineContext.startRendering();
    }

    // Convert to mono
    convertToMono(audioBuffer) {
        if (audioBuffer.numberOfChannels === 1) {
            return audioBuffer;
        }

        const monoBuffer = this.audioContext.createBuffer(
            1,
            audioBuffer.length,
            audioBuffer.sampleRate
        );

        const monoChannel = monoBuffer.getChannelData(0);
        
        // Average all channels
        for (let i = 0; i < audioBuffer.length; i++) {
            let sum = 0;
            for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
                sum += audioBuffer.getChannelData(channel)[i];
            }
            monoChannel[i] = sum / audioBuffer.numberOfChannels;
        }

        return monoBuffer;
    }

    // Apply Voice Activity Detection
    applyVAD(audioBuffer) {
        if (!audioProcessingParams.vad.enabled) return audioBuffer;
        
        const channelData = audioBuffer.getChannelData(0);
        const windowSize = Math.floor(audioBuffer.sampleRate * audioProcessingParams.vad.frameDuration / 1000);
        const threshold = audioProcessingParams.vad.energyThreshold;
        
        // Analyze energy in windows
        const energyWindows = [];
        for (let i = 0; i < channelData.length - windowSize; i += windowSize) {
            let energy = 0;
            for (let j = 0; j < windowSize; j++) {
                energy += Math.abs(channelData[i + j]);
            }
            energyWindows.push({
                start: i,
                end: i + windowSize,
                energy: energy / windowSize
            });
        }
        
        // Find speech segments
        const speechSegments = [];
        let inSpeech = false;
        let speechStart = 0;
        
        for (let i = 0; i < energyWindows.length; i++) {
            const window = energyWindows[i];
            
            if (!inSpeech && window.energy > threshold) {
                inSpeech = true;
                speechStart = window.start;
            } else if (inSpeech && window.energy < threshold) {
                // Check if silence is long enough
                let silenceDuration = 0;
                for (let j = i; j < energyWindows.length && energyWindows[j].energy < threshold; j++) {
                    silenceDuration += windowSize;
                }
                
                if (silenceDuration > audioProcessingParams.vad.maxSilenceDuration * audioBuffer.sampleRate / 1000) {
                    // End of speech segment
                    const duration = (window.start - speechStart) / audioBuffer.sampleRate * 1000;
                    if (duration >= audioProcessingParams.vad.minSpeechDuration) {
                        speechSegments.push({
                            start: speechStart,
                            end: window.start
                        });
                    }
                    inSpeech = false;
                }
            }
        }
        
        // Handle case where speech continues to end
        if (inSpeech) {
            speechSegments.push({
                start: speechStart,
                end: channelData.length
            });
        }
        
        // Create new buffer with only speech segments
        let totalLength = 0;
        for (const segment of speechSegments) {
            totalLength += segment.end - segment.start;
        }
        
        if (totalLength === 0) return audioBuffer; // No speech detected
        
        const vadBuffer = this.audioContext.createBuffer(
            audioBuffer.numberOfChannels,
            totalLength,
            audioBuffer.sampleRate
        );
        
        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const sourceData = audioBuffer.getChannelData(channel);
            const vadData = vadBuffer.getChannelData(channel);
            
            let offset = 0;
            for (const segment of speechSegments) {
                for (let i = segment.start; i < segment.end; i++) {
                    vadData[offset++] = sourceData[i];
                }
            }
        }
        
        return vadBuffer;
    }

    // Apply voice frequency filtering
    applyVoiceFilter(audioBuffer) {
        if (!audioProcessingParams.voiceFilter.enabled) return audioBuffer;
        
        const filteredBuffer = this.audioContext.createBuffer(
            audioBuffer.numberOfChannels,
            audioBuffer.length,
            audioBuffer.sampleRate
        );
        
        // For now, return a copy - real implementation would use FFT-based filtering
        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const sourceData = audioBuffer.getChannelData(channel);
            const filteredData = filteredBuffer.getChannelData(channel);
            
            // Simple high-pass filter to remove low-frequency rumble
            let prev = 0;
            const cutoff = audioProcessingParams.voiceFilter.fundamentalMin / audioBuffer.sampleRate;
            const rc = 1.0 / (2.0 * Math.PI * cutoff);
            const dt = 1.0 / audioBuffer.sampleRate;
            const alpha = rc / (rc + dt);
            
            for (let i = 0; i < sourceData.length; i++) {
                filteredData[i] = alpha * (prev + sourceData[i] - (sourceData[i-1] || 0));
                prev = filteredData[i];
                
                // Apply voice band gain
                filteredData[i] *= audioProcessingParams.voiceFilter.voiceBandGain;
            }
        }
        
        return filteredBuffer;
    }
    
    // Apply enhanced noise reduction with voice preservation
    applyNoiseReduction(audioBuffer) {
        if (!audioProcessingParams.noiseReduction.enabled) return audioBuffer;
        
        const denoisedBuffer = this.audioContext.createBuffer(
            audioBuffer.numberOfChannels,
            audioBuffer.length,
            audioBuffer.sampleRate
        );

        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            const denoisedData = denoisedBuffer.getChannelData(channel);
            
            // Apply noise gate with voice detection
            const gateThreshold = audioProcessingParams.noiseReduction.gateThreshold;
            const voiceThreshold = audioProcessingParams.noiseReduction.voiceThreshold;
            
            // Apply smoothing filter
            const smoothing = audioProcessingParams.noiseReduction.smoothing;
            let smoothedValue = 0;
            let envelope = 0;
            
            for (let i = 0; i < channelData.length; i++) {
                // Update envelope follower
                const rectified = Math.abs(channelData[i]);
                envelope = Math.max(rectified, envelope * 0.999);
                
                // Smooth the signal
                smoothedValue = smoothedValue * smoothing + channelData[i] * (1 - smoothing);
                
                // Voice detection based on envelope
                const isVoice = envelope > voiceThreshold;
                
                // Apply adaptive noise gate
                const adaptiveThreshold = isVoice ? gateThreshold * 0.5 : gateThreshold;
                
                if (Math.abs(smoothedValue) < adaptiveThreshold && !isVoice) {
                    denoisedData[i] = 0;
                } else {
                    // Apply noise reduction with voice protection
                    const strength = isVoice ? 
                        audioProcessingParams.noiseReduction.strength * 0.7 : 
                        audioProcessingParams.noiseReduction.strength;
                    
                    denoisedData[i] = channelData[i] * (1 - strength) + smoothedValue * strength;
                    
                    // Preserve transients
                    if (Math.abs(channelData[i] - smoothedValue) > 0.1) {
                        denoisedData[i] = channelData[i] * 0.7 + denoisedData[i] * 0.3;
                    }
                }
            }
        }

        return denoisedBuffer;
    }
    
    // Apply voice-specific enhancement
    applyVoiceEnhancement(audioBuffer) {
        if (!audioProcessingParams.voiceEnhancement.enabled) return audioBuffer;
        
        const enhancedBuffer = this.audioContext.createBuffer(
            audioBuffer.numberOfChannels,
            audioBuffer.length,
            audioBuffer.sampleRate
        );

        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            const enhancedData = enhancedBuffer.getChannelData(channel);
            
            // Apply gentle gain
            const gain = audioProcessingParams.voiceEnhancement.gain;
            
            // Find peak for optional normalization
            let peak = 0;
            for (let i = 0; i < channelData.length; i++) {
                peak = Math.max(peak, Math.abs(channelData[i] * gain));
            }
            
            // Calculate normalization factor (only if enabled)
            const targetPeak = audioProcessingParams.voiceEnhancement.targetPeak;
            const normFactor = (audioProcessingParams.voiceEnhancement.normalize && peak > targetPeak) ? 
                targetPeak / peak : 1;
            
            // Apply voice-specific compression
            const compressor = audioProcessingParams.voiceEnhancement.compressor;
            const threshold = Math.pow(10, compressor.threshold / 20);
            const ratio = compressor.ratio;
            const knee = compressor.knee;
            const makeupGain = compressor.makeupGain;
            
            // Slower attack/release for natural speech
            const attack = Math.exp(-1 / (compressor.attack * audioBuffer.sampleRate));
            const release = Math.exp(-1 / (compressor.release * audioBuffer.sampleRate));
            
            let envelope = 0;
            
            for (let i = 0; i < channelData.length; i++) {
                let sample = channelData[i] * gain * normFactor;
                
                if (compressor.enabled) {
                    const absSample = Math.abs(sample);
                    
                    // Update envelope with asymmetric attack/release
                    const targetEnv = absSample;
                    const rate = targetEnv > envelope ? attack : release;
                    envelope = targetEnv + (envelope - targetEnv) * rate;
                    
                    // Soft knee compression
                    if (envelope > threshold - knee/2) {
                        let compressionRatio = 1;
                        
                        if (envelope > threshold + knee/2) {
                            // Full compression
                            compressionRatio = ratio;
                        } else {
                            // Soft knee region
                            const kneePosition = (envelope - (threshold - knee/2)) / knee;
                            compressionRatio = 1 + (ratio - 1) * kneePosition * kneePosition;
                        }
                        
                        const reduction = threshold + (envelope - threshold) / compressionRatio;
                        sample *= (reduction / envelope) * makeupGain;
                    }
                }
                
                // Voice clarity enhancement
                if (audioProcessingParams.voiceEnhancement.clarity.enabled) {
                    // Simple presence boost around 2.5kHz
                    // In real implementation, this would be frequency-domain processing
                    const clarityAmount = audioProcessingParams.voiceEnhancement.clarity.amount;
                    if (i > 0 && i < channelData.length - 1) {
                        const highFreq = channelData[i] - 0.5 * (channelData[i-1] + channelData[i+1]);
                        sample += highFreq * clarityAmount;
                    }
                }
                
                enhancedData[i] = Math.max(-1, Math.min(1, sample));
            }
        }

        return enhancedBuffer;
    }
    
    // Apply advanced voice processing
    applyVoiceProcessing(audioBuffer) {
        if (!audioProcessingParams.voiceProcessing.enabled) return audioBuffer;
        
        const processedBuffer = this.audioContext.createBuffer(
            audioBuffer.numberOfChannels,
            audioBuffer.length,
            audioBuffer.sampleRate
        );
        
        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            const processedData = processedBuffer.getChannelData(channel);
            
            // Apply expander gate for dynamic range
            const expander = audioProcessingParams.voiceProcessing.expanderGate;
            if (expander.enabled) {
                const threshold = Math.pow(10, expander.threshold / 20);
                const ratio = expander.ratio;
                const attack = Math.exp(-1 / (expander.attack * audioBuffer.sampleRate));
                const release = Math.exp(-1 / (expander.release * audioBuffer.sampleRate));
                
                let envelope = 0;
                
                for (let i = 0; i < channelData.length; i++) {
                    const absSample = Math.abs(channelData[i]);
                    
                    // Update envelope
                    const rate = absSample > envelope ? attack : release;
                    envelope = absSample + (envelope - absSample) * rate;
                    
                    // Apply expansion below threshold
                    if (envelope < threshold && envelope > 0) {
                        const expansion = Math.pow(envelope / threshold, 1 / ratio) * threshold;
                        processedData[i] = channelData[i] * (expansion / envelope);
                    } else {
                        processedData[i] = channelData[i];
                    }
                }
            } else {
                // Copy data if expander is disabled
                processedData.set(channelData);
            }
        }
        
        return processedBuffer;
    }

    // Apply audio enhancement with compression
    applyEnhancement(audioBuffer) {
        if (!audioProcessingParams.enhancement.enabled) return audioBuffer;
        
        const enhancedBuffer = this.audioContext.createBuffer(
            audioBuffer.numberOfChannels,
            audioBuffer.length,
            audioBuffer.sampleRate
        );

        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            const enhancedData = enhancedBuffer.getChannelData(channel);
            
            // Apply gain
            const gain = audioProcessingParams.enhancement.gain;
            
            // Find peak for normalization
            let peak = 0;
            for (let i = 0; i < channelData.length; i++) {
                peak = Math.max(peak, Math.abs(channelData[i] * gain));
            }
            
            // Calculate normalization factor
            const targetPeak = audioProcessingParams.enhancement.targetPeak;
            const normFactor = audioProcessingParams.enhancement.normalize && peak > 0 ? targetPeak / peak : 1;
            
            // Apply compression if enabled
            const compressor = audioProcessingParams.enhancement.compressor;
            const threshold = Math.pow(10, compressor.threshold / 20);
            const ratio = compressor.ratio;
            const attack = Math.exp(-1 / (compressor.attack * audioBuffer.sampleRate));
            const release = Math.exp(-1 / (compressor.release * audioBuffer.sampleRate));
            
            let envelope = 0;
            
            for (let i = 0; i < channelData.length; i++) {
                let sample = channelData[i] * gain * normFactor;
                
                if (compressor.enabled) {
                    const absSample = Math.abs(sample);
                    
                    // Update envelope
                    const targetEnv = absSample;
                    const rate = targetEnv > envelope ? attack : release;
                    envelope = targetEnv + (envelope - targetEnv) * rate;
                    
                    // Apply compression
                    if (envelope > threshold) {
                        const reduction = threshold + (envelope - threshold) / ratio;
                        sample *= reduction / envelope;
                    }
                }
                
                enhancedData[i] = Math.max(-1, Math.min(1, sample));
            }
        }

        return enhancedBuffer;
    }

    // Apply silence trimming
    applySilenceTrimming(audioBuffer) {
        if (!audioProcessingParams.silence.enabled) return audioBuffer;
        
        const channelData = audioBuffer.getChannelData(0);
        const windowSize = Math.floor(audioBuffer.sampleRate * 0.02); // 20ms windows
        const threshold = Math.pow(10, audioProcessingParams.silence.threshold / 20);
        const padSamples = Math.floor(audioProcessingParams.silence.padDuration * audioBuffer.sampleRate / 1000);
        
        // Find speech boundaries
        let startIndex = 0;
        let endIndex = channelData.length;
        
        if (audioProcessingParams.silence.trimStart) {
            // Find start of speech
            for (let i = 0; i < channelData.length - windowSize; i += windowSize) {
                let energy = 0;
                for (let j = 0; j < windowSize; j++) {
                    energy += Math.abs(channelData[i + j]);
                }
                if (energy / windowSize > threshold) {
                    startIndex = Math.max(0, i - padSamples);
                    break;
                }
            }
        }
        
        if (audioProcessingParams.silence.trimEnd) {
            // Find end of speech
            for (let i = channelData.length - windowSize; i >= 0; i -= windowSize) {
                let energy = 0;
                for (let j = 0; j < windowSize; j++) {
                    energy += Math.abs(channelData[i + j]);
                }
                if (energy / windowSize > threshold) {
                    endIndex = Math.min(channelData.length, i + windowSize + padSamples);
                    break;
                }
            }
        }
        
        // Create trimmed buffer
        const trimmedLength = endIndex - startIndex;
        const trimmedBuffer = this.audioContext.createBuffer(
            audioBuffer.numberOfChannels,
            trimmedLength,
            audioBuffer.sampleRate
        );
        
        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const sourceData = audioBuffer.getChannelData(channel);
            const trimmedData = trimmedBuffer.getChannelData(channel);
            for (let i = 0; i < trimmedLength; i++) {
                trimmedData[i] = sourceData[startIndex + i];
            }
        }
        
        return trimmedBuffer;
    }

    // Play a specific stage
    async playStage(stageName) {
        const stage = this.processedStages[stageName];
        if (!stage) {
            console.error('[AudioTester] Stage not found:', stageName);
            return;
        }

        try {
            let audioUrl;
            if (stage.blob) {
                audioUrl = URL.createObjectURL(stage.blob);
            } else if (stage.audioBuffer) {
                const blob = this.audioBufferToWav(stage.audioBuffer);
                audioUrl = URL.createObjectURL(blob);
            } else {
                console.error('[AudioTester] No audio data for stage:', stageName);
                return;
            }

            const audio = new Audio(audioUrl);
            audio.onended = () => URL.revokeObjectURL(audioUrl);
            await audio.play();
            
            this.currentStage = { name: stageName, audio };
            
        } catch (error) {
            console.error('[AudioTester] Playback error:', error);
        }
    }

    // Analyze audio characteristics
    analyzeAudio(audioBuffer) {
        const channelData = audioBuffer.getChannelData(0);
        
        // Calculate RMS
        let sum = 0;
        for (let i = 0; i < channelData.length; i++) {
            sum += channelData[i] * channelData[i];
        }
        const rms = Math.sqrt(sum / channelData.length);
        
        // Calculate peak
        let peak = 0;
        for (let i = 0; i < channelData.length; i++) {
            peak = Math.max(peak, Math.abs(channelData[i]));
        }
        
        // Calculate dynamic range
        const dynamicRange = peak > 0 ? 20 * Math.log10(peak / rms) : 0;
        
        // Detect clipping
        let clippedSamples = 0;
        for (let i = 0; i < channelData.length; i++) {
            if (Math.abs(channelData[i]) >= 0.99) {
                clippedSamples++;
            }
        }
        
        return {
            rms: rms.toFixed(4),
            peak: peak.toFixed(4),
            rmsDb: rms > 0 ? (20 * Math.log10(rms)).toFixed(2) + ' dB' : '-∞ dB',
            peakDb: peak > 0 ? (20 * Math.log10(peak)).toFixed(2) + ' dB' : '-∞ dB',
            dynamicRange: dynamicRange.toFixed(2) + ' dB',
            clipping: clippedSamples > 0 ? `${clippedSamples} samples (${(clippedSamples/channelData.length*100).toFixed(2)}%)` : 'None',
            duration: audioBuffer.duration.toFixed(2) + 's',
            sampleRate: audioBuffer.sampleRate + ' Hz',
            channels: audioBuffer.numberOfChannels,
            samples: channelData.length.toLocaleString()
        };
    }

    // Export stage as file
    async exportStage(stageName) {
        const stage = this.processedStages[stageName];
        if (!stage) return;

        let blob;
        if (stage.blob) {
            blob = stage.blob;
        } else if (stage.audioBuffer) {
            blob = this.audioBufferToWav(stage.audioBuffer);
        } else {
            return;
        }

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `audio_${stageName}_${Date.now()}.wav`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // Send to whisper for testing
    async sendToWhisper(stageName, modelName) {
        const stage = this.processedStages[stageName];
        if (!stage) return null;

        let blob;
        if (stage.blob) {
            blob = stage.blob;
        } else if (stage.audioBuffer) {
            blob = this.audioBufferToWav(stage.audioBuffer);
        } else {
            return null;
        }

        const formData = new FormData();
        formData.append('audio', blob, 'test_audio.wav');
        
        const url = testConfig.whisperApiUrl ? 
            `${testConfig.whisperApiUrl}/transcribe/${modelName}` :
            `/api/whisper/transcribe/${modelName}`;
            
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        return await response.json();
    }
}

// Create parameter controls UI
function createParameterControls() {
    const container = document.createElement('div');
    container.className = 'parameter-controls';
    container.innerHTML = `
        <h3>Audio Processing Parameters</h3>
        
        <!-- VAD Parameters -->
        <div class="param-section">
            <h4>Voice Activity Detection</h4>
            <label>
                <input type="checkbox" id="vad-enabled" ${audioProcessingParams.vad.enabled ? 'checked' : ''}>
                Enable VAD
            </label>
            <div class="param-grid">
                <div class="param-item">
                    <label>Aggressiveness (0-3)</label>
                    <input type="range" id="vad-aggressiveness" min="0" max="3" step="1" 
                           value="${audioProcessingParams.vad.aggressiveness}">
                    <span id="vad-aggressiveness-value">${audioProcessingParams.vad.aggressiveness}</span>
                </div>
                <div class="param-item">
                    <label>Energy Threshold</label>
                    <input type="range" id="vad-energy-threshold" min="0" max="0.1" step="0.001"
                           value="${audioProcessingParams.vad.energyThreshold}">
                    <span id="vad-energy-threshold-value">${audioProcessingParams.vad.energyThreshold.toFixed(3)}</span>
                </div>
                <div class="param-item">
                    <label>Min Speech Duration (ms)</label>
                    <input type="number" id="vad-min-speech" min="50" max="1000" step="50"
                           value="${audioProcessingParams.vad.minSpeechDuration}">
                </div>
                <div class="param-item">
                    <label>Max Silence Duration (ms)</label>
                    <input type="number" id="vad-max-silence" min="100" max="2000" step="100"
                           value="${audioProcessingParams.vad.maxSilenceDuration}">
                </div>
            </div>
        </div>
        
        <!-- Noise Reduction Parameters -->
        <div class="param-section">
            <h4>Noise Reduction</h4>
            <label>
                <input type="checkbox" id="noise-enabled" ${audioProcessingParams.noiseReduction.enabled ? 'checked' : ''}>
                Enable Noise Reduction
            </label>
            <div class="param-grid">
                <div class="param-item">
                    <label>Strength (0-1)</label>
                    <input type="range" id="noise-strength" min="0" max="1" step="0.1"
                           value="${audioProcessingParams.noiseReduction.strength}">
                    <span id="noise-strength-value">${audioProcessingParams.noiseReduction.strength}</span>
                </div>
                <div class="param-item">
                    <label>Smoothing (0-1)</label>
                    <input type="range" id="noise-smoothing" min="0" max="1" step="0.05"
                           value="${audioProcessingParams.noiseReduction.smoothing}">
                    <span id="noise-smoothing-value">${audioProcessingParams.noiseReduction.smoothing}</span>
                </div>
                <div class="param-item">
                    <label>Gate Threshold</label>
                    <input type="range" id="noise-gate-threshold" min="0" max="0.05" step="0.001"
                           value="${audioProcessingParams.noiseReduction.gateThreshold}">
                    <span id="noise-gate-threshold-value">${audioProcessingParams.noiseReduction.gateThreshold.toFixed(3)}</span>
                </div>
            </div>
        </div>
        
        <!-- Enhancement Parameters -->
        <div class="param-section">
            <h4>Audio Enhancement</h4>
            <label>
                <input type="checkbox" id="enhance-enabled" ${audioProcessingParams.enhancement.enabled ? 'checked' : ''}>
                Enable Enhancement
            </label>
            <div class="param-grid">
                <div class="param-item">
                    <label>Gain (0.5-2.0)</label>
                    <input type="range" id="enhance-gain" min="0.5" max="2" step="0.1"
                           value="${audioProcessingParams.enhancement.gain}">
                    <span id="enhance-gain-value">${audioProcessingParams.enhancement.gain}</span>
                </div>
                <div class="param-item">
                    <label>Target Peak (0-1)</label>
                    <input type="range" id="enhance-target-peak" min="0.5" max="1" step="0.05"
                           value="${audioProcessingParams.enhancement.targetPeak}">
                    <span id="enhance-target-peak-value">${audioProcessingParams.enhancement.targetPeak}</span>
                </div>
                <div class="param-item">
                    <label>
                        <input type="checkbox" id="enhance-normalize" ${audioProcessingParams.enhancement.normalize ? 'checked' : ''}>
                        Normalize
                    </label>
                </div>
            </div>
        </div>
        
        <!-- Compressor Parameters -->
        <div class="param-subsection">
            <h5>Compressor</h5>
            <label>
                <input type="checkbox" id="compressor-enabled" ${audioProcessingParams.enhancement.compressor.enabled ? 'checked' : ''}>
                Enable Compressor
            </label>
            <div class="param-grid">
                <div class="param-item">
                    <label>Threshold (dB)</label>
                    <input type="number" id="compressor-threshold" min="-60" max="0" step="1"
                           value="${audioProcessingParams.enhancement.compressor.threshold}">
                </div>
                <div class="param-item">
                    <label>Ratio (1-20)</label>
                    <input type="range" id="compressor-ratio" min="1" max="20" step="1"
                           value="${audioProcessingParams.enhancement.compressor.ratio}">
                    <span id="compressor-ratio-value">${audioProcessingParams.enhancement.compressor.ratio}</span>
                </div>
                <div class="param-item">
                    <label>Attack (s)</label>
                    <input type="number" id="compressor-attack" min="0.001" max="0.1" step="0.001"
                           value="${audioProcessingParams.enhancement.compressor.attack}">
                </div>
                <div class="param-item">
                    <label>Release (s)</label>
                    <input type="number" id="compressor-release" min="0.01" max="1" step="0.01"
                           value="${audioProcessingParams.enhancement.compressor.release}">
                </div>
            </div>
        </div>
        
        <!-- Silence Detection Parameters -->
        <div class="param-section">
            <h4>Silence Detection</h4>
            <label>
                <input type="checkbox" id="silence-enabled" ${audioProcessingParams.silence.enabled ? 'checked' : ''}>
                Enable Silence Detection
            </label>
            <div class="param-grid">
                <div class="param-item">
                    <label>Threshold (dB)</label>
                    <input type="range" id="silence-threshold" min="-80" max="-20" step="1"
                           value="${audioProcessingParams.silence.threshold}">
                    <span id="silence-threshold-value">${audioProcessingParams.silence.threshold}</span>
                </div>
                <div class="param-item">
                    <label>Min Duration (ms)</label>
                    <input type="number" id="silence-duration" min="100" max="1000" step="50"
                           value="${audioProcessingParams.silence.minDuration}">
                </div>
                <div class="param-item">
                    <label>Padding (ms)</label>
                    <input type="number" id="silence-padding" min="0" max="500" step="50"
                           value="${audioProcessingParams.silence.padDuration}">
                </div>
                <div class="param-item">
                    <label>
                        <input type="checkbox" id="silence-trim-start" ${audioProcessingParams.silence.trimStart ? 'checked' : ''}>
                        Trim Start
                    </label>
                </div>
                <div class="param-item">
                    <label>
                        <input type="checkbox" id="silence-trim-end" ${audioProcessingParams.silence.trimEnd ? 'checked' : ''}>
                        Trim End
                    </label>
                </div>
            </div>
        </div>
        
        <!-- Resampling Parameters -->
        <div class="param-section">
            <h4>Resampling</h4>
            <label>
                <input type="checkbox" id="resample-enabled" ${audioProcessingParams.resampling.enabled ? 'checked' : ''}>
                Enable Resampling
            </label>
            <div class="param-grid">
                <div class="param-item">
                    <label>Target Rate (Hz)</label>
                    <select id="resample-rate">
                        <option value="8000">8000 Hz</option>
                        <option value="16000" ${audioProcessingParams.resampling.targetRate === 16000 ? 'selected' : ''}>16000 Hz</option>
                        <option value="22050">22050 Hz</option>
                        <option value="44100">44100 Hz</option>
                        <option value="48000">48000 Hz</option>
                    </select>
                </div>
                <div class="param-item">
                    <label>Quality</label>
                    <select id="resample-quality">
                        <option value="low">Low (Fast)</option>
                        <option value="medium">Medium</option>
                        <option value="high" ${audioProcessingParams.resampling.quality === 'high' ? 'selected' : ''}>High (Slow)</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="param-controls">
            <button onclick="resetParameters()" class="button secondary">Reset to Defaults</button>
            <button onclick="saveParameters()" class="button primary">Save Parameters</button>
        </div>
    `;
    
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        .parameter-controls {
            background: var(--background);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .param-section {
            margin-bottom: 1.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        
        .param-section:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        
        .param-subsection {
            margin-left: 1rem;
            margin-top: 1rem;
        }
        
        .param-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 0.5rem;
        }
        
        .param-item {
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }
        
        .param-item label {
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .param-item input[type="range"] {
            width: 100%;
        }
        
        .param-item input[type="number"] {
            padding: 0.3rem;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: var(--input-bg);
            color: var(--text);
        }
        
        .param-controls {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
            justify-content: flex-end;
        }
        
        .stage-paused {
            border-color: var(--warning) !important;
            background: rgba(255, 193, 7, 0.1) !important;
        }
        
        .pause-overlay {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--panel-bg);
            border: 2px solid var(--primary);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            z-index: 1000;
            text-align: center;
        }
        
        .pause-overlay h3 {
            margin-bottom: 1rem;
        }
        
        .pause-overlay .button-group {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 1.5rem;
        }
    `;
    document.head.appendChild(style);
    
    return container;
}

// Update parameters from UI
function updateParameters() {
    // VAD
    audioProcessingParams.vad.enabled = document.getElementById('vad-enabled').checked;
    audioProcessingParams.vad.aggressiveness = parseInt(document.getElementById('vad-aggressiveness').value);
    audioProcessingParams.vad.energyThreshold = parseFloat(document.getElementById('vad-energy-threshold').value);
    audioProcessingParams.vad.minSpeechDuration = parseInt(document.getElementById('vad-min-speech').value);
    audioProcessingParams.vad.maxSilenceDuration = parseInt(document.getElementById('vad-max-silence').value);
    
    // Noise Reduction
    audioProcessingParams.noiseReduction.enabled = document.getElementById('noise-enabled').checked;
    audioProcessingParams.noiseReduction.strength = parseFloat(document.getElementById('noise-strength').value);
    audioProcessingParams.noiseReduction.smoothing = parseFloat(document.getElementById('noise-smoothing').value);
    audioProcessingParams.noiseReduction.gateThreshold = parseFloat(document.getElementById('noise-gate-threshold').value);
    
    // Enhancement
    audioProcessingParams.enhancement.enabled = document.getElementById('enhance-enabled').checked;
    audioProcessingParams.enhancement.gain = parseFloat(document.getElementById('enhance-gain').value);
    audioProcessingParams.enhancement.targetPeak = parseFloat(document.getElementById('enhance-target-peak').value);
    audioProcessingParams.enhancement.normalize = document.getElementById('enhance-normalize').checked;
    
    // Compressor
    audioProcessingParams.enhancement.compressor.enabled = document.getElementById('compressor-enabled').checked;
    audioProcessingParams.enhancement.compressor.threshold = parseInt(document.getElementById('compressor-threshold').value);
    audioProcessingParams.enhancement.compressor.ratio = parseInt(document.getElementById('compressor-ratio').value);
    audioProcessingParams.enhancement.compressor.attack = parseFloat(document.getElementById('compressor-attack').value);
    audioProcessingParams.enhancement.compressor.release = parseFloat(document.getElementById('compressor-release').value);
    
    // Silence
    audioProcessingParams.silence.enabled = document.getElementById('silence-enabled').checked;
    audioProcessingParams.silence.threshold = parseInt(document.getElementById('silence-threshold').value);
    audioProcessingParams.silence.minDuration = parseInt(document.getElementById('silence-duration').value);
    audioProcessingParams.silence.padDuration = parseInt(document.getElementById('silence-padding').value);
    audioProcessingParams.silence.trimStart = document.getElementById('silence-trim-start').checked;
    audioProcessingParams.silence.trimEnd = document.getElementById('silence-trim-end').checked;
    
    // Resampling
    audioProcessingParams.resampling.enabled = document.getElementById('resample-enabled').checked;
    audioProcessingParams.resampling.targetRate = parseInt(document.getElementById('resample-rate').value);
    audioProcessingParams.resampling.quality = document.getElementById('resample-quality').value;
}

// Process audio through pipeline with pause capability
async function processAudioPipeline() {
    if (!testState.recordedBlob) {
        addTestLog('WARNING', 'No recording to process');
        return;
    }
    
    // Initialize audio tester if needed
    if (!processingState.audioTester) {
        processingState.audioTester = new AudioProcessingTester();
        await processingState.audioTester.initialize();
    }
    
    // Update parameters from UI
    updateParameters();
    
    try {
        // Convert blob to audio buffer
        const audioBuffer = await processingState.audioTester.blobToAudioBuffer(testState.recordedBlob);
        
        // Reset state
        processingState.currentStage = 0;
        processingState.isPaused = false;
        processingState.stageResults = {};
        processingState.audioTester.processedStages = {};
        
        // Clear previous results
        document.getElementById('processingStages').innerHTML = '';
        document.getElementById('processingStages').style.display = 'block';
        document.getElementById('audioAnalysis').style.display = 'block';
        
        // Insert parameter controls if not already present
        if (!document.querySelector('.parameter-controls')) {
            const paramControls = createParameterControls();
            document.getElementById('processingStages').before(paramControls);
            
            // Set up event listeners for real-time updates
            setupParameterListeners();
        }
        
        // Store original
        processingState.audioTester.processedStages['original'] = {
            blob: testState.recordedBlob,
            audioBuffer: audioBuffer,
            description: 'Original recorded audio'
        };
        
        // Start processing
        await processNextStage(audioBuffer);
        
    } catch (error) {
        addTestLog('ERROR', `Pipeline processing failed: ${error.message}`);
    }
}

// Process next stage in pipeline
async function processNextStage(audioData) {
    if (processingState.currentStage >= processingStages.length) {
        addTestLog('SUCCESS', 'Audio processing pipeline completed');
        updateProcessingStatus('Pipeline completed successfully');
        return;
    }
    
    const stage = processingStages[processingState.currentStage];
    updateProcessingStatus(`Processing: ${stage.name}...`);
    
    try {
        let processedAudio = audioData;
        
        // Apply processing based on stage
        switch (stage.id) {
            case 'voiceFilter':
                processedAudio = processingState.audioTester.applyVoiceFilter(audioData);
                break;
            case 'vad':
                processedAudio = processingState.audioTester.applyVAD(audioData);
                break;
            case 'noise':
                processedAudio = processingState.audioTester.applyNoiseReduction(audioData);
                break;
            case 'voiceEnhance':
                processedAudio = processingState.audioTester.applyVoiceEnhancement(audioData);
                break;
            case 'voiceProcess':
                processedAudio = processingState.audioTester.applyVoiceProcessing(audioData);
                break;
            case 'silence':
                processedAudio = processingState.audioTester.applySilenceTrimming(audioData);
                break;
            case 'resample':
                if (audioProcessingParams.resampling.enabled) {
                    processedAudio = await processingState.audioTester.resampleAudio(
                        audioData, 
                        audioProcessingParams.resampling.targetRate
                    );
                }
                break;
        }
        
        // Store result
        processingState.stageResults[stage.id] = processedAudio;
        processingState.audioTester.processedStages[stage.id] = {
            audioBuffer: processedAudio,
            description: stage.description
        };
        
        // Create stage UI
        createStageUI(stage, processedAudio);
        
        // Check if we should pause
        if (stage.canPause && document.getElementById(`pause-${stage.id}`)?.checked) {
            processingState.isPaused = true;
            showPauseOverlay(stage, processedAudio);
        } else {
            // Continue to next stage - use non-recursive approach
            processingState.currentStage++;
            // Schedule next stage processing without recursion
            setTimeout(() => processNextStage(processedAudio), 0);
        }
        
    } catch (error) {
        addTestLog('ERROR', `Stage '${stage.name}' failed: ${error.message}`);
        updateProcessingStatus(`Failed at ${stage.name}: ${error.message}`);
    }
}

// Create UI for a processing stage
function createStageUI(stage, audioData) {
    const stageCard = document.createElement('div');
    stageCard.className = 'stage-card';
    stageCard.id = `stage-${stage.id}`;
    
    // Analyze audio
    const analysis = processingState.audioTester.analyzeAudio(audioData);
    
    stageCard.innerHTML = `
        <div class="stage-info">
            <h4>${stage.name}</h4>
            <div class="stage-details">${stage.description}</div>
            <div class="stage-stats" id="stats-${stage.id}">
                Duration: ${analysis.duration} | 
                Samples: ${analysis.samples} |
                Rate: ${analysis.sampleRate} |
                RMS: ${analysis.rmsDb} |
                Peak: ${analysis.peakDb}
            </div>
        </div>
        <div class="stage-controls">
            ${stage.canPause ? `
                <label style="margin-right: 0.5rem;">
                    <input type="checkbox" id="pause-${stage.id}" ${stage.id === 'vad' ? 'checked' : ''}>
                    Pause Here
                </label>
            ` : ''}
            <button onclick="playStageAudio('${stage.id}')" class="button small">▶ Play</button>
            <button onclick="downloadStageAudio('${stage.id}')" class="button small secondary">💾 Download</button>
            <button onclick="analyzeStageAudio('${stage.id}')" class="button small secondary">📊 Analyze</button>
            <button onclick="transcribeStageAudio('${stage.id}')" class="button small primary">🎯 Transcribe</button>
        </div>
    `;
    
    document.getElementById('processingStages').appendChild(stageCard);
}

// Show pause overlay
function showPauseOverlay(stage, audioData) {
    const overlay = document.createElement('div');
    overlay.className = 'pause-overlay';
    
    const analysis = processingState.audioTester.analyzeAudio(audioData);
    
    overlay.innerHTML = `
        <h3>⏸ Paused at: ${stage.name}</h3>
        <p>Review the processed audio and adjust parameters if needed.</p>
        <div style="text-align: left; margin: 1rem 0;">
            <strong>Audio Analysis:</strong><br>
            Duration: ${analysis.duration}<br>
            RMS Level: ${analysis.rmsDb}<br>
            Peak Level: ${analysis.peakDb}<br>
            Dynamic Range: ${analysis.dynamicRange}<br>
            Clipping: ${analysis.clipping}
        </div>
        <div class="button-group">
            <button onclick="resumePipeline()" class="button primary">▶ Continue</button>
            <button onclick="restartPipeline()" class="button secondary">🔄 Restart Pipeline</button>
            <button onclick="cancelPipeline()" class="button danger">✖ Cancel</button>
        </div>
    `;
    
    document.body.appendChild(overlay);
    
    // Highlight paused stage
    document.getElementById(`stage-${stage.id}`).classList.add('stage-paused');
}

// Resume pipeline processing
function resumePipeline() {
    // Remove overlay
    document.querySelector('.pause-overlay')?.remove();
    
    // Remove highlight
    document.querySelectorAll('.stage-paused').forEach(el => {
        el.classList.remove('stage-paused');
    });
    
    // Continue processing
    processingState.isPaused = false;
    processingState.currentStage++;
    
    const lastStageId = processingStages[processingState.currentStage - 1].id;
    const lastAudioData = processingState.stageResults[lastStageId];
    
    // Use non-recursive approach for safety
    setTimeout(() => processNextStage(lastAudioData), 0);
}

// Restart pipeline from beginning
function restartPipeline() {
    document.querySelector('.pause-overlay')?.remove();
    document.getElementById('processingStages').innerHTML = '';
    // Reset processing state and start over
    processingState.currentStage = 0;
    processingState.isPaused = false;
    // Don't call processAudioPipeline directly to avoid recursion
    // Let the user click the button again or call the enhanced version directly
    updateProcessingStatus('Click "Process Audio" to restart pipeline');
}

// Cancel pipeline
function cancelPipeline() {
    document.querySelector('.pause-overlay')?.remove();
    processingState.currentStage = processingStages.length;
    updateProcessingStatus('Pipeline cancelled');
}

// Audio processing wrapper functions
async function applyVoiceFilter(audioData) {
    return processingState.audioTester.applyVoiceFilter(audioData);
}

async function applyVAD(audioData) {
    return processingState.audioTester.applyVAD(audioData);
}

async function applyNoiseReduction(audioData) {
    return processingState.audioTester.applyNoiseReduction(audioData);
}

async function applyVoiceEnhancement(audioData) {
    return processingState.audioTester.applyVoiceEnhancement(audioData);
}

async function applyVoiceProcessing(audioData) {
    return processingState.audioTester.applyVoiceProcessing(audioData);
}

async function applyEnhancement(audioData) {
    return processingState.audioTester.applyEnhancement(audioData);
}

async function applySilenceTrimming(audioData) {
    return processingState.audioTester.applySilenceTrimming(audioData);
}

async function applyResampling(audioData) {
    if (!audioProcessingParams.resampling.enabled) return audioData;
    return processingState.audioTester.resampleAudio(audioData, audioProcessingParams.resampling.targetRate);
}

// Helper functions
function playStageAudio(stageId) {
    processingState.audioTester.playStage(stageId);
    addTestLog('INFO', `Playing audio from stage: ${stageId}`);
}

function downloadStageAudio(stageId) {
    processingState.audioTester.exportStage(stageId);
    addTestLog('INFO', `Downloading audio from stage: ${stageId}`);
}

function analyzeStageAudio(stageId) {
    const analysis = processingState.audioTester.analyzeAudio(processingState.stageResults[stageId]);
    
    // Update analysis display
    const analysisGrid = document.getElementById('analysisGrid');
    if (analysisGrid) {
        analysisGrid.innerHTML = `
            <div class="analysis-item">
                <label>Stage</label>
                <div class="value">${processingStages.find(s => s.id === stageId).name}</div>
            </div>
            <div class="analysis-item">
                <label>Duration</label>
                <div class="value">${analysis.duration}</div>
            </div>
            <div class="analysis-item">
                <label>Sample Rate</label>
                <div class="value">${analysis.sampleRate}</div>
            </div>
            <div class="analysis-item">
                <label>Channels</label>
                <div class="value">${analysis.channels}</div>
            </div>
            <div class="analysis-item">
                <label>RMS Level</label>
                <div class="value">${analysis.rmsDb}</div>
            </div>
            <div class="analysis-item">
                <label>Peak Level</label>
                <div class="value">${analysis.peakDb}</div>
            </div>
            <div class="analysis-item">
                <label>Dynamic Range</label>
                <div class="value">${analysis.dynamicRange}</div>
            </div>
            <div class="analysis-item">
                <label>Clipping</label>
                <div class="value">${analysis.clipping}</div>
            </div>
        `;
    }
    
    addTestLog('INFO', `Analysis for ${stageId}: RMS=${analysis.rmsDb}, Peak=${analysis.peakDb}, DR=${analysis.dynamicRange}`);
}

async function transcribeStageAudio(stageId) {
    const modelSelect = document.getElementById('testModel');
    const selectedModel = modelSelect?.value;
    
    if (!selectedModel) {
        addTestLog('WARNING', 'Please select a model first');
        return;
    }
    
    try {
        addTestLog('INFO', `Transcribing ${stageId} with model: ${selectedModel}`);
        const result = await processingState.audioTester.sendToWhisper(stageId, selectedModel);
        
        if (result.text) {
            addTestLog('SUCCESS', `Transcription: "${result.text}"`);
        } else if (result.error) {
            addTestLog('ERROR', `Transcription failed: ${result.error}`);
        }
    } catch (error) {
        addTestLog('ERROR', `Transcription error: ${error.message}`);
    }
}

function updateProcessingStatus(message) {
    const statusDiv = document.getElementById('processingStatus');
    if (statusDiv) {
        statusDiv.textContent = message;
    }
}

// Set up event listeners for parameter controls
function setupParameterListeners() {
    // Range inputs - update display values
    document.querySelectorAll('input[type="range"]').forEach(input => {
        input.addEventListener('input', (e) => {
            const valueSpan = document.getElementById(e.target.id + '-value');
            if (valueSpan) {
                valueSpan.textContent = e.target.value;
                if (e.target.id.includes('threshold') && e.target.id.includes('energy')) {
                    valueSpan.textContent = parseFloat(e.target.value).toFixed(3);
                }
            }
        });
    });
    
    // All inputs - update parameters on change
    document.querySelectorAll('.parameter-controls input, .parameter-controls select').forEach(input => {
        input.addEventListener('change', updateParameters);
    });
}

// Reset parameters to defaults
function resetParameters() {
    if (confirm('Reset all parameters to defaults?')) {
        location.reload(); // Simple way to reset
    }
}

// Save parameters to localStorage
function saveParameters() {
    updateParameters();
    localStorage.setItem('audioProcessingParams', JSON.stringify(audioProcessingParams));
    addTestLog('SUCCESS', 'Parameters saved to local storage');
}

// Load saved parameters
function loadSavedParameters() {
    const saved = localStorage.getItem('audioProcessingParams');
    if (saved) {
        try {
            Object.assign(audioProcessingParams, JSON.parse(saved));
            addTestLog('INFO', 'Loaded saved parameters');
        } catch (e) {
            addTestLog('WARNING', 'Failed to load saved parameters');
        }
    }
}

// Clear processing results
function clearProcessingResults() {
    document.getElementById('processingStages').innerHTML = '';
    document.getElementById('processingStages').style.display = 'none';
    document.getElementById('audioAnalysis').style.display = 'none';
    document.querySelector('.parameter-controls')?.remove();
    
    processingState = {
        currentStage: 0,
        isPaused: false,
        stageResults: {},
        audioContext: processingState.audioContext,
        audioTester: processingState.audioTester
    };
    
    updateProcessingStatus('Record audio first, then process through pipeline');
    addTestLog('INFO', 'Cleared processing results');
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadSavedParameters();
    
    // Override the global processAudioPipeline function
    window.processAudioPipelineEnhanced = processAudioPipeline;
    
    // Make enhanced functions globally available  
    window.processAudioThroughPipelineEnhanced = processAudioPipeline;
    window.clearProcessingResults = clearProcessingResults;
    window.resumePipeline = resumePipeline;
    window.restartPipeline = restartPipeline;
    window.cancelPipeline = cancelPipeline;
    window.playStageAudio = playStageAudio;
    window.downloadStageAudio = downloadStageAudio;
    window.analyzeStageAudio = analyzeStageAudio;
    window.transcribeStageAudio = transcribeStageAudio;
    window.resetParameters = resetParameters;
    window.saveParameters = saveParameters;
});

// Export for use in test page
window.AudioProcessingTester = AudioProcessingTester;