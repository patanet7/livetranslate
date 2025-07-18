// Audio module for recording and streaming functionality
const AudioModule = {
    async loadAudioDevices() {
        try {
            addLog('INFO', 'Loading audio input devices...');
            
            // Request permission first
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioDevices = devices.filter(device => device.kind === 'audioinput');
            
            // Stop the permission stream
            stream.getTracks().forEach(track => track.stop());
            
            if (elements.audioDevice) {
                const currentValue = elements.audioDevice.value;
                const wasVisualizationRunning = !!state.visualizationStream;
                
                addLog('DEBUG', `Current device selection: ${currentValue}, Visualization running: ${wasVisualizationRunning}`);
                
                elements.audioDevice.innerHTML = '<option value="">Default Device</option>';
                
                audioDevices.forEach((device, index) => {
                    const option = document.createElement('option');
                    option.value = device.deviceId;
                    option.textContent = device.label || `Audio Device ${index + 1} (${device.deviceId.substring(0, 8)})`;
                    elements.audioDevice.appendChild(option);
                    
                    addLog('DEBUG', `Found device: ${option.textContent} (ID: ${device.deviceId})`);
                });
                
                // Restore previous selection if still available
                if (currentValue && [...elements.audioDevice.options].some(opt => opt.value === currentValue)) {
                    elements.audioDevice.value = currentValue;
                    addLog('INFO', `Restored previous device selection: ${currentValue}`);
                } else if (currentValue) {
                    addLog('WARNING', `Previous device ${currentValue} no longer available`);
                }
            }
            
            addLog('SUCCESS', `Found ${audioDevices.length} audio input devices`);
            
            // Don't automatically start visualization here - let the main app control it
            // This prevents conflicts when device list is refreshed
            
        } catch (error) {
            addLog('ERROR', `Failed to load audio devices: ${error.message}`);
            
            // Add helpful error information
            if (error.name === 'NotAllowedError') {
                addLog('ERROR', 'Microphone access denied. Please allow microphone access and refresh the page.');
            } else if (error.name === 'NotFoundError') {
                addLog('ERROR', 'No audio input devices found.');
            }
        }
    },

    async toggleRecording() {
        if (state.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    },

    async startRecording() {
        try {
            if (!state.selectedModel) {
                addLog('WARNING', 'Please select a model first');
                return;
            }
            
            addLog('INFO', 'Starting audio recording...');
            
            const constraints = {
                audio: {
                    deviceId: elements.audioDevice?.value ? { exact: elements.audioDevice.value } : undefined,
                    sampleRate: parseInt(elements.sampleRate?.value || '16000'),
                    echoCancellation: false,  // Match successful test
                    noiseSuppression: false,  // Match successful test
                    autoGainControl: false    // Match successful test
                }
            };
            
            state.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Log actual audio settings for debugging
            const audioTrack = state.currentStream.getAudioTracks()[0];
            const settings = audioTrack.getSettings();
            addLog('INFO', `Recording settings: ${settings.sampleRate}Hz, ${settings.channelCount} channels`);
            addLog('INFO', `Requested: ${elements.sampleRate?.value || '16000'}Hz, Got: ${settings.sampleRate}Hz`);
            
            state.mediaRecorder = new MediaRecorder(state.currentStream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            state.audioChunks = [];
            
            state.mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    state.audioChunks.push(event.data);
                    addLog('INFO', `Recording chunk: ${event.data.size} bytes`);
                }
            };
            
            state.mediaRecorder.onstop = async function() {
                const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
                state.lastAudioBlob = audioBlob;
                addLog('INFO', `Recording complete: ${audioBlob.size} bytes total`);
                
                // Transcribe the audio
                if (audioBlob.size > 0) {
                    await API.transcribeAudio(audioBlob, state.selectedModel);
                }
                
                state.isRecording = false;
                AudioModule.updateRecordingUI();
            };
            
            state.mediaRecorder.start();
            state.isRecording = true;
            this.updateRecordingUI();
            
            addLog('SUCCESS', 'Recording started');
            
        } catch (error) {
            addLog('ERROR', `Failed to start recording: ${error.message}`);
            state.isRecording = false;
            this.updateRecordingUI();
        }
    },

    stopRecording() {
        if (state.mediaRecorder && state.isRecording) {
            addLog('INFO', 'Stopping recording...');
            state.mediaRecorder.stop();
            if (state.currentStream) {
                state.currentStream.getTracks().forEach(track => track.stop());
            }
        }
    },

    async toggleStreaming() {
        if (state.isStreaming) {
            this.stopStreaming();
        } else {
            await this.startStreaming();
        }
    },

    async startStreaming() {
        try {
            if (!state.selectedModel) {
                addLog('WARNING', 'Please select a model first');
                return;
            }
            
            addLog('INFO', 'Starting continuous streaming with 10-second chunks...');
            
            const constraints = {
                audio: {
                    deviceId: elements.audioDevice?.value ? { exact: elements.audioDevice.value } : undefined,
                    sampleRate: parseInt(elements.sampleRate?.value || '16000'),
                    echoCancellation: false,  // Match successful test
                    noiseSuppression: false,  // Match successful test
                    autoGainControl: false    // Match successful test
                }
            };
            
            state.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Log actual audio settings for debugging
            const audioTrack = state.currentStream.getAudioTracks()[0];
            const settings = audioTrack.getSettings();
            addLog('INFO', `Streaming settings: ${settings.sampleRate}Hz, ${settings.channelCount} channels`);
            
            // DETAILED MediaRecorder format testing
            addLog('INFO', 'Testing MediaRecorder format support...');
            const testFormats = [
                'audio/mp4',
                'audio/mp4;codecs=mp4a.40.2',
                'audio/ogg;codecs=opus',
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/wav'
            ];
            
            let supportedFormats = [];
            testFormats.forEach(format => {
                const supported = MediaRecorder.isTypeSupported(format);
                addLog('INFO', `${format}: ${supported ? '✓ SUPPORTED' : '✗ not supported'}`);
                if (supported) supportedFormats.push(format);
            });
            
            // Use format selection like the successful test
            let streamingMimeType = 'audio/webm;codecs=opus'; // fallback
            
            // Auto-detect best format (like the test)
            if (supportedFormats.includes('audio/mp4')) {
                streamingMimeType = 'audio/mp4';
                addLog('INFO', '✓ Using MP4 format (matches successful test)');
            } else if (supportedFormats.includes('audio/ogg;codecs=opus')) {
                streamingMimeType = 'audio/ogg;codecs=opus';
                addLog('INFO', '✓ Using OGG/Opus format (reliable backup)');
            } else if (supportedFormats.includes('audio/webm;codecs=opus')) {
                streamingMimeType = 'audio/webm;codecs=opus';
                addLog('INFO', '✓ Using WebM/Opus format (last resort)');
            } else if (supportedFormats.includes('audio/webm')) {
                streamingMimeType = 'audio/webm';
                addLog('WARNING', '⚠ Using basic WebM format');
            } else {
                addLog('ERROR', 'No reliable audio formats supported by MediaRecorder!');
            }
            
            // Setup for complete recording approach (like successful test)
            addLog('INFO', `Will use ${streamingMimeType} for complete 10-second recordings`);
            
            state.chunkCount = 0;
            state.processingChunk = false;
            state.isStreaming = true;
            this.updateStreamingUI();
            
            // Start the complete recording loop
            this.startSimpleChunkLoop(streamingMimeType);
            
            addLog('SUCCESS', `Streaming started with ${streamingMimeType} (complete 10-second recordings)`);
            
        } catch (error) {
            addLog('ERROR', `Failed to start streaming: ${error.message}`);
            state.isStreaming = false;
            this.updateStreamingUI();
        }
    },

    async startSimpleChunkLoop(mimeType) {
        // Use complete recordings like the successful test
        const recordingDuration = 10000; // 10 seconds like successful test
        
        const recordCompleteChunk = async () => {
            if (!state.isStreaming) return;
            
            try {
                const chunkNumber = ++state.chunkCount;
                addLog('INFO', `Starting complete recording ${chunkNumber} (${recordingDuration/1000}s)...`);
                
                // Create new MediaRecorder for each complete recording (like test)
                const recorder = new MediaRecorder(state.currentStream, { mimeType: mimeType });
                const chunks = [];
                
                recorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        chunks.push(event.data);
                        addLog('DEBUG', `Recording ${chunkNumber} chunk: ${event.data.size} bytes`);
                    }
                };
                
                recorder.onstop = () => {
                    if (chunks.length > 0) {
                        const completeBlob = new Blob(chunks, { type: mimeType });
                        addLog('INFO', `Complete recording ${chunkNumber}: ${completeBlob.size} bytes`);
                        
                        // Process the complete recording
                        this.processStreamingChunk(completeBlob, chunkNumber);
                    }
                    
                    // Schedule next recording if still streaming
                    if (state.isStreaming) {
                        setTimeout(recordCompleteChunk, 500); // Small gap between recordings
                    }
                };
                
                // Start recording for complete duration
                recorder.start();
                
                // Stop after duration (like test)
                setTimeout(() => {
                    if (recorder.state === 'recording') {
                        recorder.stop();
                    }
                }, recordingDuration);
                
            } catch (error) {
                addLog('ERROR', `Recording ${state.chunkCount} failed: ${error.message}`);
                if (state.isStreaming) {
                    setTimeout(recordCompleteChunk, 2000); // Retry after error
                }
            }
        };
        
        // Start the first recording
        recordCompleteChunk();
    },

    stopStreaming() {
        try {
            addLog('INFO', 'Stopping streaming...');
            
            state.isStreaming = false;
            
            // Stop all audio tracks
            if (state.currentStream) {
                state.currentStream.getTracks().forEach(track => {
                    track.stop();
                    addLog('DEBUG', `Stopped audio track: ${track.kind}`);
                });
                state.currentStream = null;
            }
            
            // Clean up state (no single mediaRecorder to stop)
            state.mediaRecorder = null;
            state.processingChunk = false;
            state.chunkCount = 0;
            
            this.updateStreamingUI();
            addLog('SUCCESS', 'Streaming stopped');
            
        } catch (error) {
            addLog('ERROR', `Error stopping streaming: ${error.message}`);
            
            // Force cleanup anyway
            state.isStreaming = false;
            state.currentStream = null;
            state.mediaRecorder = null;
            state.processingChunk = false;
            this.updateStreamingUI();
        }
    },

    async processStreamingChunk(blob, chunkNumber) {
        try {
            addLog('INFO', `Processing chunk ${chunkNumber} (${(blob.size / 1024).toFixed(1)}KB)`);
            
            // Add blob content inspection (helpful for debugging)
            if (blob.size > 0) {
                const reader = new FileReader();
                reader.onload = () => {
                    const arrayBuffer = reader.result;
                    const uint8Array = new Uint8Array(arrayBuffer);
                    const firstBytes = Array.from(uint8Array.slice(0, 16)).map(b => b.toString(16).padStart(2, '0')).join(' ');
                    addLog('DEBUG', `Chunk ${chunkNumber} first 16 bytes: ${firstBytes}`);
                };
                reader.readAsArrayBuffer(blob.slice(0, 16));
            }
            
            // Direct transcription (like the test)
            const url = `${config.serverUrl}/transcribe/${state.selectedModel}`;
            const response = await fetch(url, {
                method: 'POST',
                body: blob
            });
            
            const data = await response.json();
            
            if (response.ok) {
                const transcribedText = data.text.trim();
                if (transcribedText) {
                    addLog('SUCCESS', `[Chunk ${chunkNumber}] ${transcribedText}`);
                    
                    // Add to streaming output
                    const streamingOutput = elements.streamingOutput;
                    if (streamingOutput) {
                        const timestamp = new Date().toLocaleTimeString();
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry SUCCESS';
                        logEntry.innerHTML = `[${timestamp}] [Chunk ${chunkNumber}] ${transcribedText}`;
                        streamingOutput.appendChild(logEntry);
                        streamingOutput.scrollTop = streamingOutput.scrollHeight;
                    }
                } else {
                    addLog('DEBUG', `Chunk ${chunkNumber}: Empty transcription (silence or unclear audio)`);
                }
                
                // Log processing details
                addLog('DEBUG', `Chunk ${chunkNumber} processing: ${data.processing_time}, audio: ${data.audio_length}, device: ${data.device}`);
                
            } else {
                addLog('ERROR', `Chunk ${chunkNumber} transcription failed: ${data.error}`);
                if (data.debug_info) {
                    addLog('DEBUG', `Debug info: ${JSON.stringify(data.debug_info)}`);
                }
            }
            
        } catch (error) {
            addLog('ERROR', `Chunk ${chunkNumber} processing error: ${error.message}`);
        }
    },

    updateRecordingUI() {
        if (elements.recordButton) {
            if (state.isRecording) {
                elements.recordButton.textContent = '⏹ Stop Recording';
                elements.recordButton.classList.add('recording');
            } else {
                elements.recordButton.textContent = '🎤 Start Recording';
                elements.recordButton.classList.remove('recording');
            }
        }
    },

    updateStreamingUI() {
        if (elements.streamButton) {
            if (state.isStreaming) {
                elements.streamButton.textContent = '⏹ Stop Streaming';
                elements.streamButton.classList.add('recording');
            } else {
                elements.streamButton.textContent = '🔄 Start Streaming';
                elements.streamButton.classList.remove('recording');
            }
        }
    },

    async handleAudioDeviceChange() {
        const selectedDevice = elements.audioDevice?.value;
        const selectedDeviceName = elements.audioDevice?.options[elements.audioDevice?.selectedIndex]?.text || 'Default';
        
        addLog('INFO', `Audio device change requested to: ${selectedDeviceName} (ID: ${selectedDevice || 'default'})`);
        
        // Stop current visualization
        AudioVisualization.stopAudioVisualization();
        
        // Wait a moment then restart with new device
        setTimeout(async () => {
            addLog('INFO', 'Restarting audio visualization with new device...');
            await AudioVisualization.startAudioVisualization();
        }, 500);
    },

    async handleSampleRateChange() {
        const sampleRate = elements.sampleRate?.value;
        addLog('INFO', `Sample rate change requested to: ${sampleRate}Hz - restarting visualization...`);
        
        // Stop current visualization
        AudioVisualization.stopAudioVisualization();
        
        // Wait a moment then restart with new sample rate
        setTimeout(async () => {
            addLog('INFO', 'Restarting audio visualization with new sample rate...');
            await AudioVisualization.startAudioVisualization();
        }, 500);
    }
};

// Audio Visualization Functions
const AudioVisualization = {
    async startAudioVisualization() {
        try {
            addLog('INFO', 'Starting audio visualization...');
            
            // Get current device selection with detailed logging
            const deviceId = elements.audioDevice?.value;
            const selectedDeviceName = elements.audioDevice?.options[elements.audioDevice?.selectedIndex]?.text || 'Default';
            const sampleRate = parseInt(elements.sampleRate?.value || '16000');
            
            addLog('INFO', `Device selection: Name="${selectedDeviceName}", ID="${deviceId || 'default'}", SampleRate=${sampleRate}Hz`);
            
            // Build constraints with detailed logging
            const constraints = {
                audio: {
                    echoCancellation: false, // Better for visualization
                    noiseSuppression: false  // Better for visualization
                }
            };
            
            // Add device constraint if specified
            if (deviceId && deviceId !== '') {
                constraints.audio.deviceId = { exact: deviceId };
                addLog('INFO', `Using specific device ID: ${deviceId}`);
            } else {
                addLog('INFO', 'Using default audio device');
            }
            
            // Add sample rate constraint
            constraints.audio.sampleRate = sampleRate;
            addLog('DEBUG', `Audio constraints: ${JSON.stringify(constraints)}`);

            // Stop any existing visualization first
            this.stopAudioVisualization();

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Log actual audio settings
            const audioTrack = stream.getAudioTracks()[0];
            const settings = audioTrack.getSettings();
            addLog('SUCCESS', `Audio stream created: Device="${settings.deviceId}", SampleRate=${settings.sampleRate}Hz, Channels=${settings.channelCount}`);
            
            // Check if we got the device we requested
            if (deviceId && deviceId !== '' && settings.deviceId !== deviceId) {
                addLog('WARNING', `Requested device ${deviceId} but got ${settings.deviceId}`);
            }
            
            // Create Web Audio API context
            state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            state.analyser = state.audioContext.createAnalyser();
            state.microphone = state.audioContext.createMediaStreamSource(stream);
            
            // Configure analyser
            state.analyser.fftSize = 512;
            state.analyser.smoothingTimeConstant = 0.3;
            
            // Connect microphone to analyser
            state.microphone.connect(state.analyser);
            
            // Store the stream for cleanup
            state.visualizationStream = stream;
            
            // Update selected device display
            if (elements.selectedAudioDevice) {
                elements.selectedAudioDevice.textContent = selectedDeviceName;
            }
            
            // Start visualization loops
            this.startAudioLevelMonitoring();
            this.startFFTVisualization();
            
            addLog('SUCCESS', `Audio visualization started successfully with device: ${selectedDeviceName}`);
            
        } catch (error) {
            addLog('ERROR', `Failed to start audio visualization: ${error.message}`);
            
            // Reset UI on error
            if (elements.audioLevelBar) elements.audioLevelBar.style.height = '0%';
            if (elements.audioLevelText) elements.audioLevelText.textContent = '0%';
            if (elements.selectedAudioDevice) elements.selectedAudioDevice.textContent = 'Error - No device';
            
            // If it's a device constraint error, provide helpful message
            if (error.name === 'ConstraintNotSatisfiedError' || error.name === 'OverconstrainedError') {
                addLog('ERROR', 'Device constraint failed - the selected device may not be available');
                addLog('INFO', 'Trying to fall back to default device...');
                
                // Try with default device
                try {
                    const fallbackConstraints = {
                        audio: {
                            sampleRate: parseInt(elements.sampleRate?.value || '16000'),
                            echoCancellation: false,
                            noiseSuppression: false
                        }
                    };
                    
                    const fallbackStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
                    
                    // Set up with fallback stream
                    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    state.analyser = state.audioContext.createAnalyser();
                    state.microphone = state.audioContext.createMediaStreamSource(fallbackStream);
                    state.analyser.fftSize = 512;
                    state.analyser.smoothingTimeConstant = 0.3;
                    state.microphone.connect(state.analyser);
                    state.visualizationStream = fallbackStream;
                    
                    this.startAudioLevelMonitoring();
                    this.startFFTVisualization();
                    
                    if (elements.selectedAudioDevice) {
                        elements.selectedAudioDevice.textContent = 'Default (fallback)';
                    }
                    
                    addLog('SUCCESS', 'Audio visualization started with fallback to default device');
                    
                } catch (fallbackError) {
                    addLog('ERROR', `Fallback also failed: ${fallbackError.message}`);
                }
            }
        }
    },

    stopAudioVisualization() {
        addLog('INFO', 'Stopping audio visualization...');
        
        // Clear intervals and animation frames
        if (state.audioLevelInterval) {
            cancelAnimationFrame(state.audioLevelInterval);
            state.audioLevelInterval = null;
        }
        
        if (state.fftAnimationFrame) {
            cancelAnimationFrame(state.fftAnimationFrame);
            state.fftAnimationFrame = null;
        }
        
        // Disconnect Web Audio API
        if (state.microphone) {
            try {
                state.microphone.disconnect();
            } catch (error) {
                addLog('DEBUG', `Error disconnecting microphone: ${error.message}`);
            }
            state.microphone = null;
        }
        
        // Stop visualization stream
        if (state.visualizationStream) {
            state.visualizationStream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (error) {
                    addLog('DEBUG', `Error stopping track: ${error.message}`);
                }
            });
            state.visualizationStream = null;
        }
        
        // Close audio context
        if (state.audioContext) {
            try {
                state.audioContext.close();
            } catch (error) {
                addLog('DEBUG', `Error closing audio context: ${error.message}`);
            }
            state.audioContext = null;
        }
        
        state.analyser = null;
        
        // Reset UI
        if (elements.audioLevelBar) elements.audioLevelBar.style.height = '0%';
        if (elements.audioLevelText) elements.audioLevelText.textContent = '0%';
        if (elements.selectedAudioDevice) elements.selectedAudioDevice.textContent = 'No device selected';
        
        // Clear FFT canvas
        if (elements.fftCanvas) {
            const canvas = elements.fftCanvas;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
        
        addLog('INFO', 'Audio visualization stopped');
    },

    // Helper function to interpolate between colors
    interpolateColor(level) {
        // Define our color stops
        const colors = [
            { level: 0, color: '#4CAF50' },    // Green
            { level: 90, color: '#4CAF50' },   // Green
            { level: 100, color: '#FFC107' }   // Yellow
        ];

        // Find the two colors to interpolate between
        let color1, color2;
        for (let i = 0; i < colors.length - 1; i++) {
            if (level <= colors[i + 1].level) {
                color1 = colors[i];
                color2 = colors[i + 1];
                break;
            }
        }

        // If level is above 100, use red
        if (level >= 100) {
            return '#F44336';
        }

        // Calculate the interpolation factor
        const factor = (level - color1.level) / (color2.level - color1.level);

        // Convert hex to RGB
        const hexToRgb = (hex) => {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            } : null;
        };

        const rgb1 = hexToRgb(color1.color);
        const rgb2 = hexToRgb(color2.color);

        // Interpolate each RGB component
        const r = Math.round(rgb1.r + (rgb2.r - rgb1.r) * factor);
        const g = Math.round(rgb1.g + (rgb2.g - rgb1.g) * factor);
        const b = Math.round(rgb1.b + (rgb2.b - rgb1.b) * factor);

        // Convert back to hex
        return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
    },

    startAudioLevelMonitoring() {
        if (!state.audioContext || !state.analyser) {
            addLog('ERROR', 'Audio context or analyser not initialized for level monitoring');
            return;
        }

        addLog('DEBUG', 'Starting audio level monitoring...');
        
        let peakLevel = 0;
        let peakTimer = null;

        const updateLevel = () => {
            // Check if we still have the analyzer (in case it was stopped)
            if (!state.analyser) {
                return;
            }
            
            const dataArray = new Uint8Array(state.analyser.frequencyBinCount);
            state.analyser.getByteFrequencyData(dataArray);
            
            // Calculate RMS for proper dB scaling
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                const value = dataArray[i] / 255;
                sum += value * value;
            }
            
            const rms = Math.sqrt(sum / dataArray.length);
            
            // Convert to dB scale (proper volume meter scaling)
            // -60dB = 0%, 0dB = 100%
            let dB = -Infinity;
            if (rms > 0) {
                dB = 20 * Math.log10(rms);
            }
            
            // Map dB to percentage (-60dB to 0dB = 0% to 100%)
            const minDB = -60;
            const maxDB = 0;
            const level = Math.min(100, Math.max(0, Math.round(((dB - minDB) / (maxDB - minDB)) * 100)));
            
            // Update peak level
            if (level > peakLevel) {
                peakLevel = level;
                // Clear any existing timer
                if (peakTimer) {
                    clearTimeout(peakTimer);
                }
                // Set new timer to reset peak after 50ms
                peakTimer = setTimeout(() => {
                    peakLevel = 0;
                }, 50);
            }
            
            // Get interpolated color based on level
            let color = this.interpolateColor(level);
            
            // If we're at peak level, use red
            if (peakLevel >= 100) {
                color = '#F44336';
            }
            
            // Update vertical level bar
            const levelBar = document.getElementById('audioLevelBar');
            const levelText = document.getElementById('audioLevelText');
            if (levelBar && levelText) {
                levelBar.style.height = `${level}%`;
                levelText.textContent = `${level}%`;
                levelBar.style.background = color;
            }

            // Update test page level bar if it exists
            const testLevelBar = document.getElementById('testAudioLevelBar');
            const testLevelText = document.getElementById('testAudioLevel');
            if (testLevelBar && testLevelText) {
                testLevelBar.style.width = `${level}%`;
                testLevelText.textContent = `${level}%`;
                testLevelBar.style.background = color;
            }

            // Continue monitoring
            state.audioLevelInterval = requestAnimationFrame(updateLevel);
        };
        
        updateLevel();
    },

    startFFTVisualization() {
        if (!elements.fftCanvas) {
            addLog('DEBUG', 'No FFT canvas found, skipping FFT visualization');
            return;
        }
        
        if (!state.analyser) {
            addLog('ERROR', 'No analyser available for FFT visualization');
            return;
        }
        
        addLog('DEBUG', 'Starting FFT visualization...');
        
        const canvas = elements.fftCanvas;
        const ctx = canvas.getContext('2d');
        
        const draw = () => {
            // Check if we still have the analyzer (in case it was stopped)
            if (!state.analyser) {
                return;
            }
            
            state.fftAnimationFrame = requestAnimationFrame(draw);
            
            const bufferLength = state.analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            state.analyser.getByteFrequencyData(dataArray);
            
            // Clear canvas
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw frequency bars
            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                barHeight = (dataArray[i] / 255) * canvas.height;
                
                // Create gradient based on frequency
                const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
                gradient.addColorStop(0, '#4CAF50');   // Green
                gradient.addColorStop(0.5, '#FFC107'); // Yellow
                gradient.addColorStop(1, '#F44336');   // Red
                
                ctx.fillStyle = gradient;
                ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                
                x += barWidth + 1;
            }
        };
        
        draw();
    }
};

// Global functions for backwards compatibility - ONLY for recording/streaming, not device handling
window.toggleRecording = AudioModule.toggleRecording.bind(AudioModule);
window.toggleStreaming = AudioModule.toggleStreaming.bind(AudioModule);
window.loadAudioDevices = AudioModule.loadAudioDevices.bind(AudioModule);

// Export AudioVisualization to global scope so main.js can access it
window.AudioVisualization = AudioVisualization;
window.AudioModule = AudioModule;

// Remove these global assignments to prevent conflicts with main.js
// window.handleAudioDeviceChange = AudioModule.handleAudioDeviceChange.bind(AudioModule);
// window.handleSampleRateChange = AudioModule.handleSampleRateChange.bind(AudioModule);
// window.startAudioVisualization = AudioVisualization.startAudioVisualization.bind(AudioVisualization);
// window.stopAudioVisualization = AudioVisualization.stopAudioVisualization.bind(AudioVisualization); 