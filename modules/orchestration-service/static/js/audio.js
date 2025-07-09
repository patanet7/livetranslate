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
            
            addLog('INFO', 'Starting high-quality audio recording...');
            
            const selectedSampleRate = parseInt(elements.sampleRate?.value || '16000');
            const selectedDevice = elements.audioDevice?.value;
            
            // Enhanced audio constraints for better quality
            const constraints = {
                audio: {
                    deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
                    sampleRate: selectedSampleRate,
                    channelCount: 1, // Mono for speech
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    // Advanced constraints for better quality
                    sampleSize: 16,
                    latency: 0.01 // 10ms latency for real-time processing
                }
            };
            
            state.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Verify actual stream settings
            const audioTrack = state.currentStream.getAudioTracks()[0];
            const settings = audioTrack.getSettings();
            const capabilities = audioTrack.getCapabilities();
            
            addLog('INFO', `Recording settings: ${settings.sampleRate}Hz, ${settings.channelCount}ch`);
            addLog('INFO', `Device capabilities: ${capabilities.sampleRate?.min || 'N/A'}-${capabilities.sampleRate?.max || 'N/A'}Hz`);
            
            // Validate that we got the requested sample rate
            if (settings.sampleRate !== selectedSampleRate) {
                addLog('WARNING', `Requested ${selectedSampleRate}Hz but got ${settings.sampleRate}Hz`);
            }
            
            // Check for audio processing features
            const audioFeatures = {
                echoCancellation: settings.echoCancellation,
                noiseSuppression: settings.noiseSuppression,
                autoGainControl: settings.autoGainControl
            };
            addLog('INFO', `Audio features: ${JSON.stringify(audioFeatures)}`);
            
            // Calculate high-quality bit rate based on sample rate
            const bitRate = selectedSampleRate * 16; // 16 bits per sample for high quality
            
            // Enhanced MediaRecorder options based on format
            let mediaRecorderOptions = {};
            
            if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                mediaRecorderOptions = {
                    mimeType: 'audio/webm;codecs=opus',
                    audioBitsPerSecond: bitRate
                };
                addLog('INFO', 'Using WebM/Opus format for recording');
            } else if (MediaRecorder.isTypeSupported('audio/mp4;codecs=mp4a.40.2')) {
                mediaRecorderOptions = {
                    mimeType: 'audio/mp4;codecs=mp4a.40.2',
                    audioBitsPerSecond: bitRate
                };
                addLog('INFO', 'Using MP4/AAC format for recording');
            } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                mediaRecorderOptions = {
                    mimeType: 'audio/ogg;codecs=opus',
                    audioBitsPerSecond: bitRate
                };
                addLog('INFO', 'Using OGG/Opus format for recording');
            } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                mediaRecorderOptions = {
                    mimeType: 'audio/webm',
                    audioBitsPerSecond: bitRate
                };
                addLog('INFO', 'Using basic WebM format for recording');
            } else {
                mediaRecorderOptions = { audioBitsPerSecond: bitRate };
                addLog('WARNING', 'Using default MediaRecorder format');
            }
            
            addLog('INFO', `Recording with: ${selectedSampleRate}Hz, ${mediaRecorderOptions.audioBitsPerSecond}bps`);
            addLog('INFO', `Format: ${mediaRecorderOptions.mimeType || 'default'}`);
            
            state.mediaRecorder = new MediaRecorder(state.currentStream, mediaRecorderOptions);
            
            state.audioChunks = [];
            
            state.mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    state.audioChunks.push(event.data);
                    addLog('INFO', `Recording chunk: ${event.data.size} bytes`);
                }
            };
            
            state.mediaRecorder.onstop = async function() {
                const audioBlob = new Blob(state.audioChunks, { type: mediaRecorderOptions.mimeType || 'audio/webm' });
                state.lastAudioBlob = audioBlob;
                const actualBitRate = (audioBlob.size * 8) / ((Date.now() - state.recordingStartTime) / 1000);
                addLog('INFO', `Recording complete: ${audioBlob.size} bytes, ${actualBitRate.toFixed(0)}bps actual`);
                
                // Transcribe the audio
                if (audioBlob.size > 0) {
                    await API.transcribeAudio(audioBlob, state.selectedModel);
                }
                
                state.isRecording = false;
                AudioModule.updateRecordingUI();
            };
            
            state.mediaRecorder.start();
            state.isRecording = true;
            state.recordingStartTime = Date.now();
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
            
            addLog('INFO', 'Starting high-quality continuous streaming...');
            
            const selectedSampleRate = parseInt(elements.sampleRate?.value || '16000');
            const selectedDevice = elements.audioDevice?.value;
            
            // Enhanced audio constraints for better quality streaming
            const constraints = {
                audio: {
                    deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
                    sampleRate: selectedSampleRate,
                    channelCount: 1, // Mono for speech
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    // Advanced constraints for better quality
                    sampleSize: 16,
                    latency: 0.01 // 10ms latency for real-time processing
                }
            };
            
            state.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Verify actual stream settings
            const audioTrack = state.currentStream.getAudioTracks()[0];
            const settings = audioTrack.getSettings();
            const capabilities = audioTrack.getCapabilities();
            
            addLog('INFO', `Streaming settings: ${settings.sampleRate}Hz, ${settings.channelCount}ch`);
            addLog('INFO', `Device capabilities: ${capabilities.sampleRate?.min || 'N/A'}-${capabilities.sampleRate?.max || 'N/A'}Hz`);
            
            // Validate that we got the requested sample rate
            if (settings.sampleRate !== selectedSampleRate) {
                addLog('WARNING', `Requested ${selectedSampleRate}Hz but got ${settings.sampleRate}Hz`);
            }
            
            // Check for audio processing features
            const audioFeatures = {
                echoCancellation: settings.echoCancellation,
                noiseSuppression: settings.noiseSuppression,
                autoGainControl: settings.autoGainControl
            };
            addLog('INFO', `Audio features: ${JSON.stringify(audioFeatures)}`);
            
            // Calculate high-quality bit rate based on sample rate
            const bitRate = selectedSampleRate * 16; // 16 bits per sample for high quality
            
            // Enhanced MediaRecorder options based on format
            let streamingMediaRecorderOptions = {};
            
            if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                streamingMediaRecorderOptions = {
                    mimeType: 'audio/webm;codecs=opus',
                    audioBitsPerSecond: bitRate
                };
                addLog('INFO', '✓ Using WebM/Opus format (best compatibility)');
            } else if (MediaRecorder.isTypeSupported('audio/mp4;codecs=mp4a.40.2')) {
                streamingMediaRecorderOptions = {
                    mimeType: 'audio/mp4;codecs=mp4a.40.2',
                    audioBitsPerSecond: bitRate
                };
                addLog('INFO', '✓ Using MP4/AAC format (high compatibility)');
            } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                streamingMediaRecorderOptions = {
                    mimeType: 'audio/ogg;codecs=opus',
                    audioBitsPerSecond: bitRate
                };
                addLog('INFO', '✓ Using OGG/Opus format (good compatibility)');
            } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                streamingMediaRecorderOptions = {
                    mimeType: 'audio/webm',
                    audioBitsPerSecond: bitRate
                };
                addLog('INFO', '✓ Using basic WebM format');
            } else {
                streamingMediaRecorderOptions = { audioBitsPerSecond: bitRate };
                addLog('WARNING', 'Using default MediaRecorder format');
            }
            
            addLog('INFO', `Streaming with: ${selectedSampleRate}Hz, ${streamingMediaRecorderOptions.audioBitsPerSecond}bps`);
            addLog('INFO', `Format: ${streamingMediaRecorderOptions.mimeType || 'default'}`);
            
            state.chunkCount = 0;
            state.processingChunk = false;
            state.isStreaming = true;
            this.updateStreamingUI();
            
            // Start the complete recording loop with enhanced options
            this.startSimpleChunkLoop(streamingMediaRecorderOptions);
            
            addLog('SUCCESS', `Streaming started with ${streamingMediaRecorderOptions.mimeType || 'default'} (complete recordings)`);
            
        } catch (error) {
            addLog('ERROR', `Failed to start streaming: ${error.message}`);
            state.isStreaming = false;
            this.updateStreamingUI();
        }
    },

    async startSimpleChunkLoop(mediaRecorderOptions) {
        // Optimized for translation continuity - longer chunks for better sentence completion
        const recordingDuration = 4000; // 4 seconds for better Chinese sentence boundaries
        const sendInterval = 3000; // Send every 3 seconds for real-time feel
        const translationDelay = 500; // Small delay to allow sentence completion
        
        // Generate session ID for translation continuity
        const sessionId = `session_${Date.now()}`;
        state.translationSessionId = sessionId;
        
        addLog('INFO', `Starting translation session: ${sessionId}`);
        
        const startSequentialRecording = async () => {
            if (!state.isStreaming) return;
            
            try {
                const chunkNumber = ++state.chunkCount;
                const startTime = Date.now();
                addLog('INFO', `Recording chunk ${chunkNumber} (${recordingDuration/1000}s for Chinese sentence completion)...`);
                
                // Create new MediaRecorder for this chunk
                const recorder = new MediaRecorder(state.currentStream, mediaRecorderOptions);
                const chunks = [];
                
                recorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        chunks.push(event.data);
                    }
                };
                
                recorder.onstop = () => {
                    if (chunks.length > 0) {
                        const completeBlob = new Blob(chunks, { type: mediaRecorderOptions.mimeType || 'audio/webm' });
                        const recordingEnd = Date.now();
                        const actualDuration = (recordingEnd - startTime) / 1000;
                        const actualBitRate = (completeBlob.size * 8) / actualDuration;
                        addLog('INFO', `Chunk ${chunkNumber}: ${completeBlob.size} bytes, ${actualDuration.toFixed(1)}s, ${actualBitRate.toFixed(0)}bps`);
                        
                        // Process chunk with continuity management
                        this.processStreamingChunkWithContinuity(completeBlob, chunkNumber, sessionId);
                    }
                    
                    // Schedule next recording after a brief pause
                    if (state.isStreaming) {
                        setTimeout(startSequentialRecording, translationDelay);
                    }
                };
                
                // Start recording
                recorder.start();
                
                // Stop after duration 
                setTimeout(() => {
                    if (recorder.state === 'recording') {
                        recorder.stop();
                    }
                }, recordingDuration);
                
            } catch (error) {
                addLog('ERROR', `Recording chunk ${state.chunkCount} failed: ${error.message}`);
                if (state.isStreaming) {
                    setTimeout(startSequentialRecording, 1000); // Retry after 1 second
                }
            }
        };
        
        // Alternative: Use continuous streaming approach
        const startContinuousStreaming = () => {
            if (!state.isStreaming) return;
            
            try {
                addLog('INFO', 'Starting continuous audio streaming to match backend buffer...');
                
                // Create a single continuous recorder
                const recorder = new MediaRecorder(state.currentStream, { 
                    mimeType: mimeType,
                    audioBitsPerSecond: 128000 // Consistent bitrate
                });
                
                let chunkBuffer = [];
                let lastSendTime = Date.now();
                
                recorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        chunkBuffer.push(event.data);
                        
                        // Send data at inference intervals
                        const now = Date.now();
                        if (now - lastSendTime >= inferenceInterval) {
                            const chunkNumber = ++state.chunkCount;
                            const audioBlob = new Blob(chunkBuffer, { type: mimeType });
                            addLog('INFO', `Sending continuous chunk ${chunkNumber}: ${audioBlob.size} bytes`);
                            
                            this.processStreamingChunk(audioBlob, chunkNumber);
                            
                            // Keep overlap by not clearing entire buffer
                            const overlapChunks = Math.floor(chunkBuffer.length * (overlapDuration / inferenceInterval));
                            chunkBuffer = chunkBuffer.slice(-overlapChunks);
                            lastSendTime = now;
                        }
                    }
                };
                
                recorder.onstop = () => {
                    // Send any remaining data
                    if (chunkBuffer.length > 0) {
                        const finalBlob = new Blob(chunkBuffer, { type: mimeType });
                        this.processStreamingChunk(finalBlob, ++state.chunkCount);
                    }
                    addLog('INFO', 'Continuous streaming stopped');
                };
                
                // Request data every 100ms for smooth streaming
                recorder.start(100);
                state.mediaRecorder = recorder;
                
            } catch (error) {
                addLog('ERROR', `Continuous streaming failed: ${error.message}`);
            }
        };
        
        // Start sequential recording optimized for translation continuity
        startSequentialRecording();
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
            
            // Verify blob before sending
            if (blob.size === 0) {
                addLog('ERROR', `Chunk ${chunkNumber} is empty! Skipping.`);
                return;
            }
            
            // Detect actual format from blob content
            let filename = 'audio.webm'; // Default
            let contentType = blob.type || 'audio/webm';
            
            // Add blob content inspection (helpful for debugging)
            const reader = new FileReader();
            const arrayBufferPromise = new Promise((resolve) => {
                reader.onload = () => resolve(reader.result);
                reader.readAsArrayBuffer(blob.slice(0, 16));
            });
            
            const arrayBuffer = await arrayBufferPromise;
            const uint8Array = new Uint8Array(arrayBuffer);
            const firstBytes = Array.from(uint8Array.slice(0, 16)).map(b => b.toString(16).padStart(2, '0')).join(' ');
            addLog('DEBUG', `Chunk ${chunkNumber} first 16 bytes: ${firstBytes}`);
            
            // Detect format from magic bytes
            if (firstBytes.includes('1a 45 df a3')) {
                filename = 'audio.webm';
                contentType = 'audio/webm';
                addLog('DEBUG', `Detected WebM format for chunk ${chunkNumber}`);
            } else if (firstBytes.includes('4f 67 67 53')) {
                filename = 'audio.ogg';
                contentType = 'audio/ogg';
                addLog('DEBUG', `Detected OGG format for chunk ${chunkNumber}`);
            } else if (firstBytes.includes('52 49 46 46')) {
                filename = 'audio.wav';
                contentType = 'audio/wav';
                addLog('DEBUG', `Detected WAV format for chunk ${chunkNumber}`);
            } else if (firstBytes.includes('66 74 79 70')) {
                // Still detect MP4 but log a warning
                filename = 'audio.mp4';
                contentType = 'audio/mp4';
                addLog('WARNING', `Detected MP4 format for chunk ${chunkNumber} - backend may need ffmpeg`);
            } else {
                addLog('DEBUG', `Using blob type or default WebM format for chunk ${chunkNumber}`);
            }
            
            // Create FormData with proper debugging
            const formData = new FormData();
            formData.append('audio', blob, filename);
            
            // Verify FormData was created correctly
            const formDataEntries = Array.from(formData.entries());
            addLog('DEBUG', `FormData entries for chunk ${chunkNumber}: ${formDataEntries.length}`);
            formDataEntries.forEach(([key, value]) => {
                if (value instanceof File || value instanceof Blob) {
                    addLog('DEBUG', `  ${key}: ${value.name || 'blob'} (${value.size} bytes, ${value.type})`);
                } else {
                    addLog('DEBUG', `  ${key}: ${value}`);
                }
            });
            
            // Direct transcription through orchestration gateway
            const url = `${config.whisperApiUrl}/transcribe/${state.selectedModel}`;
            addLog('DEBUG', `Sending chunk ${chunkNumber} to: ${url}`);
            
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });
            
            const responseText = await response.text();
            addLog('DEBUG', `Chunk ${chunkNumber} response status: ${response.status}`);
            addLog('DEBUG', `Chunk ${chunkNumber} response text length: ${responseText.length}`);
            
            let data;
            try {
                data = JSON.parse(responseText);
            } catch (parseError) {
                addLog('ERROR', `Chunk ${chunkNumber} JSON parse error: ${parseError.message}`);
                addLog('ERROR', `Response preview: ${responseText.substring(0, 200)}...`);
                throw new Error('Invalid JSON response from server');
            }
            
            if (response.ok) {
                const transcribedText = data.text ? data.text.trim() : '';
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

    async processStreamingChunkWithContinuity(blob, chunkNumber, sessionId) {
        try {
            addLog('INFO', `Processing chunk ${chunkNumber} with continuity (session: ${sessionId}, ${(blob.size / 1024).toFixed(1)}KB)`);
            
            // Verify blob before sending
            if (blob.size === 0) {
                addLog('ERROR', `Chunk ${chunkNumber} is empty! Skipping.`);
                return;
            }
            
            // Create FormData for transcription
            const formData = new FormData();
            formData.append('file', blob, 'audio.webm');
            formData.append('language', 'auto');
            formData.append('task', 'transcribe');
            formData.append('chunk_id', `chunk_${chunkNumber}`);
            formData.append('session_id', sessionId);
            
            // Send for transcription first
            addLog('DEBUG', `Sending chunk ${chunkNumber} for transcription...`);
            const transcriptionResponse = await fetch('/api/whisper/transcribe/whisper-small.en', {
                method: 'POST',
                body: formData
            });
            
            if (!transcriptionResponse.ok) {
                throw new Error(`Transcription failed: ${transcriptionResponse.status} ${transcriptionResponse.statusText}`);
            }
            
            const transcriptionData = await transcriptionResponse.json();
            
            if (transcriptionData.text && transcriptionData.text.trim()) {
                const transcribedText = transcriptionData.text.trim();
                addLog('SUCCESS', `Chunk ${chunkNumber} transcribed: "${transcribedText}" (Lang: ${transcriptionData.language})`);
                
                // Send to translation service with continuity
                const translationResponse = await fetch('/api/translation/translate/continuity', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: transcribedText,
                        session_id: sessionId,
                        chunk_id: `chunk_${chunkNumber}`,
                        target_language: 'en',
                        source_language: transcriptionData.language || 'zh'
                    })
                });
                
                if (translationResponse.ok) {
                    const translationData = await translationResponse.json();
                    
                    if (translationData.status === 'translated') {
                        addLog('SUCCESS', `[Chunk ${chunkNumber}] Translated: "${translationData.translated_text}"`);
                        
                        // Display translation result
                        if (elements.translationResults) {
                            const resultDiv = document.createElement('div');
                            resultDiv.className = 'translation-result';
                            resultDiv.innerHTML = `
                                <div class="chunk-info">[Chunk ${chunkNumber}] Context: ${translationData.context_items} items</div>
                                <div class="source-text"><strong>Chinese:</strong> ${translationData.source_text}</div>
                                <div class="translated-text"><strong>English:</strong> ${translationData.translated_text}</div>
                                <div class="translation-meta">
                                    <span>Confidence: ${(translationData.confidence_score * 100).toFixed(1)}%</span>
                                    <span>Backend: ${translationData.backend_used}</span>
                                    <span>Time: ${translationData.processing_time.toFixed(2)}s</span>
                                </div>
                            `;
                            elements.translationResults.appendChild(resultDiv);
                            elements.translationResults.scrollTop = elements.translationResults.scrollHeight;
                        }
                        
                    } else if (translationData.status === 'buffering') {
                        addLog('INFO', `[Chunk ${chunkNumber}] Buffering for sentence completion (${translationData.buffer_length} chars)`);
                        
                    } else if (translationData.status === 'error') {
                        addLog('ERROR', `[Chunk ${chunkNumber}] Translation error: ${translationData.error}`);
                    }
                } else {
                    addLog('ERROR', `Translation request failed: ${translationResponse.status}`);
                }
                
            } else {
                addLog('INFO', `Chunk ${chunkNumber} produced no transcription or silence detected`);
            }
            
        } catch (error) {
            addLog('ERROR', `Chunk ${chunkNumber} continuity processing error: ${error.message}`);
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