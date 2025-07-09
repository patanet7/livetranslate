// API module for server communication
const API = {
    async checkServerHealth(silent = false) {
        try {
            // Check orchestration service health
            const response = await fetch(`${config.healthApiUrl}`);
            
            if (response.ok) {
                const data = await response.json();
                updateServerStatus('healthy', data);
                if (!silent) {
                    addLog('SUCCESS', 'Orchestration service health check passed');
                }
                return true;
            } else {
                throw new Error(`Server returned ${response.status}`);
            }
        } catch (error) {
            updateServerStatus('error', null);
            addLog('ERROR', `Orchestration service health check failed: ${error.message}`);
            return false;
        }
    },

    async loadModels() {
        try {
            console.log('API.loadModels: Fetching models from whisper service...');
            addLog('DEBUG', 'Fetching models from whisper service through orchestration gateway');
            
            // Use whisper API through orchestration gateway
            const response = await fetch(`${config.whisperApiUrl}/models`);
            const data = await response.json();
            
            console.log('API.loadModels: Server response:', data);
            console.log('API.loadModels: Response OK:', response.ok);
            addLog('DEBUG', `Models API response: ${JSON.stringify(data)}`);
            
            if (response.ok) {
                // Handle whisper service response format
                state.availableModels = data.available_models || data.models || [];
                console.log('API.loadModels: Set state.availableModels to:', state.availableModels);
                addLog('DEBUG', `Parsed ${state.availableModels.length} models: ${state.availableModels.join(', ')}`);
                
                UI.updateModelsDisplay(data);
                addLog('INFO', `Loaded ${state.availableModels.length} models from whisper service`);
            } else {
                throw new Error(`Failed to load models: ${response.status} - ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            addLog('ERROR', `Failed to load models: ${error.message}`);
            console.error('API.loadModels error:', error);
            state.availableModels = [];
            if (elements.modelsContent) {
                elements.modelsContent.innerHTML = '<div class="log-entry ERROR">Failed to load models</div>';
            }
        }
    },

    async transcribeAudio(audioBlob, modelName) {
        try {
            const finalModel = modelName || state.selectedModel;
            addLog('INFO', `Transcribing with whisper service using model: ${finalModel} (blob: ${(audioBlob.size/1024).toFixed(1)}KB)`);
            
            // Verify blob before sending
            if (audioBlob.size === 0) {
                throw new Error('Audio blob is empty');
            }
            
            // Use whisper API through orchestration gateway
            const url = finalModel ? 
                `${config.whisperApiUrl}/transcribe/${finalModel}` : 
                `${config.whisperApiUrl}/transcribe`;
            
            // Detect format and create proper filename
            let filename = 'audio.webm'; // Default
            if (audioBlob.type) {
                if (audioBlob.type.includes('mp4')) filename = 'audio.mp4';
                else if (audioBlob.type.includes('ogg')) filename = 'audio.ogg';
                else if (audioBlob.type.includes('wav')) filename = 'audio.wav';
                else if (audioBlob.type.includes('webm')) filename = 'audio.webm';
            }
            
            addLog('DEBUG', `Creating FormData: ${filename} (${audioBlob.type || 'no type'}, ${audioBlob.size} bytes)`);
            
            // Create FormData for proper file upload
            const formData = new FormData();
            formData.append('audio', audioBlob, filename);
            if (state.selectedLanguage) {
                formData.append('language', state.selectedLanguage);
            }
            
            // Verify FormData creation
            const entries = Array.from(formData.entries());
            addLog('DEBUG', `FormData has ${entries.length} entries`);
            entries.forEach(([key, value]) => {
                if (value instanceof Blob) {
                    addLog('DEBUG', `  ${key}: ${value.name || 'blob'} (${value.size} bytes, ${value.type})`);
                } else {
                    addLog('DEBUG', `  ${key}: ${value}`);
                }
            });
            
            // Debug: Test if blob can be read
            try {
                const arrayBuffer = await audioBlob.arrayBuffer();
                addLog('DEBUG', `Blob arrayBuffer size: ${arrayBuffer.byteLength} bytes`);
                const uint8Array = new Uint8Array(arrayBuffer);
                const firstBytes = Array.from(uint8Array.slice(0, 8)).map(b => b.toString(16).padStart(2, '0')).join(' ');
                addLog('DEBUG', `First 8 bytes: ${firstBytes}`);
            } catch (blobError) {
                addLog('ERROR', `Failed to read blob: ${blobError.message}`);
            }
            
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                if (data.text) {
                    addTranscription(data.text, true);
                    addLog('SUCCESS', `Transcription completed: "${data.text.substring(0, 50)}..."`);
                    
                    // Trigger translation if enabled
                    if (window.TranslationManager && window.TranslationManager.isEnabled) {
                        window.TranslationManager.handleTranscriptionForTranslation(data.text);
                    }
                } else if (data.message) {
                    addTranscription(data.message, false);
                    addLog('WARNING', data.message);
                }
                return data;
            } else {
                throw new Error(data.error || 'Transcription failed');
            }
        } catch (error) {
            addLog('ERROR', `Transcription failed: ${error.message}`);
            addTranscription(`Error: ${error.message}`, false);
            throw error;
        }
    },

    async transcribeFile(file, modelName) {
        try {
            const finalModel = modelName || state.selectedModel;
            addLog('INFO', `Transcribing uploaded file with whisper service: ${file.name} (${(file.size/1024/1024).toFixed(2)}MB) using model: ${finalModel}`);
            
            // Use whisper API through orchestration gateway
            const url = finalModel ? 
                `${config.whisperApiUrl}/transcribe/${finalModel}` : 
                `${config.whisperApiUrl}/transcribe`;
            
            // Create FormData for proper file upload
            const formData = new FormData();
            formData.append('audio', file);
            if (state.selectedLanguage) {
                formData.append('language', state.selectedLanguage);
            }
            
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                if (data.text) {
                    addTranscription(`[FILE: ${file.name}] ${data.text}`, true);
                    addLog('SUCCESS', `File transcription completed: "${data.text.substring(0, 50)}..."`);
                } else {
                    addTranscription(`[FILE: ${file.name}] No speech detected`, false);
                    addLog('INFO', 'File transcription completed - no speech detected');
                }
                return data;
            } else {
                throw new Error(data.error || 'File transcription failed');
            }
        } catch (error) {
            addLog('ERROR', `File transcription failed: ${error.message}`);
            throw error;
        }
    },

    async clearCache() {
        try {
            const response = await fetch(`${config.serverUrl}/clear-cache`, { method: 'POST' });
            const data = await response.json();
            
            if (response.ok) {
                addLog('SUCCESS', 'NPU cache cleared successfully');
                return data;
            } else {
                throw new Error(data.error || 'Failed to clear cache');
            }
        } catch (error) {
            addLog('ERROR', `Failed to clear cache: ${error.message}`);
            throw error;
        }
    }
};

// Global functions for backwards compatibility
window.checkServerHealth = API.checkServerHealth;
window.loadModels = API.loadModels;
window.transcribeAudio = API.transcribeAudio; 