// API module for server communication
const API = {
    async checkServerHealth(silent = false) {
        try {
            const response = await fetch(`${config.serverUrl}/health`);
            
            if (response.ok) {
                const data = await response.json();
                updateServerStatus('healthy', data);
                if (!silent) {
                    addLog('SUCCESS', 'Server health check passed');
                }
                return true;
            } else {
                throw new Error(`Server returned ${response.status}`);
            }
        } catch (error) {
            updateServerStatus('error', null);
            addLog('ERROR', `Server health check failed: ${error.message}`);
            return false;
        }
    },

    async loadModels() {
        try {
            console.log('API.loadModels: Fetching models from server...');
            const response = await fetch(`${config.serverUrl}/models`);
            const data = await response.json();
            
            console.log('API.loadModels: Server response:', data);
            console.log('API.loadModels: Response OK:', response.ok);
            
            if (response.ok) {
                state.availableModels = data.models || [];
                console.log('API.loadModels: Set state.availableModels to:', state.availableModels);
                
                UI.updateModelsDisplay(data);
                addLog('INFO', `Loaded ${state.availableModels.length} models`);
            } else {
                throw new Error(`Failed to load models: ${response.status}`);
            }
        } catch (error) {
            addLog('ERROR', `Failed to load models: ${error.message}`);
            state.availableModels = [];
            if (elements.modelsContent) {
                elements.modelsContent.innerHTML = '<div class="log-entry ERROR">Failed to load models</div>';
            }
        }
    },

    async transcribeAudio(audioBlob, modelName) {
        try {
            addLog('INFO', `Transcribing with model: ${modelName}`);
            
            const url = modelName ? 
                `${config.serverUrl}/transcribe/${modelName}` : 
                `${config.serverUrl}/transcribe`;
            
            const response = await fetch(url, {
                method: 'POST',
                body: audioBlob
            });
            
            const data = await response.json();
            
            if (response.ok) {
                if (data.text) {
                    addTranscription(data.text, true);
                    addLog('SUCCESS', `Transcription completed: "${data.text.substring(0, 50)}..."`);
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
            addLog('INFO', `Transcribing uploaded file: ${file.name} (${(file.size/1024/1024).toFixed(2)}MB)`);
            
            const url = modelName ? 
                `${config.serverUrl}/transcribe/${modelName}` : 
                `${config.serverUrl}/transcribe`;
            
            const response = await fetch(url, {
                method: 'POST',
                body: file
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